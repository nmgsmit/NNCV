"""
Prediction pipeline for the current experiment branch.
"""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize, ToDtype, ToImage

from model import Model, infer_model_variant_from_state_dict


IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"
IMAGE_SIZE = (1024, 2048)
CITYSCAPES_MEAN = (0.485, 0.456, 0.406)
CITYSCAPES_STD = (0.229, 0.224, 0.225)
TTA_SCALES = (1.0, 1.25)
TTA_FLIP = False
SEGFIX_RADII = (1, 2, 4)
SEGFIX_SCORE_MARGIN = 0.05
SEGFIX_BOUNDARY_DILATION = 1
SEGFIX_DIRECTIONS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD),
    ])
    return transform(img).unsqueeze(0)


def autocast_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def resize_logits(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def multi_scale_inference(model: Model, image_tensor: torch.Tensor, amp_enabled: bool) -> torch.Tensor:
    target_size = image_tensor.shape[-2:]
    fused_logits = None
    num_predictions = 0

    for scale in TTA_SCALES:
        if scale == 1.0:
            scaled_images = image_tensor
        else:
            scaled_size = (
                max(1, int(round(target_size[0] * scale))),
                max(1, int(round(target_size[1] * scale))),
            )
            scaled_images = F.interpolate(image_tensor, size=scaled_size, mode="bilinear", align_corners=False)

        with autocast_context(amp_enabled):
            logits = model(scaled_images)
        logits = resize_logits(logits.float(), target_size)
        fused_logits = logits if fused_logits is None else fused_logits + logits
        num_predictions += 1

        if TTA_FLIP:
            flipped_images = torch.flip(scaled_images, dims=[3])
            with autocast_context(amp_enabled):
                flipped_logits = model(flipped_images)
            flipped_logits = torch.flip(flipped_logits, dims=[3])
            flipped_logits = resize_logits(flipped_logits.float(), target_size)
            fused_logits = fused_logits + flipped_logits
            num_predictions += 1

    return fused_logits / max(num_predictions, 1)


def shift_tensor(x: torch.Tensor, dy: int, dx: int, fill_value: int | float) -> torch.Tensor:
    shifted = torch.full_like(x, fill_value)
    source_y_start = max(-dy, 0)
    source_y_end = x.shape[-2] - max(dy, 0)
    source_x_start = max(-dx, 0)
    source_x_end = x.shape[-1] - max(dx, 0)

    target_y_start = max(dy, 0)
    target_y_end = x.shape[-2] - max(-dy, 0)
    target_x_start = max(dx, 0)
    target_x_end = x.shape[-1] - max(-dx, 0)

    shifted[..., target_y_start:target_y_end, target_x_start:target_x_end] = x[
        ..., source_y_start:source_y_end, source_x_start:source_x_end
    ]
    return shifted


def compute_boundary_mask(prediction: torch.Tensor) -> torch.Tensor:
    boundary_mask = torch.zeros_like(prediction, dtype=torch.bool)
    boundary_mask[:, 1:, :] |= prediction[:, 1:, :] != prediction[:, :-1, :]
    boundary_mask[:, :-1, :] |= prediction[:, :-1, :] != prediction[:, 1:, :]
    boundary_mask[:, :, 1:] |= prediction[:, :, 1:] != prediction[:, :, :-1]
    boundary_mask[:, :, :-1] |= prediction[:, :, :-1] != prediction[:, :, 1:]

    if SEGFIX_BOUNDARY_DILATION > 0:
        kernel_size = 2 * SEGFIX_BOUNDARY_DILATION + 1
        boundary_mask = F.max_pool2d(
            boundary_mask.float().unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=SEGFIX_BOUNDARY_DILATION,
        ).squeeze(1) > 0

    return boundary_mask


def local_agreement_score(prediction: torch.Tensor) -> torch.Tensor:
    agreement = torch.zeros_like(prediction, dtype=torch.float32)
    for dy, dx in SEGFIX_DIRECTIONS:
        shifted_prediction = shift_tensor(prediction, dy, dx, fill_value=-1)
        agreement += (shifted_prediction == prediction).float()
    return agreement / float(len(SEGFIX_DIRECTIONS))


def segfix_style_refine(logits: torch.Tensor) -> torch.Tensor:
    # True SegFix needs extra learned boundary and direction predictions. This branch keeps
    # the current checkpoint untouched and applies an inference-only refinement that follows
    # the same core idea: replace uncertain boundary labels with stronger nearby interior ones.
    probabilities = logits.softmax(dim=1)
    prediction = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    boundary_mask = compute_boundary_mask(prediction)
    agreement_score = local_agreement_score(prediction)
    interior_score = confidence + agreement_score

    refined_prediction = prediction.clone()
    best_score = interior_score.clone()

    for radius in SEGFIX_RADII:
        for dy, dx in SEGFIX_DIRECTIONS:
            shifted_prediction = shift_tensor(prediction, dy * radius, dx * radius, fill_value=-1)
            shifted_score = shift_tensor(interior_score, dy * radius, dx * radius, fill_value=-1.0)
            shifted_boundary = shift_tensor(boundary_mask.to(torch.int64), dy * radius, dx * radius, fill_value=1).bool()

            better_candidate = (
                boundary_mask
                & (shifted_prediction >= 0)
                & (~shifted_boundary)
                & (shifted_score > best_score + SEGFIX_SCORE_MARGIN)
            )
            refined_prediction[better_candidate] = shifted_prediction[better_candidate]
            best_score[better_candidate] = shifted_score[better_candidate]

    return refined_prediction


def resize_prediction(prediction: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    resized_prediction = F.interpolate(
        prediction.unsqueeze(1).float(),
        size=target_size,
        mode="nearest",
    )
    return resized_prediction.squeeze(1).to(torch.int64)


def postprocess(prediction: torch.Tensor) -> np.ndarray:
    return prediction.squeeze(0).cpu().detach().numpy()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = torch.cuda.is_available()

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model_variant = infer_model_variant_from_state_dict(state_dict)
    print(f"Loaded SegFormer checkpoint as variant '{model_variant}'.")

    model = Model(variant=model_variant)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")
    print(f"Using multi-scale TTA with scales={TTA_SCALES} and flip={TTA_FLIP}.")
    print("Applying SegFix-style boundary refinement after TTA.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            pred = multi_scale_inference(model, img_tensor, amp_enabled=amp_enabled)
            refined_prediction = segfix_style_refine(pred)
            seg_pred = postprocess(resize_prediction(refined_prediction, original_shape))

            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
