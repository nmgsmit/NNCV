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
TTA_SCALES = (0.75, 1.0, 1.25)
TTA_FLIP = True


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


def postprocess(pred: torch.Tensor, original_shape: tuple[int, int]) -> np.ndarray:
    logits = F.interpolate(pred, size=original_shape, mode="bilinear", align_corners=False)
    prediction = logits.argmax(dim=1)
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

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            pred = multi_scale_inference(model, img_tensor, amp_enabled=amp_enabled)
            seg_pred = postprocess(pred, original_shape)

            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
