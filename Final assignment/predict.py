"""
Prediction pipeline for the current experiment branch.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize, ToDtype, ToImage

from model import Model, infer_model_variant_from_state_dict


DEFAULT_IMAGE_DIR = Path("/data")
DEFAULT_OUTPUT_DIR = Path("/output")
DEFAULT_MODEL_PATH = Path("/app/model.pt")
DEFAULT_IMAGE_SIZE = (1024, 2048)
CITYSCAPES_MEAN = (0.485, 0.456, 0.406)
CITYSCAPES_STD = (0.229, 0.224, 0.225)
DEFAULT_TTA_SCALES = (0.75, 1.0, 1.25)
DEFAULT_TTA_FLIP = True
DEFAULT_USE_SEGFIX = True
FALLBACK_IMAGE_SIZES = (
    (768, 1536),
    (512, 1024),
)
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


@dataclass(frozen=True)
class InferenceProfile:
    name: str
    image_size: tuple[int, int]
    tta_scales: tuple[float, ...]
    tta_flip: bool
    use_segfix: bool


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Environment variable {name} must be a boolean flag, got '{value}'.")


def parse_number_tuple_env(name: str, default: tuple[int, int], cast: type[int] | type[float]) -> tuple[int, ...] | tuple[float, ...]:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a comma-separated list, got '{value}'.") from exc

    if len(parsed) == 0:
        raise ValueError(f"Environment variable {name} must contain at least one value.")
    return parsed


def get_image_size() -> tuple[int, int]:
    parsed = parse_number_tuple_env("PREDICT_IMAGE_SIZE", DEFAULT_IMAGE_SIZE, int)
    if len(parsed) != 2:
        raise ValueError("PREDICT_IMAGE_SIZE must be formatted as 'height,width'.")
    height, width = parsed
    if height <= 0 or width <= 0:
        raise ValueError("PREDICT_IMAGE_SIZE values must be positive integers.")
    return height, width


def get_tta_scales() -> tuple[float, ...]:
    parsed = parse_number_tuple_env("PREDICT_TTA_SCALES", DEFAULT_TTA_SCALES, float)
    if any(scale <= 0 for scale in parsed):
        raise ValueError("PREDICT_TTA_SCALES must contain positive numbers.")
    return parsed


def build_inference_profiles(
    image_size: tuple[int, int],
    tta_scales: tuple[float, ...],
    tta_flip: bool,
    use_segfix: bool,
) -> list[InferenceProfile]:
    profiles: list[InferenceProfile] = []
    seen_configs: set[tuple[tuple[int, int], tuple[float, ...], bool, bool]] = set()

    def add_profile(
        name: str,
        profile_image_size: tuple[int, int],
        profile_tta_scales: tuple[float, ...],
        profile_tta_flip: bool,
        profile_use_segfix: bool,
    ) -> None:
        config = (
            profile_image_size,
            profile_tta_scales,
            profile_tta_flip,
            profile_use_segfix,
        )
        if config in seen_configs:
            return
        seen_configs.add(config)
        profiles.append(
            InferenceProfile(
                name=name,
                image_size=profile_image_size,
                tta_scales=profile_tta_scales,
                tta_flip=profile_tta_flip,
                use_segfix=profile_use_segfix,
            )
        )

    add_profile("requested", image_size, tta_scales, tta_flip, use_segfix)
    add_profile("no-flip", image_size, tta_scales, False, use_segfix)
    add_profile("single-scale", image_size, (1.0,), False, False)

    for fallback_size in FALLBACK_IMAGE_SIZES:
        if fallback_size[0] >= image_size[0] or fallback_size[1] >= image_size[1]:
            continue
        add_profile(f"single-scale-{fallback_size[0]}x{fallback_size[1]}", fallback_size, (1.0,), False, False)

    return profiles


def resolve_device() -> str:
    requested_device = os.getenv("PREDICT_DEVICE", "auto").strip().lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("PREDICT_DEVICE must be one of: auto, cpu, cuda.")

    if requested_device == "cpu":
        return "cpu"

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PREDICT_DEVICE=cuda was requested, but CUDA is not available.")
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception as exc:
            raise RuntimeError("PREDICT_DEVICE=cuda was requested, but CUDA is not usable.") from exc

    if not torch.cuda.is_available():
        return "cpu"

    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except Exception as exc:
        print(f"CUDA was detected but is not usable ({exc}). Falling back to CPU.")
        return "cpu"


def preprocess(img: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=image_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
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


def multi_scale_inference(
    model: Model,
    image_tensor: torch.Tensor,
    amp_enabled: bool,
    tta_scales: tuple[float, ...],
    tta_flip: bool,
) -> torch.Tensor:
    target_size = image_tensor.shape[-2:]
    fused_logits = None
    num_predictions = 0

    for scale in tta_scales:
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

        if tta_flip:
            flipped_images = torch.flip(scaled_images, dims=[3])
            with autocast_context(amp_enabled):
                flipped_logits = model(flipped_images)
            flipped_logits = torch.flip(flipped_logits, dims=[3])
            flipped_logits = resize_logits(flipped_logits.float(), target_size)
            fused_logits = fused_logits + flipped_logits
            num_predictions += 1

    return fused_logits / max(num_predictions, 1)


def is_cuda_oom(error: RuntimeError) -> bool:
    oom_types = (torch.OutOfMemoryError,)
    if isinstance(error, oom_types):
        return True
    return "out of memory" in str(error).lower()


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


def save_prediction(prediction: np.ndarray, out_path: Path) -> None:
    prediction_uint8 = np.ascontiguousarray(prediction.astype(np.uint8))
    Image.fromarray(prediction_uint8, mode="L").save(out_path)


def run_profile(
    model: Model,
    img: Image.Image,
    original_shape: tuple[int, int],
    device: str,
    amp_enabled: bool,
    profile: InferenceProfile,
) -> np.ndarray:
    img_tensor = preprocess(img, image_size=profile.image_size).to(device)
    pred = multi_scale_inference(
        model,
        img_tensor,
        amp_enabled=amp_enabled,
        tta_scales=profile.tta_scales,
        tta_flip=profile.tta_flip,
    )
    if profile.use_segfix:
        prediction = segfix_style_refine(pred)
    else:
        prediction = pred.argmax(dim=1)
    return postprocess(resize_prediction(prediction, original_shape))


def predict_with_fallback(
    model: Model,
    img: Image.Image,
    original_shape: tuple[int, int],
    device: str,
    amp_enabled: bool,
    profiles: list[InferenceProfile],
    start_index: int,
) -> tuple[np.ndarray, int]:
    for profile_index in range(start_index, len(profiles)):
        profile = profiles[profile_index]
        print(
            f"Trying inference profile '{profile.name}' "
            f"(image_size={profile.image_size}, scales={profile.tta_scales}, "
            f"flip={profile.tta_flip}, segfix={profile.use_segfix})."
        )
        try:
            return run_profile(
                model,
                img,
                original_shape=original_shape,
                device=device,
                amp_enabled=amp_enabled,
                profile=profile,
            ), profile_index
        except RuntimeError as exc:
            if device != "cuda" or not is_cuda_oom(exc):
                raise
            print(f"CUDA OOM while using profile '{profile.name}'. Retrying with a lighter profile.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError("All inference profiles failed due to CUDA OOM.")


def main() -> None:
    image_dir = Path(os.getenv("IMAGE_DIR", DEFAULT_IMAGE_DIR.as_posix()))
    output_dir = Path(os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR.as_posix()))
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH.as_posix()))
    image_size = get_image_size()
    tta_scales = get_tta_scales()
    tta_flip = parse_bool_env("PREDICT_TTA_FLIP", DEFAULT_TTA_FLIP)
    use_segfix = parse_bool_env("PREDICT_USE_SEGFIX", DEFAULT_USE_SEGFIX)

    device = resolve_device()
    amp_enabled = device == "cuda"

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model_variant = infer_model_variant_from_state_dict(state_dict)
    print(f"Loaded SegFormer checkpoint as variant '{model_variant}'.")
    print(f"Running prediction on device='{device}' with requested image_size={image_size}.")

    model = Model(variant=model_variant)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = sorted(image_dir.glob("*.png"))
    profiles = build_inference_profiles(
        image_size=image_size,
        tta_scales=tta_scales,
        tta_flip=tta_flip,
        use_segfix=use_segfix,
    )
    active_profile_index = 0
    print(f"Found {len(image_files)} images to process.")
    print(f"Using multi-scale TTA with scales={tta_scales} and flip={tta_flip}.")
    print(f"SegFix-style boundary refinement enabled: {use_segfix}.")
    if device == "cuda":
        print("CUDA OOM fallback is enabled. The script will retry with lighter inference settings if needed.")

    with torch.inference_mode():
        for index, img_path in enumerate(image_files, start=1):
            print(f"[{index}/{len(image_files)}] Processing {img_path.name}...")
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                original_shape = (img.height, img.width)
                seg_pred, used_profile_index = predict_with_fallback(
                    model,
                    img,
                    original_shape=original_shape,
                    device=device,
                    amp_enabled=amp_enabled,
                    profiles=profiles,
                    start_index=active_profile_index,
                )
            if used_profile_index != active_profile_index:
                active_profile_index = used_profile_index
                print(f"Switching remaining images to profile '{profiles[active_profile_index].name}'.")

            out_path = output_dir / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_prediction(seg_pred, out_path)
            unique_labels = np.unique(seg_pred)
            print(
                f"Saved prediction to {out_path} "
                f"(shape={seg_pred.shape}, labels={unique_labels.tolist()}, dtype={seg_pred.dtype})."
            )
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
