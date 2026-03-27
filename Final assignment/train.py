import os
import random
from argparse import ArgumentParser
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage
from torchvision.transforms.v2 import functional as F_v2
from torchvision.utils import make_grid

from model import MODEL_DESCRIPTION, MODEL_NAME, MODEL_VARIANTS, Model


IGNORE_INDEX = 255
NUM_CLASSES = 19
CITYSCAPES_MEAN = (0.485, 0.456, 0.406)
CITYSCAPES_STD = (0.229, 0.224, 0.225)
BASE_IMAGE_SIZE = (1024, 2048)
TRAIN_CROP_SIZE = (1024, 1024)
TRAIN_RATIO_RANGE = (0.5, 2.0)
TRAIN_CAT_MAX_RATIO = 0.75
DEFAULT_BATCH_SIZE = 1
MAX_EPOCHS = 50
BASE_LR = 6e-5
WEIGHT_DECAY = 1e-2
WARMUP_ITERS = 1500
POLY_POWER = 0.9
MIN_LR_RATIO = 1e-3
HEAD_LR_MULTIPLIER = 10.0
HFLIP_PROB = 0.5
COLOR_JITTER = 0.4
GAUSSIAN_BLUR = 0.0
EMA_DECAY = 0.999
EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-4
EVAL_SCALES = (0.75, 1.0, 1.25)
EVAL_FLIP = True
DROPOUT = 0.1
LOVASZ_LOSS_WEIGHT = 1.0
WEATHER_AUG_PROB = 0.6
DOMAIN_SHIFT_PROB = 0.4
SHADOW_PROB = 0.25
HOLDOUT_FILENAMES = (
    "tubingen_000047_000019_leftImg8bit.png",
    "tubingen_000063_000019_leftImg8bit.png",
    "tubingen_000126_000019_leftImg8bit.png",
    "tubingen_000138_000019_leftImg8bit.png",
)


id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
id_to_trainid[IGNORE_INDEX] = IGNORE_INDEX
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != IGNORE_INDEX}
train_id_to_color[IGNORE_INDEX] = (0, 0, 0)


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid.get(int(x), IGNORE_INDEX))


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = prediction.shape
    color_image = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for channel in range(3):
            color_image[:, channel][mask] = color[channel]

    return color_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 8
    return min(cpu_count, 12)


def autocast_context(enabled: bool):
    if enabled:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def resize_logits(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def sample_class_balanced_crop(
    target: torch.Tensor,
    crop_size: tuple[int, int],
    ignore_index: int = IGNORE_INDEX,
    cat_max_ratio: float = TRAIN_CAT_MAX_RATIO,
    num_classes: int = NUM_CLASSES,
    max_attempts: int = 10,
) -> tuple[int, int]:
    crop_height, crop_width = crop_size
    image_height, image_width = target.shape[-2:]
    max_top = max(image_height - crop_height, 0)
    max_left = max(image_width - crop_width, 0)

    top = 0
    left = 0
    for _ in range(max_attempts):
        top = 0 if max_top == 0 else random.randint(0, max_top)
        left = 0 if max_left == 0 else random.randint(0, max_left)

        current_height = min(crop_height, image_height)
        current_width = min(crop_width, image_width)
        crop = TF.crop(target, top, left, current_height, current_width).to(torch.int64).squeeze(0)
        valid = crop != ignore_index
        if not valid.any():
            return top, left

        class_histogram = torch.bincount(crop[valid].flatten(), minlength=num_classes)
        total = class_histogram.sum()
        if total == 0 or (class_histogram.max().float() / total.float()) < cat_max_ratio:
            return top, left

    return top, left


def lovasz_gradient(sorted_ground_truth: torch.Tensor) -> torch.Tensor:
    total_positive = sorted_ground_truth.sum()
    intersection = total_positive - sorted_ground_truth.cumsum(dim=0)
    union = total_positive + (1.0 - sorted_ground_truth).cumsum(dim=0)
    jaccard = 1.0 - intersection / union
    if sorted_ground_truth.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def flatten_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, probabilities.shape[1])
    labels = labels.reshape(-1)
    if ignore_index is None:
        return probabilities, labels

    valid = labels != ignore_index
    return probabilities[valid], labels[valid]


def lovasz_softmax_flat(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if probabilities.numel() == 0:
        return probabilities.sum() * 0.0

    losses = []
    num_classes = probabilities.shape[1]
    for class_index in range(num_classes):
        foreground = (labels == class_index).float()
        if foreground.sum() == 0:
            continue

        class_predictions = probabilities[:, class_index]
        errors = (foreground - class_predictions).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, lovasz_gradient(foreground_sorted)))

    if not losses:
        return probabilities.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    probabilities, labels = flatten_probabilities(probabilities, labels, ignore_index=ignore_index)
    return lovasz_softmax_flat(probabilities, labels)


def poly_lr(
    base_lr: float,
    current_iter: int,
    max_iter: int,
    warmup_iters: int,
    power: float,
    min_lr_ratio: float,
) -> float:
    if max_iter <= 0:
        return base_lr

    if warmup_iters > 0 and current_iter < warmup_iters:
        return base_lr * float(current_iter + 1) / float(max(warmup_iters, 1))

    poly_start_iter = min(max(warmup_iters, 0), max_iter)
    poly_total_iters = max(max_iter - poly_start_iter, 1)
    poly_progress = min(max((current_iter - poly_start_iter) / poly_total_iters, 0.0), 1.0)
    min_lr = base_lr * min_lr_ratio
    return max(base_lr * ((1.0 - poly_progress) ** power), min_lr)


def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        for name, ema_param in ema_model.named_parameters():
            ema_param.mul_(decay).add_(model_params[name], alpha=1.0 - decay)

        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_model.named_buffers():
            ema_buffer.copy_(model_buffers[name])


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    prediction = logits.argmax(dim=1)
    valid = target != ignore_index
    if not valid.any():
        return

    target = target[valid]
    prediction = prediction[valid]
    indices = target * num_classes + prediction
    confusion_matrix += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_mean_iou(confusion_matrix: torch.Tensor) -> float:
    confusion_matrix = confusion_matrix.float()
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
    valid = union > 0
    if not valid.any():
        return 0.0
    return (intersection[valid] / union[valid]).mean().item()


def compute_mean_dice(confusion_matrix: torch.Tensor) -> float:
    confusion_matrix = confusion_matrix.float()
    true_positives = torch.diag(confusion_matrix)
    false_positives = confusion_matrix.sum(dim=0) - true_positives
    false_negatives = confusion_matrix.sum(dim=1) - true_positives

    denominator = 2.0 * true_positives + false_positives + false_negatives
    valid = denominator > 0
    if not valid.any():
        return 0.0

    return ((2.0 * true_positives[valid]) / denominator[valid]).mean().item()


class RobustnessAugmentor:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < WEATHER_AUG_PROB:
            image = random.choice(
                [self.apply_fog, self.apply_rain, self.apply_snow, self.apply_low_light]
            )(image)

        if random.random() < DOMAIN_SHIFT_PROB:
            image = self.apply_domain_shift(image)

        if random.random() < SHADOW_PROB:
            image = self.apply_shadow(image)

        return image.clamp(0.0, 1.0)

    def smooth_noise(self, image: torch.Tensor, scale: int = 16) -> torch.Tensor:
        height, width = image.shape[-2:]
        coarse_height = max(2, height // scale)
        coarse_width = max(2, width // scale)
        noise = torch.rand((1, 1, coarse_height, coarse_width), dtype=image.dtype)
        noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
        noise = F.avg_pool2d(noise, kernel_size=5, stride=1, padding=2)
        return noise.squeeze(0)

    def apply_fog(self, image: torch.Tensor) -> torch.Tensor:
        haze = self.smooth_noise(image, scale=20)
        strength = random.uniform(0.2, 0.45)
        fog = 0.75 + 0.25 * haze
        image = image * (1.0 - strength) + fog.expand_as(image) * strength
        image = TF.adjust_contrast(image, random.uniform(0.6, 0.85))
        return image

    def apply_low_light(self, image: torch.Tensor) -> torch.Tensor:
        gamma = random.uniform(1.4, 2.4)
        brightness = random.uniform(0.35, 0.7)
        sensor_noise = torch.randn_like(image) * random.uniform(0.01, 0.03)
        image = TF.adjust_gamma(image, gamma=gamma)
        image = image * brightness + sensor_noise
        return image

    def apply_rain(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        rain_mask = (torch.rand((1, 1, height, width), dtype=image.dtype) > 0.992).float()
        kernel = torch.zeros((1, 1, 9, 9), dtype=image.dtype)
        if random.random() < 0.5:
            for index in range(9):
                kernel[0, 0, index, index] = 1.0
        else:
            for index in range(9):
                kernel[0, 0, index, 8 - index] = 1.0

        streaks = F.conv2d(rain_mask, kernel / kernel.sum(), padding=4)
        streaks = F.avg_pool2d(streaks, kernel_size=3, stride=1, padding=1).squeeze(0)
        strength = random.uniform(0.15, 0.3)
        image = TF.adjust_brightness(image, random.uniform(0.75, 0.95))
        image = TF.adjust_contrast(image, random.uniform(0.7, 0.9))
        image = image + streaks.expand_as(image) * strength
        return image

    def apply_snow(self, image: torch.Tensor) -> torch.Tensor:
        snow = self.smooth_noise(image, scale=10)
        snow = (snow > snow.mean() + 0.35 * snow.std()).float()
        snow = F.avg_pool2d(snow.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        haze_strength = random.uniform(0.05, 0.15)
        snow_strength = random.uniform(0.15, 0.3)
        image = TF.adjust_saturation(image, random.uniform(0.7, 0.95))
        image = image * (1.0 - haze_strength) + haze_strength
        image = image + snow.expand_as(image) * snow_strength
        return image

    def apply_domain_shift(self, image: torch.Tensor) -> torch.Tensor:
        channel_gain = torch.empty((image.shape[0], 1, 1), dtype=image.dtype).uniform_(0.8, 1.2)
        channel_bias = torch.empty((image.shape[0], 1, 1), dtype=image.dtype).uniform_(-0.08, 0.08)
        image = image * channel_gain + channel_bias
        image = TF.adjust_contrast(image, random.uniform(0.7, 1.35))
        image = TF.adjust_saturation(image, random.uniform(0.6, 1.4))
        image = TF.adjust_hue(image, random.uniform(-0.08, 0.08))
        return image

    def apply_shadow(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        y_coords = torch.linspace(0.0, 1.0, steps=height, dtype=image.dtype).view(1, height, 1)
        x_coords = torch.linspace(0.0, 1.0, steps=width, dtype=image.dtype).view(1, 1, width)
        mask = y_coords.expand(1, height, width) if random.random() < 0.5 else x_coords.expand(1, height, width)
        if random.random() < 0.5:
            mask = 1.0 - mask
        shadow_strength = random.uniform(0.2, 0.45)
        shadow = 1.0 - shadow_strength * mask
        return image * shadow


class CityscapesJointTransform:
    def __init__(self, training: bool):
        self.training = training
        self.robustness_augmentor = RobustnessAugmentor()

        self.eval_image_transform = Compose([
            ToImage(),
            Resize(BASE_IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True),
            ToDtype(torch.float32, scale=True),
            Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD),
        ])
        self.eval_target_transform = Compose([
            ToImage(),
            Resize(BASE_IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
            ToDtype(torch.int64),
        ])
        self.color_jitter = v2.ColorJitter(
            brightness=COLOR_JITTER,
            contrast=COLOR_JITTER,
            saturation=COLOR_JITTER,
            hue=min(0.1, COLOR_JITTER),
        )

    def __call__(self, image, target):
        if not self.training:
            image = self.eval_image_transform(image)
            target = self.eval_target_transform(target)
            return image, target

        scale = random.uniform(*TRAIN_RATIO_RANGE)
        scaled_size = (
            max(1, int(round(BASE_IMAGE_SIZE[0] * scale))),
            max(1, int(round(BASE_IMAGE_SIZE[1] * scale))),
        )

        image = ToImage()(image)
        target = ToImage()(target)
        image = TF.resize(image, scaled_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        target = TF.resize(target, scaled_size, interpolation=InterpolationMode.NEAREST)

        top, left = sample_class_balanced_crop(
            target,
            crop_size=TRAIN_CROP_SIZE,
            ignore_index=IGNORE_INDEX,
            cat_max_ratio=TRAIN_CAT_MAX_RATIO,
        )
        current_height = min(TRAIN_CROP_SIZE[0], image.shape[-2])
        current_width = min(TRAIN_CROP_SIZE[1], image.shape[-1])
        image = TF.crop(image, top, left, current_height, current_width)
        target = TF.crop(target, top, left, current_height, current_width)

        pad_height = max(TRAIN_CROP_SIZE[0] - image.shape[-2], 0)
        pad_width = max(TRAIN_CROP_SIZE[1] - image.shape[-1], 0)
        if pad_height > 0 or pad_width > 0:
            padding = [0, 0, pad_width, pad_height]
            image = TF.pad(image, padding, fill=0)
            target = TF.pad(target, padding, fill=IGNORE_INDEX)

        if random.random() < HFLIP_PROB:
            image = TF.hflip(image)
            target = TF.hflip(target)

        image = ToDtype(torch.float32, scale=True)(image)
        image = self.color_jitter(image)
        image = self.robustness_augmentor(image)
        if GAUSSIAN_BLUR > 0.0 and random.random() < GAUSSIAN_BLUR:
            image = F_v2.gaussian_blur(image, kernel_size=3)

        image = Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD)(image)
        target = ToDtype(torch.int64)(ToImage()(target))
        return image, target


class CityscapesSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: CityscapesJointTransform,
        include_filenames: set[str] | None = None,
        exclude_filenames: set[str] | None = None,
    ):
        self.dataset = Cityscapes(root, split=split, mode="fine", target_type="semantic")
        self.transform = transform
        self.image_paths = list(self.dataset.images)
        self.target_paths = [target[0] for target in self.dataset.targets]

        if include_filenames is not None or exclude_filenames is not None:
            filtered_indices = []
            for index, image_path in enumerate(self.image_paths):
                filename = os.path.basename(image_path)
                if include_filenames is not None and filename not in include_filenames:
                    continue
                if exclude_filenames is not None and filename in exclude_filenames:
                    continue
                filtered_indices.append(index)

            self.dataset.images = [self.dataset.images[index] for index in filtered_indices]
            self.dataset.targets = [self.dataset.targets[index] for index in filtered_indices]
            self.image_paths = [self.image_paths[index] for index in filtered_indices]
            self.target_paths = [self.target_paths[index] for index in filtered_indices]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return self.transform(image, target)

    def filenames(self) -> list[str]:
        return [os.path.basename(path) for path in self.image_paths]


def multi_scale_inference(
    model: nn.Module,
    images: torch.Tensor,
    scales: tuple[float, ...],
    flip: bool,
    amp_enabled: bool,
) -> torch.Tensor:
    target_size = images.shape[-2:]
    fused_logits = None
    num_predictions = 0

    for scale in scales:
        if scale == 1.0:
            scaled_images = images
        else:
            scaled_size = (
                max(1, int(round(target_size[0] * scale))),
                max(1, int(round(target_size[1] * scale))),
            )
            scaled_images = F.interpolate(images, size=scaled_size, mode="bilinear", align_corners=False)

        with autocast_context(amp_enabled):
            logits = model(scaled_images)
        logits = resize_logits(logits.float(), target_size)
        fused_logits = logits if fused_logits is None else fused_logits + logits
        num_predictions += 1

        if flip:
            flipped_images = torch.flip(scaled_images, dims=[3])
            with autocast_context(amp_enabled):
                flipped_logits = model(flipped_images)
            flipped_logits = torch.flip(flipped_logits, dims=[3])
            flipped_logits = resize_logits(flipped_logits.float(), target_size)
            fused_logits = fused_logits + flipped_logits
            num_predictions += 1

    return fused_logits / max(num_predictions, 1)


def get_args_parser() -> ArgumentParser:
    parser = ArgumentParser("Training script for the SegFormer progression experiments")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=default_num_workers(), help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default=None, help="Optional W&B run name override")
    parser.add_argument("--model-variant", type=str, default="b5", choices=tuple(MODEL_VARIANTS), help="Pretrained SegFormer backbone variant")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable CUDA AMP")
    parser.set_defaults(amp=True)
    return parser


def main(args) -> None:
    experiment_id = args.experiment_id or f"{MODEL_NAME}-{args.model_variant}"
    output_dir = os.path.join("checkpoints", experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    if args.model_variant == "b5" and TRAIN_CROP_SIZE == (1024, 1024) and args.batch_size > 1:
        print(
            "Warning: SegFormer-B5 with 1024x1024 crops usually requires batch_size=1 on a 40GB GPU. "
            "If you hit CUDA OOM, rerun with --batch-size 1."
        )

    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and torch.cuda.is_available()

    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=experiment_id,
        config={
            "model_name": MODEL_NAME,
            "model_description": MODEL_DESCRIPTION,
            "model_variant": args.model_variant,
            "base_image_size": BASE_IMAGE_SIZE,
            "train_crop_size": TRAIN_CROP_SIZE,
            "train_ratio_range": TRAIN_RATIO_RANGE,
            "train_cat_max_ratio": TRAIN_CAT_MAX_RATIO,
            "max_epochs": MAX_EPOCHS,
            "base_lr": BASE_LR,
            "head_lr_multiplier": HEAD_LR_MULTIPLIER,
            "weight_decay": WEIGHT_DECAY,
            "warmup_iters": WARMUP_ITERS,
            "poly_power": POLY_POWER,
            "min_lr_ratio": MIN_LR_RATIO,
            "lovasz_loss_weight": LOVASZ_LOSS_WEIGHT,
            "weather_aug_prob": WEATHER_AUG_PROB,
            "domain_shift_prob": DOMAIN_SHIFT_PROB,
            "shadow_prob": SHADOW_PROB,
            "ema_decay": EMA_DECAY,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
            "eval_scales": EVAL_SCALES,
            "eval_flip": EVAL_FLIP,
            "dropout": DROPOUT,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "amp": amp_enabled,
            "holdout_filenames": HOLDOUT_FILENAMES,
        },
    )

    train_transform = CityscapesJointTransform(training=True)
    valid_transform = CityscapesJointTransform(training=False)
    holdout_filenames = set(HOLDOUT_FILENAMES)
    base_train_dataset = CityscapesSegmentationDataset(args.data_dir, split="train", transform=train_transform)
    extra_train_dataset = CityscapesSegmentationDataset(
        args.data_dir,
        split="val",
        transform=train_transform,
        exclude_filenames=holdout_filenames,
    )
    valid_dataset = CityscapesSegmentationDataset(
        args.data_dir,
        split="val",
        transform=valid_transform,
        include_filenames=holdout_filenames,
    )
    train_dataset = ConcatDataset([base_train_dataset, extra_train_dataset])

    missing_holdouts = sorted(holdout_filenames - set(valid_dataset.filenames()))
    if missing_holdouts:
        missing = ", ".join(missing_holdouts)
        raise RuntimeError(f"Could not find all fixed holdout validation images in the Cityscapes val split: {missing}")

    persistent_workers = args.num_workers > 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
    )

    model = Model(in_channels=3, n_classes=NUM_CLASSES, variant=args.model_variant, dropout=DROPOUT)
    pretrained_path = f"./mit-{args.model_variant}"
    pretrained_loaded = False
    try:
        model.load_pretrained(pretrained_path)
        pretrained_loaded = True
    except Exception as exc:
        print(f"Warning: Could not load pretrained weights from {pretrained_path}. {exc}")
        print("Continuing with random initialization.")

    model = model.to(device)
    ema_model = Model(in_channels=3, n_classes=NUM_CLASSES, variant=args.model_variant, dropout=DROPOUT).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad = False

    ce_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("decode_head."):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": BASE_LR},
            {"params": head_parameters, "lr": BASE_LR * HEAD_LR_MULTIPLIER},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_valid_loss = float("inf")
    best_miou = -float("inf")
    best_dice = -float("inf")
    current_best_model_path = None
    epochs_without_improvement = 0
    global_step = 0
    total_iters = MAX_EPOCHS * max(len(train_dataloader), 1)

    print(f"Training {MODEL_NAME} ({args.model_variant})")
    print(MODEL_DESCRIPTION)
    print(f"Pretrained MiT loaded: {pretrained_loaded}")
    print(f"Training on {len(train_dataset)} images: {len(base_train_dataset)} from train + {len(extra_train_dataset)} from val.")
    print(f"Using {len(valid_dataset)} fixed holdout validation images: {', '.join(valid_dataset.filenames())}")

    for epoch in range(MAX_EPOCHS):
        print(f"Epoch {epoch + 1:04}/{MAX_EPOCHS:04}")

        model.train()
        train_losses = []
        train_ce_losses = []
        train_lovasz_losses = []
        for images, labels in train_dataloader:
            labels = convert_to_train_id(labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().squeeze(1)

            current_lr = poly_lr(
                base_lr=BASE_LR,
                current_iter=global_step,
                max_iter=total_iters,
                warmup_iters=WARMUP_ITERS,
                power=POLY_POWER,
                min_lr_ratio=MIN_LR_RATIO,
            )
            optimizer.param_groups[0]["lr"] = current_lr
            optimizer.param_groups[1]["lr"] = current_lr * HEAD_LR_MULTIPLIER

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                outputs = model(images)
                outputs = resize_logits(outputs, labels.shape[-2:])
                ce_loss = ce_criterion(outputs, labels)
            lovasz_loss = lovasz_softmax_loss(outputs.float(), labels)
            loss = ce_loss + LOVASZ_LOSS_WEIGHT * lovasz_loss

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered; try lowering the learning rate.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema_model(ema_model, model, EMA_DECAY)

            train_losses.append(loss.item())
            train_ce_losses.append(ce_loss.item())
            train_lovasz_losses.append(lovasz_loss.item())
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_ce_loss": ce_loss.item(),
                    "train_lovasz_loss": lovasz_loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "head_learning_rate": optimizer.param_groups[1]["lr"],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )
            global_step += 1

        eval_model = ema_model
        eval_model.eval()

        with torch.no_grad():
            valid_losses = []
            valid_ce_losses = []
            valid_lovasz_losses = []
            confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
            preview_predictions = []
            preview_labels = []

            for images, labels in valid_dataloader:
                labels = convert_to_train_id(labels)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long().squeeze(1)

                outputs = multi_scale_inference(
                    eval_model,
                    images,
                    scales=EVAL_SCALES,
                    flip=EVAL_FLIP,
                    amp_enabled=amp_enabled,
                )
                ce_loss = ce_criterion(outputs, labels)
                lovasz_loss = lovasz_softmax_loss(outputs.float(), labels)
                loss = ce_loss + LOVASZ_LOSS_WEIGHT * lovasz_loss
                valid_losses.append(loss.item())
                valid_ce_losses.append(ce_loss.item())
                valid_lovasz_losses.append(lovasz_loss.item())
                update_confusion_matrix(confusion_matrix, outputs, labels, num_classes=NUM_CLASSES)

                predictions = outputs.softmax(1).argmax(1, keepdim=True)
                labels_vis = labels.unsqueeze(1)

                preview_predictions.append(convert_train_id_to_color(predictions.cpu()))
                preview_labels.append(convert_train_id_to_color(labels_vis.cpu()))

            if preview_predictions:
                predictions = torch.cat(preview_predictions, dim=0)
                labels_vis = torch.cat(preview_labels, dim=0)
                grid_columns = min(2, predictions.shape[0])

                predictions_img = make_grid(predictions, nrow=grid_columns).permute(1, 2, 0).numpy()
                labels_img = make_grid(labels_vis, nrow=grid_columns).permute(1, 2, 0).numpy()

                wandb.log(
                    {
                        "predictions": [wandb.Image(predictions_img, caption="Fixed holdout predictions")],
                        "labels": [wandb.Image(labels_img, caption="Fixed holdout labels")],
                    },
                    step=max(global_step - 1, 0),
                )

            train_loss = sum(train_losses) / max(len(train_losses), 1)
            train_ce_loss = sum(train_ce_losses) / max(len(train_ce_losses), 1)
            train_lovasz_loss = sum(train_lovasz_losses) / max(len(train_lovasz_losses), 1)
            valid_loss = sum(valid_losses) / max(len(valid_losses), 1)
            valid_ce_loss = sum(valid_ce_losses) / max(len(valid_ce_losses), 1)
            valid_lovasz_loss = sum(valid_lovasz_losses) / max(len(valid_lovasz_losses), 1)
            valid_miou = compute_mean_iou(confusion_matrix)
            valid_mean_dice = compute_mean_dice(confusion_matrix)
            improved = valid_mean_dice > best_dice + EARLY_STOP_MIN_DELTA

            if improved:
                best_dice = valid_mean_dice
                best_miou = valid_miou
                best_valid_loss = valid_loss
                epochs_without_improvement = 0

                if current_best_model_path and os.path.exists(current_best_model_path):
                    os.remove(current_best_model_path)

                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-dice={valid_mean_dice:.4f}-miou={valid_miou:.4f}-val_loss={valid_loss:.4f}.pt",
                )
                torch.save(ema_model.state_dict(), current_best_model_path)
            else:
                epochs_without_improvement += 1

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "epoch_train_loss": train_loss,
                    "epoch_train_ce_loss": train_ce_loss,
                    "epoch_train_lovasz_loss": train_lovasz_loss,
                    "valid_loss": valid_loss,
                    "valid_ce_loss": valid_ce_loss,
                    "valid_lovasz_loss": valid_lovasz_loss,
                    "valid_miou": valid_miou,
                    "valid_mean_dice": valid_mean_dice,
                    "best_valid_mean_dice": best_dice,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                step=max(global_step - 1, 0),
            )

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"no mean Dice improvement for {EARLY_STOP_PATIENCE} epochs."
                )
                break

    print("Training complete!")

    final_model_path = os.path.join(
        output_dir,
        f"final_model-epoch={epoch:04}-dice={best_dice:.4f}-miou={best_miou:.4f}-val_loss={best_valid_loss:.4f}.pt",
    )
    torch.save(ema_model.state_dict(), final_model_path)
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
