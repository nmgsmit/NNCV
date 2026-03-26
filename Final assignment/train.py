import os
import random
from argparse import ArgumentParser
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
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
MAX_EPOCHS = 50
BASE_LR = 6e-5
WEIGHT_DECAY = 1e-2
WARMUP_ITERS = 1500
POLY_POWER = 0.9
MIN_LR_RATIO = 1e-3
HFLIP_PROB = 0.5
COLOR_JITTER = 0.4
GAUSSIAN_BLUR = 0.0
EMA_DECAY = 0.999
EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-4
EVAL_SCALES = (0.75, 1.0, 1.25)
EVAL_FLIP = True
DROPOUT = 0.1


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


class CityscapesJointTransform:
    def __init__(self, training: bool):
        self.training = training

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

        image = self.color_jitter(image)
        if GAUSSIAN_BLUR > 0.0 and random.random() < GAUSSIAN_BLUR:
            image = F_v2.gaussian_blur(image, kernel_size=3)

        image = ToDtype(torch.float32, scale=True)(image)
        image = Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD)(image)
        target = ToDtype(torch.int64)(ToImage()(target))
        return image, target


class CityscapesSegmentationDataset(Dataset):
    def __init__(self, root: str, split: str, transform: CityscapesJointTransform):
        self.dataset = Cityscapes(root, split=split, mode="fine", target_type="semantic")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return self.transform(image, target)


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
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
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
            "weight_decay": WEIGHT_DECAY,
            "warmup_iters": WARMUP_ITERS,
            "poly_power": POLY_POWER,
            "min_lr_ratio": MIN_LR_RATIO,
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
        },
    )

    train_transform = CityscapesJointTransform(training=True)
    valid_transform = CityscapesJointTransform(training=False)
    train_dataset = CityscapesSegmentationDataset(args.data_dir, split="train", transform=train_transform)
    valid_dataset = CityscapesSegmentationDataset(args.data_dir, split="val", transform=valid_transform)

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
        batch_size=args.batch_size,
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

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
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

    for epoch in range(MAX_EPOCHS):
        print(f"Epoch {epoch + 1:04}/{MAX_EPOCHS:04}")

        model.train()
        train_losses = []
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
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                outputs = model(images)
                outputs = resize_logits(outputs, labels.shape[-2:])
                loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered; try lowering the learning rate.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema_model(ema_model, model, EMA_DECAY)

            train_losses.append(loss.item())
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )
            global_step += 1

        eval_model = ema_model
        eval_model.eval()

        with torch.no_grad():
            valid_losses = []
            confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

            for batch_index, (images, labels) in enumerate(valid_dataloader):
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
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())
                update_confusion_matrix(confusion_matrix, outputs, labels, num_classes=NUM_CLASSES)

                if batch_index == 0:
                    predictions = outputs.softmax(1).argmax(1, keepdim=True)
                    labels_vis = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions.cpu())
                    labels_vis = convert_train_id_to_color(labels_vis.cpu())

                    predictions_img = make_grid(predictions, nrow=8).permute(1, 2, 0).numpy()
                    labels_img = make_grid(labels_vis, nrow=8).permute(1, 2, 0).numpy()

                    wandb.log(
                        {
                            "predictions": [wandb.Image(predictions_img)],
                            "labels": [wandb.Image(labels_img)],
                        },
                        step=max(global_step - 1, 0),
                    )

            train_loss = sum(train_losses) / max(len(train_losses), 1)
            valid_loss = sum(valid_losses) / max(len(valid_losses), 1)
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
                    "valid_loss": valid_loss,
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
