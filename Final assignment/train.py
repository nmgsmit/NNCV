import copy
import os
import random
from argparse import ArgumentParser
from contextlib import nullcontext

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
)
from torchvision.transforms.v2 import functional as F_v2

from model import Model


IGNORE_INDEX = 255
NUM_CLASSES = 19
CITYSCAPES_MEAN = (0.485, 0.456, 0.406)
CITYSCAPES_STD = (0.229, 0.224, 0.225)
DEFAULT_TRAIN_SIZE = (512, 1024)

MODEL_CONFIGS = {
    "b0": {
        "embed_dims": (32, 64, 160, 256),
        "depths": (2, 2, 2, 2),
        "sr_ratios": (8, 4, 2, 1),
        "num_heads": (1, 2, 5, 8),
        "decoder_embedding_dim": 256,
    },
    "b5": {
        "embed_dims": (64, 128, 320, 512),
        "depths": (3, 6, 40, 3),
        "sr_ratios": (8, 4, 2, 1),
        "num_heads": (1, 2, 5, 8),
        "decoder_embedding_dim": 768,
    },
}


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != IGNORE_INDEX}
train_id_to_color[IGNORE_INDEX] = (0, 0, 0)


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for channel in range(3):
            color_image[:, channel][mask] = color[channel]

    return color_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 8
    return min(cpu_count, 12)


def parse_eval_scales(scales: str) -> tuple[float, ...]:
    parsed = []
    for value in str(scales).split(","):
        scale = float(value.strip())
        if scale <= 0.0:
            raise ValueError(f"Evaluation scales must be positive, got {scale}.")
        parsed.append(scale)
    return tuple(parsed)


def autocast_context(enabled: bool):
    if enabled:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def resize_logits(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


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
    def __init__(
        self,
        image_size: tuple[int, int],
        training: bool,
        hflip_prob: float,
        color_jitter: float,
        gaussian_blur: float,
    ):
        self.image_size = image_size
        self.training = training
        self.hflip_prob = hflip_prob
        self.gaussian_blur = gaussian_blur

        self.eval_image_transform = Compose([
            ToImage(),
            Resize(image_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            ToDtype(torch.float32, scale=True),
            Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD),
        ])
        self.eval_target_transform = Compose([
            ToImage(),
            Resize(image_size, interpolation=InterpolationMode.NEAREST),
            ToDtype(torch.int64),
        ])

        self.color_jitter = None
        if training and color_jitter > 0.0:
            self.color_jitter = v2.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(0.1, color_jitter),
            )

    def __call__(self, image, target):
        if not self.training:
            image = self.eval_image_transform(image)
            target = self.eval_target_transform(target)
            return image, target

        i, j, h, w = v2.RandomResizedCrop.get_params(image, scale=(0.5, 2.0), ratio=(1.5, 2.0))
        image = TF.resized_crop(
            image,
            i,
            j,
            h,
            w,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        target = TF.resized_crop(
            target,
            i,
            j,
            h,
            w,
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )

        if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            target = TF.hflip(target)

        image = ToImage()(image)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        if self.gaussian_blur > 0.0 and random.random() < self.gaussian_blur:
            image = F_v2.gaussian_blur(image, kernel_size=3)

        image = ToDtype(torch.float32, scale=True)(image)
        image = Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD)(image)
        target = ToDtype(torch.int64)(ToImage()(target))
        return image, target


class CityscapesSegmentationDataset(Dataset):
    def __init__(self, root: str, split: str, transform: CityscapesJointTransform):
        self.dataset = Cityscapes(root, split=split, mode="fine", target_type="semantic")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return self.transform(image, target)


def build_model_from_variant(variant: str, dropout: float) -> Model:
    try:
        config = MODEL_CONFIGS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown model variant '{variant}'. Expected one of: {', '.join(MODEL_CONFIGS)}") from exc

    return Model(
        in_channels=3,
        n_classes=NUM_CLASSES,
        dropout=dropout,
        drop_path_rate=0.1,
        **config,
    )


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


def get_args_parser():
    parser = ArgumentParser("Training script for SegFormer on Cityscapes")

    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup-iters", type=int, default=1500, help="Linear warmup iterations")
    parser.add_argument("--poly-power", type=float, default=0.9, help="Power for poly learning rate schedule")
    parser.add_argument("--min-lr-ratio", type=float, default=1e-3, help="Minimum LR as a ratio of base LR")
    parser.add_argument("--num-workers", type=int, default=default_num_workers(), help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID for Weights & Biases")

    parser.add_argument("--model-variant", type=str, default="b5", choices=tuple(MODEL_CONFIGS), help="SegFormer backbone preset to use")
    parser.add_argument("--dropout", type=float, default=0.1, help="Decoder dropout rate")
    parser.add_argument("--train-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=DEFAULT_TRAIN_SIZE, help="Training and validation resize/crop size")
    parser.add_argument("--hflip-prob", type=float, default=0.5, help="Horizontal flip probability for training augmentation")
    parser.add_argument("--color-jitter", type=float, default=0.4, help="Color jitter strength")
    parser.add_argument("--gaussian-blur", type=float, default=0.0, help="Probability of Gaussian blur")

    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for evaluation model; set <= 0 to disable")
    parser.add_argument("--early-stop-patience", type=int, default=12, help="Epochs without improvement before stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum Dice improvement to reset early stopping")

    parser.add_argument("--eval-scales", type=str, default="0.75,1.0,1.25", help="Comma-separated multi-scale validation factors")
    parser.add_argument("--eval-flip", dest="eval_flip", action="store_true", help="Enable horizontal flip during validation TTA")
    parser.add_argument("--no-eval-flip", dest="eval_flip", action="store_false", help="Disable horizontal flip during validation TTA")
    parser.set_defaults(eval_flip=True)

    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable CUDA AMP")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable CUDA AMP")
    parser.set_defaults(amp=True)

    return parser


def main(args):
    experiment_id = args.experiment_id or f"segformer-{args.model_variant}"
    output_dir = os.path.join("checkpoints", experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and torch.cuda.is_available()
    eval_scales = parse_eval_scales(args.eval_scales)

    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=experiment_id,
        config=vars(args),
    )

    train_transform = CityscapesJointTransform(
        image_size=tuple(args.train_size),
        training=True,
        hflip_prob=args.hflip_prob,
        color_jitter=args.color_jitter,
        gaussian_blur=args.gaussian_blur,
    )
    valid_transform = CityscapesJointTransform(
        image_size=tuple(args.train_size),
        training=False,
        hflip_prob=0.0,
        color_jitter=0.0,
        gaussian_blur=0.0,
    )

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

    model = build_model_from_variant(args.model_variant, args.dropout).to(device)

    pretrained_path = f"./mit-{args.model_variant}"
    try:
        model.load_pretrained(pretrained_path)
    except Exception as exc:
        print(f"Warning: Could not load pretrained weights from {pretrained_path}. {exc}")
        print("Training from scratch...")

    ema_model = None
    if args.ema_decay > 0.0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad = False

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_valid_loss = float("inf")
    best_miou = -float("inf")
    best_dice = -float("inf")
    current_best_model_path = None
    epochs_without_improvement = 0
    global_step = 0
    total_iters = args.epochs * max(len(train_dataloader), 1)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

        model.train()
        train_losses = []
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().squeeze(1)

            current_lr = poly_lr(
                base_lr=args.lr,
                current_iter=global_step,
                max_iter=total_iters,
                warmup_iters=args.warmup_iters,
                power=args.poly_power,
                min_lr_ratio=args.min_lr_ratio,
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

            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)

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

        model.eval()
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()

        with torch.no_grad():
            valid_losses = []
            confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long().squeeze(1)

                outputs = multi_scale_inference(
                    eval_model,
                    images,
                    scales=eval_scales,
                    flip=args.eval_flip,
                    amp_enabled=amp_enabled,
                )
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())
                update_confusion_matrix(confusion_matrix, outputs, labels, num_classes=NUM_CLASSES)

                if i == 0:
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

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "epoch_train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_miou": valid_miou,
                    "valid_mean_dice": valid_mean_dice,
                },
                step=max(global_step - 1, 0),
            )

            improved = valid_mean_dice > best_dice + args.early_stop_min_delta
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
                checkpoint_source = ema_model if ema_model is not None else model
                torch.save(checkpoint_source.state_dict(), current_best_model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stop_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no Dice improvement for {args.early_stop_patience} epochs."
                    )
                    break

    print("Training complete!")

    final_model_path = os.path.join(
        output_dir,
        f"final_model-epoch={epoch:04}-dice={best_dice:.4f}-miou={best_miou:.4f}-val_loss={best_valid_loss:.4f}.pt",
    )
    checkpoint_source = ema_model if ema_model is not None else model
    torch.save(checkpoint_source.state_dict(), final_model_path)
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
