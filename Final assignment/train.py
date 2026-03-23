"""
Training script for SegFormer on Cityscapes.
"""

import copy
import os
import random
from argparse import ArgumentParser
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F_v2
from torchvision.utils import make_grid

from model import Model


IGNORE_INDEX = 255
DEFAULT_TRAIN_SIZE = (512, 1024)
CITYSCAPES_MEAN = (0.485, 0.456, 0.406)
CITYSCAPES_STD = (0.229, 0.224, 0.225)
MODEL_VARIANTS = ("b0", "b5")


id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != IGNORE_INDEX}
train_id_to_color[IGNORE_INDEX] = (0, 0, 0)


def _as_tensor_image(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image
    return TF.pil_to_tensor(image)


def _as_float_image(image) -> torch.Tensor:
    image = _as_tensor_image(image)
    return image.float().div_(255.0)


def _as_long_target(target) -> torch.Tensor:
    target = _as_tensor_image(target)
    return target.to(dtype=torch.int64)


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


class CityscapesJointTransform:
    def __init__(
        self,
        image_size=DEFAULT_TRAIN_SIZE,
        training=True,
        hflip_prob=0.5,
        color_jitter=0.4,
        gaussian_blur=0.0,
    ):
        self.image_size = image_size
        self.training = training
        self.hflip_prob = hflip_prob
        self.gaussian_blur = gaussian_blur

        self.color_jitter = None
        if training and color_jitter > 0:
            self.color_jitter = v2.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(0.1, color_jitter),
            )

    def __call__(self, image, target):
        if self.training:
            i, j, h, w = v2.RandomResizedCrop.get_params(
                image,
                scale=(0.5, 2.0),
                ratio=(1.5, 2.0),
            )
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

            image = _as_tensor_image(image)
            if self.color_jitter is not None:
                image = self.color_jitter(image)
            if self.gaussian_blur > 0.0 and random.random() < self.gaussian_blur:
                image = F_v2.gaussian_blur(image, kernel_size=3)
        else:
            image = TF.resize(
                image,
                self.image_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            target = TF.resize(
                target,
                self.image_size,
                interpolation=InterpolationMode.NEAREST,
            )
            image = _as_tensor_image(image)

        image = _as_float_image(image)
        image = TF.normalize(image, mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD)

        target = _as_long_target(target)
        return image, target


class CityscapesSegmentationDataset(Dataset):
    def __init__(self, root, split, transform):
        self.dataset = Cityscapes(
            root,
            split=split,
            mode="fine",
            target_type="semantic",
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return self.transform(image, target)


def poly_lr(
    base_lr: float,
    current_iter: int,
    max_iter: int,
    warmup_iters: int = 1500,
    power: float = 0.9,
    min_lr_ratio: float = 0.0,
) -> float:
    if max_iter <= 0:
        return base_lr

    if warmup_iters > 0 and current_iter < warmup_iters:
        warmup_scale = float(current_iter + 1) / float(max(warmup_iters, 1))
        return base_lr * warmup_scale

    poly_start_iter = min(max(warmup_iters, 0), max_iter)
    poly_total_iters = max(max_iter - poly_start_iter, 1)
    poly_progress = min(max((current_iter - poly_start_iter) / poly_total_iters, 0.0), 1.0)

    decayed_lr = base_lr * ((1.0 - poly_progress) ** power)
    min_lr = base_lr * min_lr_ratio
    return max(decayed_lr, min_lr)


def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float):
    with torch.no_grad():
        model_state = dict(model.named_parameters())
        for name, ema_param in ema_model.named_parameters():
            model_param = model_state[name]
            ema_param.mul_(decay).add_(model_param, alpha=1.0 - decay)

        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_model.named_buffers():
            ema_buffer.copy_(model_buffers[name])


def update_confusion_matrix(confmat, logits, target, num_classes, ignore_index=IGNORE_INDEX):
    pred = logits.argmax(dim=1)
    valid = target != ignore_index
    if not valid.any():
        return

    target = target[valid]
    pred = pred[valid]
    indices = target * num_classes + pred
    confmat += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_mean_iou(confmat):
    confmat = confmat.float()
    intersection = torch.diag(confmat)
    union = confmat.sum(dim=1) + confmat.sum(dim=0) - intersection
    valid = union > 0
    if not valid.any():
        return 0.0
    iou = intersection[valid] / union[valid]
    return iou.mean().item()


def compute_mean_dice(confmat):
    confmat = confmat.float()
    true_positives = torch.diag(confmat)
    false_positives = confmat.sum(dim=0) - true_positives
    false_negatives = confmat.sum(dim=1) - true_positives

    denominator = 2.0 * true_positives + false_positives + false_negatives
    valid = denominator > 0
    if not valid.any():
        return 0.0

    dice = (2.0 * true_positives[valid]) / denominator[valid]
    return dice.mean().item()


def parse_eval_scales(scales) -> tuple[float, ...]:
    values = scales if isinstance(scales, (list, tuple)) else str(scales).split(",")
    parsed = []
    for value in values:
        scale = float(str(value).strip())
        if scale <= 0.0:
            raise ValueError(f"Evaluation scales must be positive, got {scale}.")
        parsed.append(scale)

    if not parsed:
        raise ValueError("At least one evaluation scale must be provided.")
    return tuple(parsed)


def autocast_context(enabled: bool):
    if enabled:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def forward_main_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    return outputs[0] if isinstance(outputs, tuple) else outputs


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
            scaled_images = F.interpolate(
                images,
                size=scaled_size,
                mode="bilinear",
                align_corners=False,
            )

        with autocast_context(amp_enabled):
            logits = forward_main_logits(model, scaled_images)
        logits = F.interpolate(logits.float(), size=target_size, mode="bilinear", align_corners=False)

        fused_logits = logits if fused_logits is None else fused_logits + logits
        num_predictions += 1

        if flip:
            flipped_images = torch.flip(scaled_images, dims=[3])
            with autocast_context(amp_enabled):
                flipped_logits = forward_main_logits(model, flipped_images)
            flipped_logits = torch.flip(flipped_logits, dims=[3])
            flipped_logits = F.interpolate(
                flipped_logits.float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            fused_logits = fused_logits + flipped_logits
            num_predictions += 1

    return fused_logits / max(num_predictions, 1)


def get_args_parser():
    parser = ArgumentParser("Training script for SegFormer on Cityscapes")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument(
        "--model-variant",
        type=str,
        default="b5",
        choices=MODEL_VARIANTS,
        help="SegFormer backbone preset to use",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help="Path to pretrained weights folder; defaults to ./mit-<model-variant>",
    )
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"], help="Optimizer type")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--base-batch-size", type=int, default=4, help="Reference batch size used for LR scaling")
    parser.add_argument("--scale-lr-with-batch", action="store_true", help="Scale LR linearly with batch size")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=6e-5, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup-iters", type=int, default=1500, help="Linear warmup iterations")
    parser.add_argument("--poly-power", type=float, default=0.9, help="Power for poly learning rate schedule")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0, help="Minimum LR as a ratio of base LR")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Number of epochs without validation improvement before stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum validation improvement to reset early stopping")
    parser.add_argument("--hflip-prob", type=float, default=0.5, help="Horizontal flip probability for training augmentation")
    parser.add_argument("--color-jitter", type=float, default=0.4, help="Color jitter strength (0 disables)")
    parser.add_argument("--gaussian-blur", type=float, default=0.0, help="Probability of Gaussian blur (0 disables)")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for evaluation model (<=0 disables EMA)")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval-scales", type=str, default="0.75,1.0,1.25", help="Comma-separated multi-scale validation factors")
    parser.add_argument("--eval-flip", dest="eval_flip", action="store_true", help="Enable horizontal flip during validation TTA")
    parser.add_argument("--no-eval-flip", dest="eval_flip", action="store_false", help="Disable horizontal flip during validation TTA")
    parser.set_defaults(eval_flip=True)
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable CUDA AMP training/inference")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable CUDA AMP training/inference")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID for Weights & Biases; defaults to segformer-<model-variant>",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Decoder dropout rate for SegFormer")
    return parser


def resolve_pretrained_path(args):
    if args.pretrained_path is not None:
        return args.pretrained_path
    return f"./mit-{args.model_variant}"


def build_model_from_variant(variant: str, dropout: float) -> Model:
    if variant == "b0":
        return Model(
            in_channels=3,
            n_classes=19,
            embed_dims=(32, 64, 160, 256),
            depths=(2, 2, 2, 2),
            sr_ratios=(8, 4, 2, 1),
            num_heads=(1, 2, 5, 8),
            decoder_embedding_dim=256,
            dropout=dropout,
            drop_path_rate=0.1,
        )
    if variant == "b5":
        return Model(
            in_channels=3,
            n_classes=19,
            embed_dims=(64, 128, 320, 512),
            depths=(3, 6, 40, 3),
            sr_ratios=(8, 4, 2, 1),
            num_heads=(1, 2, 5, 8),
            decoder_embedding_dim=768,
            dropout=dropout,
            drop_path_rate=0.1,
        )
    raise ValueError(f"Unknown model variant '{variant}'. Expected one of: {', '.join(MODEL_VARIANTS)}")


def main(args):
    pretrained_path = resolve_pretrained_path(args)
    eval_scales = parse_eval_scales(args.eval_scales)
    experiment_id = args.experiment_id or f"segformer-{args.model_variant}-ce-amp"
    amp_enabled = args.amp and torch.cuda.is_available()

    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=experiment_id,
        config={
            **vars(args),
            "eval_scales": list(eval_scales),
            "amp_enabled": amp_enabled,
        },
    )

    output_dir = os.path.join("checkpoints", experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = CityscapesJointTransform(
        image_size=DEFAULT_TRAIN_SIZE,
        training=True,
        hflip_prob=args.hflip_prob,
        color_jitter=args.color_jitter,
        gaussian_blur=args.gaussian_blur,
    )
    valid_transform = CityscapesJointTransform(
        image_size=DEFAULT_TRAIN_SIZE,
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

    try:
        model.load_pretrained(pretrained_path)
    except Exception as e:
        print(f"Warning: Could not load pretrained weights. {e}")
        print("Training from scratch...")

    ema_model = None
    if args.ema_decay > 0.0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    lr_scale = 1.0
    if args.scale_lr_with_batch:
        lr_scale = float(args.batch_size) / float(max(args.base_batch_size, 1))
    effective_base_lr = args.lr * lr_scale

    if args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=effective_base_lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=effective_base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    global_iter = 0
    total_iters = args.epochs * len(train_dataloader)

    best_dice = -float("inf")
    best_miou = -float("inf")
    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

        model.train()
        train_loss_total = 0.0

        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().squeeze(1)

            current_lr = poly_lr(
                base_lr=effective_base_lr,
                current_iter=global_iter,
                max_iter=total_iters,
                warmup_iters=args.warmup_iters,
                power=args.poly_power,
                min_lr_ratio=args.min_lr_ratio,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered; try lowering the learning rate.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)

            train_loss_total += loss.item()
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch + 1,
                },
                step=global_iter,
            )
            global_iter += 1

        model.eval()
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()

        with torch.no_grad():
            valid_losses_total = []
            confmat = torch.zeros((19, 19), dtype=torch.int64, device=device)

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
                valid_losses_total.append(loss.item())
                update_confusion_matrix(confmat, outputs, labels, num_classes=19)

                if i == 0:
                    predictions = outputs.argmax(1, keepdim=True)
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
                        step=global_iter - 1,
                    )

            valid_loss = sum(valid_losses_total) / len(valid_losses_total)
            valid_mean_dice = compute_mean_dice(confmat)
            valid_miou = compute_mean_iou(confmat)
            train_loss = train_loss_total / len(train_dataloader)

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "epoch_train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_mean_dice": valid_mean_dice,
                    "valid_miou": valid_miou,
                },
                step=global_iter - 1,
            )

            if valid_mean_dice > best_dice + args.early_stop_min_delta:
                best_dice = valid_mean_dice
                best_miou = valid_miou
                best_valid_loss = valid_loss
                checkpoint_source = ema_model if ema_model is not None else model
                torch.save(checkpoint_source.state_dict(), best_model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stop_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no Dice improvement for {args.early_stop_patience} epochs."
                    )
                    break

    print("Training complete!")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-best_dice={best_dice:.4f}-best_miou={best_miou:.4f}-best_val_loss={best_valid_loss:.4f}.pt",
        ),
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
