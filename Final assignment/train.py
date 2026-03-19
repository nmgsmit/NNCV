"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
import random
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Normalize,
    InterpolationMode
)

from model import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=131072, label_smoothing=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        pixel_losses = nn.functional.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        valid_mask = target != self.ignore_index
        valid_losses = pixel_losses[valid_mask]
        if valid_losses.numel() == 0:
            return pixel_losses.mean()

        with torch.no_grad():
            sorted_losses, _ = torch.sort(valid_losses, descending=True)
            if sorted_losses.numel() > self.min_kept:
                dynamic_thresh = sorted_losses[self.min_kept - 1]
                threshold = max(dynamic_thresh.item(), self.thresh)
            else:
                threshold = self.thresh

        hard_losses = valid_losses[valid_losses >= threshold]
        if hard_losses.numel() == 0:
            hard_losses = valid_losses
        return hard_losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        valid_mask = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        target_one_hot = nn.functional.one_hot(target_safe, num_classes=n_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).float()

        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
        denominator = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class CityscapesSegmentation(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str, crop_size=(256, 256), train: bool = False):
        self.dataset = Cityscapes(
            data_dir,
            split=split,
            mode="fine",
            target_type="semantic",
        )
        self.crop_size = crop_size
        self.train = train
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.dataset)

    def _random_rescale(self, image, label):
        scale = random.uniform(0.75, 1.5)
        height, width = image.height, image.width
        new_size = (max(1, int(height * scale)), max(1, int(width * scale)))
        image = TF.resize(image, new_size, interpolation=InterpolationMode.BILINEAR)
        label = TF.resize(label, new_size, interpolation=InterpolationMode.NEAREST)
        return image, label

    def _train_transform(self, image, label):
        image = TF.resize(image, self.crop_size, interpolation=InterpolationMode.BILINEAR)
        label = TF.resize(label, self.crop_size, interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        image = TF.to_tensor(image)
        image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
        image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
        image = TF.adjust_saturation(image, random.uniform(0.9, 1.1))
        image = self.normalize(image)

        label = torch.as_tensor(TF.pil_to_tensor(label), dtype=torch.int64)
        return image, label

    def _eval_transform(self, image, label):
        image = TF.resize(image, self.crop_size, interpolation=InterpolationMode.BILINEAR)
        label = TF.resize(label, self.crop_size, interpolation=InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = self.normalize(image)
        label = torch.as_tensor(TF.pil_to_tensor(label), dtype=torch.int64)
        return image, label

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.train:
            return self._train_transform(image, label)
        return self._eval_transform(image, label)


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate for poly decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--poly-power", type=float, default=0.9, help="Power for poly learning rate schedule")
    parser.add_argument("--ohem-thresh", type=float, default=0.7, help="OHEM threshold")
    parser.add_argument("--ohem-min-kept", type=int, default=131072, help="Minimum hard pixels for OHEM")
    parser.add_argument("--aux-weight", type=float, default=0.4, help="Weight for auxiliary OHEM loss")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Weight for dice loss")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for OHEM cross-entropy")
    parser.add_argument("--crop-size", type=int, default=256, help="Square train/validation image size")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping norm; 0 disables clipping")
    parser.add_argument("--early-stop-patience", type=int, default=6, help="Number of epochs without validation improvement before stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum validation improvement to reset early stopping")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="efficient DDRNET-23-slim", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset and make a split for training and validation
    crop_size = (args.crop_size, args.crop_size)
    train_dataset = CityscapesSegmentation(
        args.data_dir,
        split="train",
        crop_size=crop_size,
        train=True,
    )

    valid_dataset = CityscapesSegmentation(
        args.data_dir,
        split="val",
        crop_size=crop_size,
        train=False,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Define the loss function
    ohem_criterion = OHEMCrossEntropyLoss(
        ignore_index=255,
        thresh=args.ohem_thresh,
        min_kept=args.ohem_min_kept,
        label_smoothing=args.label_smoothing,
    )
    dice_criterion = DiceLoss(ignore_index=255)

    # Define the optimizer
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        verbose=True
    )

    best_valid_loss = float("inf")
    best_dice = -float('inf')
    epochs_without_improvement = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        train_losses_total = []
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels = labels.long().squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                main_logits, aux_logits = outputs
            else:
                main_logits, aux_logits = outputs, None
            loss_main = ohem_criterion(main_logits, labels)
            loss_aux = ohem_criterion(aux_logits, labels) if aux_logits is not None else torch.tensor(0.0, device=device)
            loss_dice = dice_criterion(main_logits, labels)
            loss = loss_main + args.aux_weight * loss_aux + args.dice_weight * loss_dice
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered; try lower lr or enable --skip-nonfinite-batches")
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_losses_total.append(loss.item())

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        # Validation
        model.eval()

        with torch.no_grad():
            valid_losses_total = []
            valid_dice_scores = []
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                labels = labels.long().squeeze(1)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss_main = ohem_criterion(outputs, labels)
                loss_dice = dice_criterion(outputs, labels)
                loss_total = loss_main + args.dice_weight * loss_dice
                valid_losses_total.append(loss_total.item())
                valid_dice_scores.append((1.0 - loss_dice.item()))
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)
                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)
                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)
                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()
                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            train_loss = sum(train_losses_total) / len(train_losses_total)
            valid_loss = sum(valid_losses_total) / len(valid_losses_total)
            valid_mean_dice = sum(valid_dice_scores) / len(valid_dice_scores)
            wandb.log({
                "valid_mean_dice": valid_mean_dice,
                "valid_loss": valid_loss,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
                "generalization_gap": valid_loss - train_loss,
            }, step=(epoch + 1) * len(train_dataloader) - 1)
            scheduler.step(valid_loss)
            if valid_mean_dice > best_dice + args.early_stop_min_delta:
                best_dice = valid_mean_dice
                best_valid_loss = valid_loss

                torch.save(model.state_dict(), best_model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1}: no Dice improvement for {args.early_stop_patience} epochs.")
                    break
    print("Training complete!")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch + 1:04}-best_val_loss={best_valid_loss:.4f}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
