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
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
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
        probs = torch.softmax(logits, dim=1)

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


class IoULoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        valid_mask = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        target_one_hot = nn.functional.one_hot(target_safe, num_classes=n_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).float()

        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return 1.0 - iou.mean()


def update_confusion_matrix(conf_mat: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, n_classes: int, ignore_index: int = 255):
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return
    indices = target * n_classes + pred
    bincount = torch.bincount(indices, minlength=n_classes * n_classes)
    conf_mat += bincount.reshape(n_classes, n_classes)


def compute_mean_iou_and_dice(conf_mat: torch.Tensor, eps: float = 1e-6):
    tp = torch.diag(conf_mat).float()
    fp = conf_mat.sum(dim=0).float() - tp
    fn = conf_mat.sum(dim=1).float() - tp

    denom_iou = tp + fp + fn
    valid_iou = denom_iou > 0
    iou = torch.zeros_like(tp)
    iou[valid_iou] = tp[valid_iou] / (denom_iou[valid_iou] + eps)

    denom_dice = 2.0 * tp + fp + fn
    valid_dice = denom_dice > 0
    dice = torch.zeros_like(tp)
    dice[valid_dice] = (2.0 * tp[valid_dice]) / (denom_dice[valid_dice] + eps)

    mean_iou = iou[valid_iou].mean().item() if valid_iou.any() else 0.0
    mean_dice = dice[valid_dice].mean().item() if valid_dice.any() else 0.0
    return mean_iou, mean_dice


def poly_lr(base_lr: float, current_iter: int, max_iter: int, power: float = 0.9) -> float:
    return base_lr * ((1.0 - float(current_iter) / max_iter) ** power)


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
    parser.add_argument("--iou-weight", type=float, default=0.5, help="Weight for IoU loss")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for OHEM cross-entropy")
    parser.add_argument("--early-stop-patience", type=int, default=6, help="Number of epochs without validation improvement before stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum validation improvement to reset early stopping")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="mean_iou_dice",
        choices=["valid_loss", "mean_iou", "mean_dice", "mean_iou_dice"],
        help="Metric used for best checkpointing and early stopping",
    )
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
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    img_transform = Compose([
    ToImage(),
    Resize((256, 256)),
    ToDtype(torch.float32, scale=True),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),  # no scaling
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
    args.data_dir,
    split="train",
    mode="fine",
    target_type="semantic",
    transform=img_transform,
    target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
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
    iou_criterion = IoULoss(ignore_index=255)

    # Define the optimizer
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    max_iter = args.epochs * len(train_dataloader)
    global_iter = 0

    # Training loop
    best_valid_loss = float('inf')
    best_selection_score = -float('inf') if args.selection_metric != "valid_loss" else float('inf')
    epochs_without_improvement = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        train_losses_total = []
        train_losses_main = []
        train_losses_dice = []
        train_losses_iou = []
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            current_lr = poly_lr(args.lr, global_iter, max_iter, args.poly_power)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.zero_grad()
            outputs = model(images)

            if isinstance(outputs, tuple):
                main_logits, aux_logits = outputs
            else:
                main_logits, aux_logits = outputs, None

            loss_main = ohem_criterion(main_logits, labels)
            loss_aux = ohem_criterion(aux_logits, labels) if aux_logits is not None else torch.tensor(0.0, device=device)
            loss_dice = dice_criterion(main_logits, labels)
            loss_iou = iou_criterion(main_logits, labels)
            loss = loss_main + args.aux_weight * loss_aux + args.dice_weight * loss_dice + args.iou_weight * loss_iou

            train_losses_total.append(loss.item())
            train_losses_main.append(loss_main.item())
            train_losses_dice.append(loss_dice.item())
            train_losses_iou.append(loss_iou.item())

            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "train_loss_main_ohem": loss_main.item(),
                "train_loss_aux_ohem": loss_aux.item() if aux_logits is not None else 0.0,
                "train_loss_dice": loss_dice.item(),
                "train_loss_iou": loss_iou.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            global_iter += 1
            
        # Validation
        model.eval()
        with torch.no_grad():
            valid_losses_main = []
            valid_losses_dice = []
            valid_losses_iou = []
            valid_losses_total = []
            conf_mat = torch.zeros((19, 19), dtype=torch.int64)
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss_main = ohem_criterion(outputs, labels)
                loss_dice = dice_criterion(outputs, labels)
                loss_iou = iou_criterion(outputs, labels)
                loss_total = loss_main + args.dice_weight * loss_dice + args.iou_weight * loss_iou
                valid_losses_main.append(loss_main.item())
                valid_losses_dice.append(loss_dice.item())
                valid_losses_iou.append(loss_iou.item())
                valid_losses_total.append(loss_total.item())

                predictions = outputs.softmax(1).argmax(1)
                update_confusion_matrix(conf_mat, predictions.cpu(), labels.cpu(), n_classes=19, ignore_index=255)
            
                if i == 0:
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
            
            train_loss_epoch = sum(train_losses_total) / len(train_losses_total)
            train_loss_main_epoch = sum(train_losses_main) / len(train_losses_main)
            train_loss_dice_epoch = sum(train_losses_dice) / len(train_losses_dice)
            train_loss_iou_epoch = sum(train_losses_iou) / len(train_losses_iou)

            valid_loss = sum(valid_losses_total) / len(valid_losses_total)
            valid_loss_main = sum(valid_losses_main) / len(valid_losses_main)
            valid_loss_dice = sum(valid_losses_dice) / len(valid_losses_dice)
            valid_loss_iou = sum(valid_losses_iou) / len(valid_losses_iou)
            valid_mean_iou, valid_mean_dice = compute_mean_iou_and_dice(conf_mat)
            valid_mean_iou_dice = 0.5 * (valid_mean_iou + valid_mean_dice)
            wandb.log({
                "train_loss_epoch": train_loss_epoch,
                "train_loss_main_ohem_epoch": train_loss_main_epoch,
                "train_loss_dice_epoch": train_loss_dice_epoch,
                "train_loss_iou_epoch": train_loss_iou_epoch,
                "valid_loss": valid_loss,
                "valid_loss_main_ohem": valid_loss_main,
                "valid_loss_dice": valid_loss_dice,
                "valid_loss_iou": valid_loss_iou,
                "valid_mean_iou": valid_mean_iou,
                "valid_mean_dice": valid_mean_dice,
                "valid_mean_iou_dice": valid_mean_iou_dice,
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if args.selection_metric == "valid_loss":
                selection_score = valid_loss
                improved = selection_score < best_selection_score - args.early_stop_min_delta
            elif args.selection_metric == "mean_iou":
                selection_score = valid_mean_iou
                improved = selection_score > best_selection_score + args.early_stop_min_delta
            elif args.selection_metric == "mean_dice":
                selection_score = valid_mean_dice
                improved = selection_score > best_selection_score + args.early_stop_min_delta
            else:
                selection_score = valid_mean_iou_dice
                improved = selection_score > best_selection_score + args.early_stop_min_delta

            if improved:
                best_valid_loss = valid_loss
                best_selection_score = selection_score
                torch.save(model.state_dict(), best_model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1}: no validation improvement for {args.early_stop_patience} epochs.")
                    break
        
    print("Training complete!")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-best_val_loss={best_valid_loss:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
