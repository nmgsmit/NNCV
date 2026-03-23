"""
Example prediction pipeline for the SegFormer model used in this project.
It loads a trained checkpoint, preprocesses input images, and saves the
predicted segmentation masks.

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

from model import Model

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"
IMAGE_SIZE = (512, 1024)
MODEL_VARIANT = "b5"
CITYSCAPES_MEAN = (0.485, 0.456, 0.406)
CITYSCAPES_STD = (0.229, 0.224, 0.225)
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


def build_model_from_variant(variant: str) -> Model:
    try:
        config = MODEL_CONFIGS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown model variant '{variant}'. Expected one of: {', '.join(MODEL_CONFIGS)}") from exc

    return Model(
        in_channels=3,
        n_classes=19,
        dropout=0.1,
        drop_path_rate=0.1,
        **config,
    )


def preprocess(img: Image.Image) -> torch.Tensor:
    # Implement your preprocessing steps here
    # For example, resizing, normalization, etc.
    # Return a tensor suitable for model input
    transform = Compose([
        ToImage(),
        Resize(size=IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD),
    ])

    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Implement your postprocessing steps here
    # For example, resizing back to original shape, converting to color mask, etc.
    # Return a numpy array suitable for saving as an image
    logits = F.interpolate(pred, size=original_shape, mode="bilinear", align_corners=False)
    prediction = logits.argmax(dim=1)
    return prediction.squeeze(0).cpu().detach().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = build_model_from_variant(MODEL_VARIANT)
    state_dict = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        state_dict, 
        strict=True,  # Ensure the state dict matches the model architecture
    )
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))  # DO NOT CHANGE, IMAGES WILL BE PROVIDED IN THIS FORMAT
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass
            pred = model(img_tensor)

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Create mirrored output folder
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
