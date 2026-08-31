import os
import argparse

import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold

from dataset import (
    SkinLesionDataset,
    val_transform,
    build_mappings
)

from model import MultimodalSkinCancerModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

# IMPORTANT:
# This is the trained MULTIMODAL checkpoint.
CHECKPOINT = "outputs/multimodal_best_model.pth"

OUTPUT_DIR = "outputs/gradcam"

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Grad-CAM visualization for the multimodal skin lesion model"
)

parser.add_argument(
    "--index",
    type=int,
    default=0,
    help="Index of the sample in the held-out test set"
)

args = parser.parse_args()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE INFORMATION
# ============================================================

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD HAM10000 METADATA
# ============================================================

metadata_path = os.path.join(
    DATA_DIR,
    "HAM10000_metadata.csv"
)

df = pd.read_csv(metadata_path)


# ============================================================
# PREPROCESS CLINICAL DATA
# ============================================================

df["sex"] = (
    df["sex"]
    .fillna("unknown")
    .astype(str)
)

df["localization"] = (
    df["localization"]
    .fillna("unknown")
    .astype(str)
)

df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)

df["age"] = df["age"].fillna(
    df["age"].median()
)


# ============================================================
# BUILD CLINICAL MAPPINGS
# ============================================================

sex_mapping, location_mapping = build_mappings(df)


# ============================================================
# RECREATE THE SAME TEST SPLIT
# ============================================================

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

_, test_idx = next(
    splitter.split(
        df,
        df["dx"],
        groups=df["lesion_id"]
    )
)

test_df = (
    df.iloc[test_idx]
    .reset_index(drop=True)
)


# ============================================================
# VALIDATE INDEX
# ============================================================

if args.index < 0 or args.index >= len(test_df):

    raise ValueError(
        f"Index must be between "
        f"0 and {len(test_df) - 1}"
    )


print(
    f"Test samples: {len(test_df)}"
)


# ============================================================
# IMAGE DIRECTORIES
# ============================================================

image_dirs = [
    os.path.join(
        DATA_DIR,
        "HAM10000_images_part_1"
    ),
    os.path.join(
        DATA_DIR,
        "HAM10000_images_part_2"
    )
]


# ============================================================
# CREATE TEST DATASET
# ============================================================

dataset = SkinLesionDataset(
    test_df,
    image_dirs,
    sex_mapping,
    location_mapping,
    val_transform
)


# ============================================================
# LOAD SELECTED SAMPLE
# ============================================================

sample = dataset[args.index]

image_tensor = (
    sample["image"]
    .unsqueeze(0)
    .to(DEVICE)
)

clinical_tensor = (
    sample["clinical"]
    .unsqueeze(0)
    .to(DEVICE)
)

true_label = int(
    sample["label"].item()
)

image_id = sample["image_id"]


print(
    f"Image ID: {image_id}"
)


# ============================================================
# LOAD MULTIMODAL CHECKPOINT
# ============================================================

if not os.path.isfile(CHECKPOINT):

    raise FileNotFoundError(
        f"Multimodal checkpoint not found: "
        f"{CHECKPOINT}"
    )


print(
    "Loading multimodal model..."
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


# ============================================================
# CREATE MODEL
# ============================================================

model = MultimodalSkinCancerModel().to(
    DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# VERIFY MODEL
# ============================================================

print(
    "Model loaded successfully."
)

print(
    "Inference mode: IMAGE + CLINICAL"
)


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================
#
# We explicitly target the final convolutional
# representation of the EfficientNet-B0 image encoder.
#
# Grad-CAM explains the IMAGE branch of the
# multimodal prediction.
#
# Clinical features are still used during the
# forward pass.
# ============================================================

target_layer = model.image_encoder.features[-1][0]

print(
    f"Grad-CAM target layer: {target_layer}"
)


# ============================================================
# HOOK STORAGE
# ============================================================

activations = None
gradients = None


# ============================================================
# FORWARD HOOK
# ============================================================

def forward_hook(module, inputs, output):

    global activations

    activations = output


# ============================================================
# BACKWARD HOOK
# ============================================================

def backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


# ============================================================
# REGISTER HOOKS
# ============================================================

forward_handle = (
    target_layer.register_forward_hook(
        forward_hook
    )
)

backward_handle = (
    target_layer.register_full_backward_hook(
        backward_hook
    )
)


# ============================================================
# MULTIMODAL FORWARD PASS
# ============================================================

model.zero_grad()

output = model(
    image_tensor,
    clinical_tensor
)


if isinstance(
    output,
    (tuple, list)
):

    output = output[0]


# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

probabilities = torch.softmax(
    output,
    dim=1
)


predicted_class = int(
    output.argmax(dim=1).item()
)


confidence = float(
    probabilities[
        0,
        predicted_class
    ].item()
)


# ============================================================
# BACKPROPAGATE PREDICTED CLASS
# ============================================================

score = output[
    0,
    predicted_class
]

score.backward()


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()
backward_handle.remove()


# ============================================================
# VERIFY HOOK OUTPUT
# ============================================================

if activations is None:

    raise RuntimeError(
        "Grad-CAM did not capture activations."
    )


if gradients is None:

    raise RuntimeError(
        "Grad-CAM did not capture gradients."
    )


# ============================================================
# DETACH ACTIVATIONS AND GRADIENTS
# ============================================================

activations = activations.detach()

gradients = gradients.detach()


# ============================================================
# GRAD-CAM COMPUTATION
# ============================================================

# Global average pooling over spatial dimensions

weights = gradients.mean(
    dim=(2, 3),
    keepdim=True
)


# Weighted combination of feature maps

cam = (
    weights * activations
).sum(
    dim=1,
    keepdim=True
)


# ReLU

cam = torch.relu(cam)


# Remove batch/channel dimensions

cam = (
    cam
    .squeeze()
    .cpu()
    .numpy()
)


# ============================================================
# NORMALIZE CAM
# ============================================================

if cam.max() > 0:

    cam = cam / cam.max()


# ============================================================
# RESIZE CAM TO IMAGE SIZE
# ============================================================

cam = cv2.resize(
    cam,
    (224, 224)
)


# ============================================================
# RECONSTRUCT ORIGINAL IMAGE
# ============================================================

image_np = (
    image_tensor
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
)

image_np = np.transpose(
    image_np,
    (1, 2, 0)
)


# ImageNet normalization values

mean = np.array(
    [0.485, 0.456, 0.406]
)

std = np.array(
    [0.229, 0.224, 0.225]
)


# Undo normalization

image_np = (
    image_np * std
) + mean


image_np = np.clip(
    image_np,
    0,
    1
)


image_uint8 = (
    image_np * 255
).astype(np.uint8)


# ============================================================
# CREATE HEATMAP
# ============================================================

heatmap = (
    cam * 255
).astype(np.uint8)


heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# CREATE OVERLAY
# ============================================================

original_bgr = cv2.cvtColor(
    image_uint8,
    cv2.COLOR_RGB2BGR
)


heatmap_bgr = cv2.cvtColor(
    heatmap,
    cv2.COLOR_RGB2BGR
)


overlay = cv2.addWeighted(
    original_bgr,
    0.55,
    heatmap_bgr,
    0.45,
    0
)


overlay = cv2.cvtColor(
    overlay,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# OUTPUT FILENAME
# ============================================================

base_name = (
    f"{image_id}_gradcam"
)


output_path = os.path.join(
    OUTPUT_DIR,
    f"{base_name}.png"
)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(
    figsize=(15, 5)
)


# ------------------------------------------------------------
# ORIGINAL IMAGE
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    image_uint8
)

plt.title(
    f"Original\n"
    f"True: "
    f"{CLASS_NAMES[true_label].upper()}"
)

plt.axis("off")


# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    heatmap
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis("off")


# ------------------------------------------------------------
# OVERLAY
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    overlay
)

plt.title(
    f"Prediction: "
    f"{CLASS_NAMES[predicted_class].upper()}\n"
    f"Confidence: "
    f"{confidence * 100:.2f}%"
)

plt.axis("off")


plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# RESULT INFORMATION
# ============================================================

result = {

    "image_id": image_id,

    "true_class":
        CLASS_NAMES[true_label],

    "predicted_class":
        CLASS_NAMES[predicted_class],

    "confidence":
        confidence,

    "model":
        "multimodal",

    "inputs": [
        "image",
        "age",
        "sex",
        "localization"
    ],

    "gradcam_target":
        "EfficientNet-B0 final convolutional layer",

    "output":
        output_path
}


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("==============================")
print("GRAD-CAM RESULT")
print("==============================")

print(
    f"Image ID:       {image_id}"
)

print(
    f"True Class:     "
    f"{CLASS_NAMES[true_label].upper()}"
)

print(
    f"Prediction:     "
    f"{CLASS_NAMES[predicted_class].upper()}"
)

print(
    f"Confidence:     "
    f"{confidence * 100:.2f}%"
)

print(
    f"Model:          MULTIMODAL"
)

print(
    f"Inputs:         IMAGE + AGE + SEX + LOCATION"
)

print(
    f"Saved to:       {output_path}"
)

print("==============================")