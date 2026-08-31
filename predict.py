import os
import argparse

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from dataset import build_mappings
from model import MultimodalSkinCancerModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

CHECKPOINT = "outputs/multimodal_best_model.pth"

# Learned during calibration.py
TEMPERATURE = 0.8334

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
    description="Multimodal Skin Cancer Prediction"
)

parser.add_argument(
    "--image",
    type=str,
    required=True
)

parser.add_argument(
    "--age",
    type=float,
    required=True
)

parser.add_argument(
    "--sex",
    type=str,
    required=True
)

parser.add_argument(
    "--location",
    type=str,
    required=True
)

args = parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("Inference mode: IMAGE + AGE + SEX + LOCATION")
print(f"Calibration temperature: {TEMPERATURE}")


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.isfile(args.image):
    raise FileNotFoundError(
        f"Image not found: {args.image}"
    )


# ============================================================
# LOAD METADATA
# ============================================================

metadata_path = os.path.join(
    DATA_DIR,
    "HAM10000_metadata.csv"
)

df = pd.read_csv(metadata_path)

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

sex = args.sex.lower().strip()
location = args.location.lower().strip()


if sex not in sex_mapping:

    print(
        f"Warning: '{sex}' not found in training data."
    )

    print(
        "Using 'unknown' instead."
    )

    sex = "unknown"


if location not in location_mapping:

    print(
        f"Warning: '{location}' not found in training data."
    )

    print(
        "Using 'unknown' instead."
    )

    location = "unknown"


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    args.image
).convert("RGB")

image_tensor = (
    image_transform(image)
    .unsqueeze(0)
    .to(DEVICE)
)


# ============================================================
# CLINICAL FEATURES
# ============================================================

age_value = float(args.age) / 100.0

sex_value = sex_mapping[sex]

location_value = location_mapping[location]

clinical_tensor = torch.tensor(
    [
        [
            age_value,
            sex_value,
            location_value
        ]
    ],
    dtype=torch.float32
).to(DEVICE)


# ============================================================
# LOAD MULTIMODAL MODEL
# ============================================================

print("\nLoading multimodal model...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

checkpoint=torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

sex_mapping=checkpoint["sex_mapping"]
location_mapping=checkpoint["location_mapping"]

model=MultimodalSkinCancerModel(
    num_sex_categories=len(sex_mapping),
    num_location_categories=len(location_mapping),
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():

    output = model(
        image_tensor,
        clinical_tensor
    )

    # --------------------------------------------------------
    # TEMPERATURE CALIBRATION
    # --------------------------------------------------------
    #
    # Calibration changes confidence probabilities.
    # It does NOT retrain or modify model weights.
    #
    calibrated_probabilities = torch.softmax(
        output / TEMPERATURE,
        dim=1
    )[0]


# ============================================================
# PREDICTION
# ============================================================

predicted_class = int(
    calibrated_probabilities.argmax().item()
)

predicted_name = CLASS_NAMES[
    predicted_class
]

confidence = float(
    calibrated_probabilities[
        predicted_class
    ].item()
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("MULTIMODAL SKIN LESION PREDICTION")
print("========================================")

print(
    f"Image:       {args.image}"
)

print(
    f"Age:         {args.age}"
)

print(
    f"Sex:         {sex}"
)

print(
    f"Location:    {location}"
)

print(
    "\nModel:       MULTIMODAL"
)

print(
    "Inputs:      IMAGE + AGE + SEX + LOCATION"
)

print(
    f"Temperature: {TEMPERATURE}"
)


# ============================================================
# CLASS PROBABILITIES
# ============================================================

print("\nClass Probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    probability = float(
        calibrated_probabilities[i].item()
    )

    print(
        f"{class_name.upper():6s}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# FINAL PREDICTION
# ============================================================

print("\n========================================")
print("FINAL PREDICTION")
print("========================================")

print(
    f"Class:       {predicted_name.upper()}"
)

print(
    f"Confidence:  {confidence * 100:.2f}%"
)

print(
    "Model:       MULTIMODAL"
)

print(
    "Calibration: TEMPERATURE SCALING"
)

print("========================================")