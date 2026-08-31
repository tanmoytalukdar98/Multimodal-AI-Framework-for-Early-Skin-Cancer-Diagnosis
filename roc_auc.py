import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from dataset import SkinLesionDataset, val_transform, build_mappings
from model import MultimodalSkinCancerModel

DATA_DIR = "data"
CHECKPOINT = "outputs/multimodal_best_model.pth"
OUTPUT_DIR = "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

df = pd.read_csv(
    os.path.join(DATA_DIR, "HAM10000_metadata.csv")
)

df["sex"] = df["sex"].fillna("unknown").astype(str)
df["localization"] = df["localization"].fillna("unknown").astype(str)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["age"] = df["age"].fillna(df["age"].median())

sex_mapping, location_mapping = build_mappings(df)

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

test_df = df.iloc[test_idx].reset_index(drop=True)

image_dirs = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2")
]

test_dataset = SkinLesionDataset(
    test_df,
    image_dirs,
    sex_mapping,
    location_mapping,
    val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model = MultimodalSkinCancerModel().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

all_labels = []
all_probabilities = []

print(f"Test samples: {len(test_dataset)}")
print("Generating predictions...")

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        clinical = batch["clinical"].to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            images,
            clinical
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            batch["label"].cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

all_labels = np.array(all_labels)
all_probabilities = np.array(all_probabilities)

binary_labels = label_binarize(
    all_labels,
    classes=np.arange(NUM_CLASSES)
)

auc_scores = {}

for i, class_name in enumerate(CLASS_NAMES):
    try:
        score = roc_auc_score(
            binary_labels[:, i],
            all_probabilities[:, i]
        )

        auc_scores[class_name] = float(score)

    except ValueError:
        auc_scores[class_name] = None

macro_auc = roc_auc_score(
    binary_labels,
    all_probabilities,
    average="macro",
    multi_class="ovr"
)

weighted_auc = roc_auc_score(
    binary_labels,
    all_probabilities,
    average="weighted",
    multi_class="ovr"
)

print("\n==============================")
print("ROC-AUC RESULTS")
print("==============================")

for class_name, score in auc_scores.items():
    if score is not None:
        print(
            f"{class_name.upper():6s}: {score:.4f}"
        )
    else:
        print(
            f"{class_name.upper():6s}: N/A"
        )

print(f"\nMacro ROC-AUC:    {macro_auc:.4f}")
print(f"Weighted ROC-AUC: {weighted_auc:.4f}")

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(
        binary_labels[:, i],
        all_probabilities[:, i]
    )

    class_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name.upper()} (AUC={class_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(
    "Multiclass ROC Curves - Multimodal Skin Cancer Model"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

roc_curve_path = os.path.join(
    OUTPUT_DIR,
    "roc_auc_curves.png"
)

plt.savefig(
    roc_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

roc_results = {
    "class_auc": auc_scores,
    "macro_auc": float(macro_auc),
    "weighted_auc": float(weighted_auc)
}

results_path = os.path.join(
    OUTPUT_DIR,
    "roc_auc_results.json"
)

with open(
    results_path,
    "w"
) as f:
    json.dump(
        roc_results,
        f,
        indent=2
    )

print("\nResults saved:")
print("outputs/roc_auc_results.json")
print("outputs/roc_auc_curves.png")