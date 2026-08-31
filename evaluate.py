import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
from dataset import SkinLesionDataset,val_transform,build_mappings
from model import MultimodalSkinCancerModel

DATA_DIR="data"
OUTPUT_DIR="outputs"
METADATA_PATH=os.path.join(DATA_DIR,"HAM10000_metadata.csv")
CHECKPOINT=os.path.join(OUTPUT_DIR,"multimodal_best_model.pth")
SPLIT_PATH=os.path.join(OUTPUT_DIR,"data_splits.json")
METRICS_PATH=os.path.join(OUTPUT_DIR,"test_metrics.json")
CONFUSION_PATH=os.path.join(OUTPUT_DIR,"confusion_matrix.png")

IMAGE_DIRS=[
    os.path.join(DATA_DIR,"HAM10000_images_part_1"),
    os.path.join(DATA_DIR,"HAM10000_images_part_2")
]

CLASS_NAMES=["akiec","bcc","bkl","df","mel","nv","vasc"]
NUM_CLASSES=len(CLASS_NAMES)
BATCH_SIZE=32
NUM_WORKERS=0
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("FINAL MULTIMODAL MODEL EVALUATION")
print("="*70)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

if not os.path.isfile(METADATA_PATH):
    raise FileNotFoundError(
        f"Metadata not found: {METADATA_PATH}"
    )

print("\nLoading metadata...")
df=pd.read_csv(METADATA_PATH)

df["sex"]=df["sex"].fillna("unknown").astype(str).str.lower().str.strip()
df["localization"]=df["localization"].fillna("unknown").astype(str).str.lower().str.strip()
df["age"]=pd.to_numeric(df["age"],errors="coerce")
df["age"]=df["age"].fillna(df["age"].median())

print(f"Total samples: {len(df)}")

print("\nLoading saved test split...")

if not os.path.isfile(SPLIT_PATH):
    raise FileNotFoundError(
        f"Saved split file not found: {SPLIT_PATH}"
    )

with open(SPLIT_PATH,"r",encoding="utf-8") as f:
    split_data=json.load(f)

test_ids=set(split_data["test_ids"])

test_df=df[
    df["image_id"].astype(str).isin(test_ids)
].copy().reset_index(drop=True)

if len(test_df)==0:
    raise RuntimeError(
        "No test samples found using outputs/data_splits.json"
    )

print(f"Test samples: {len(test_df)}")

sex_mapping,location_mapping=build_mappings(df)

print("\nLoading multimodal model...")

checkpoint=torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "sex_mapping" in checkpoint:
    sex_mapping=checkpoint["sex_mapping"]

if "location_mapping" in checkpoint:
    location_mapping=checkpoint["location_mapping"]

model=MultimodalSkinCancerModel(
    num_sex_categories=len(sex_mapping),
    num_location_categories=len(location_mapping),
    num_classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")
print("Inference mode: IMAGE + AGE + SEX + LOCATION")

test_dataset=SkinLesionDataset(
    test_df,
    IMAGE_DIRS,
    sex_mapping,
    location_mapping,
    val_transform
)

test_loader=DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nGenerating test predictions...")

all_labels=[]
all_predictions=[]
all_probabilities=[]

with torch.no_grad():
    for batch in test_loader:
        images=batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        clinical=batch["clinical"].to(
            DEVICE,
            non_blocking=True
        )

        labels=batch["label"].to(
            DEVICE,
            non_blocking=True
        )

        outputs=model(
            images,
            clinical
        )

        probabilities=torch.softmax(
            outputs,
            dim=1
        )

        predictions=outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

labels=np.array(all_labels)
predictions=np.array(all_predictions)
probabilities=np.array(all_probabilities)

accuracy=accuracy_score(
    labels,
    predictions
)

balanced_accuracy=balanced_accuracy_score(
    labels,
    predictions
)

macro_precision=precision_score(
    labels,
    predictions,
    average="macro",
    zero_division=0
)

macro_recall=recall_score(
    labels,
    predictions,
    average="macro",
    zero_division=0
)

macro_f1=f1_score(
    labels,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1=f1_score(
    labels,
    predictions,
    average="weighted",
    zero_division=0
)

print("\n"+"="*70)
print("TEST RESULTS")
print("="*70)

print(f"Accuracy:            {accuracy:.4f}")
print(f"Balanced Accuracy:   {balanced_accuracy:.4f}")
print(f"Macro Precision:     {macro_precision:.4f}")
print(f"Macro Recall:        {macro_recall:.4f}")
print(f"Macro F1:            {macro_f1:.4f}")
print(f"Weighted F1:         {weighted_f1:.4f}")

print("\nClassification Report:")

report=classification_report(
    labels,
    predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)

print("="*70)
print("CONFUSION MATRIX")
print("="*70)

cm=confusion_matrix(
    labels,
    predictions,
    labels=np.arange(NUM_CLASSES)
)

print(cm)

os.makedirs(OUTPUT_DIR,exist_ok=True)

plt.figure(figsize=(10,8))
plt.imshow(cm)
plt.title("Multimodal Skin Cancer Classification")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(
    np.arange(NUM_CLASSES),
    [x.upper() for x in CLASS_NAMES],
    rotation=45
)
plt.yticks(
    np.arange(NUM_CLASSES),
    [x.upper() for x in CLASS_NAMES]
)

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(
            j,
            i,
            str(cm[i,j]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(
    CONFUSION_PATH,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

metrics={
    "model":"MultimodalSkinCancerModel",
    "inputs":[
        "image",
        "age",
        "sex",
        "location"
    ],
    "num_classes":NUM_CLASSES,
    "classes":CLASS_NAMES,
    "test_samples":int(len(labels)),
    "accuracy":float(accuracy),
    "balanced_accuracy":float(balanced_accuracy),
    "macro_precision":float(macro_precision),
    "macro_recall":float(macro_recall),
    "macro_f1":float(macro_f1),
    "weighted_f1":float(weighted_f1),
    "classification_report":classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    ),
    "confusion_matrix":cm.tolist()
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=2
    )

print("\n"+"="*70)
print("EVALUATION COMPLETED")
print("="*70)
print(f"Test samples: {len(labels)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print("\nResults saved to:")
print(METRICS_PATH)
print(CONFUSION_PATH)
print("="*70)