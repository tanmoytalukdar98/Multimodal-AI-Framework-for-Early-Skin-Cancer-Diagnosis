import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from dataset import SkinLesionDataset,val_transform,build_mappings
from model import MultimodalSkinCancerModel

DATA_DIR="data"
CHECKPOINT="outputs/multimodal_best_model.pth"
OUTPUT_DIR="outputs"
NUM_CLASSES=7
BATCH_SIZE=32
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("MULTIMODAL MODEL CALIBRATION")
print("="*60)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(OUTPUT_DIR,exist_ok=True)

metadata_path=os.path.join(
    DATA_DIR,
    "HAM10000_metadata.csv"
)

df=pd.read_csv(metadata_path)

df["sex"]=df["sex"].fillna("unknown").astype(str)
df["localization"]=df["localization"].fillna("unknown").astype(str)
df["age"]=pd.to_numeric(df["age"],errors="coerce")
df["age"]=df["age"].fillna(df["age"].median())

print(f"Total samples: {len(df)}")

sex_mapping,location_mapping=build_mappings(df)

splitter=StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

splits=list(
    splitter.split(
        df,
        df["dx"],
        groups=df["lesion_id"]
    )
)

train_val_idx,test_idx=splits[0]

train_val_df=df.iloc[train_val_idx].reset_index(drop=True)
test_df=df.iloc[test_idx].reset_index(drop=True)

val_splitter=StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx,val_idx=next(
    val_splitter.split(
        train_val_df,
        train_val_df["dx"],
        groups=train_val_df["lesion_id"]
    )
)

val_df=train_val_df.iloc[val_idx].reset_index(drop=True)

print("="*60)
print("DATA SPLITS")
print("="*60)
print(f"Validation samples: {len(val_df)}")
print(f"Test samples:       {len(test_df)}")

image_dirs=[
    os.path.join(
        DATA_DIR,
        "HAM10000_images_part_1"
    ),
    os.path.join(
        DATA_DIR,
        "HAM10000_images_part_2"
    )
]

val_dataset=SkinLesionDataset(
    val_df,
    image_dirs,
    sex_mapping,
    location_mapping,
    val_transform
)

test_dataset=SkinLesionDataset(
    test_df,
    image_dirs,
    sex_mapping,
    location_mapping,
    val_transform
)

val_loader=DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader=DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print("Loading multimodal model...")

checkpoint=torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

checkpoint_sex_mapping=checkpoint.get(
    "sex_mapping",
    sex_mapping
)

checkpoint_location_mapping=checkpoint.get(
    "location_mapping",
    location_mapping
)

model=MultimodalSkinCancerModel(
    num_sex_categories=len(checkpoint_sex_mapping),
    num_location_categories=len(checkpoint_location_mapping),
    num_classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")
print("Inference mode: IMAGE + AGE + SEX + LOCATION")

def collect_logits(model,loader):
    logits=[]
    labels=[]

    model.eval()

    with torch.no_grad():
        for batch in loader:
            images=batch["image"].to(DEVICE)
            clinical=batch["clinical"].to(DEVICE)
            targets=batch["label"].to(DEVICE)

            outputs=model(
                images,
                clinical
            )

            if isinstance(outputs,(tuple,list)):
                outputs=outputs[0]

            logits.append(
                outputs.cpu()
            )

            labels.append(
                targets.cpu()
            )

    return(
        torch.cat(logits),
        torch.cat(labels)
    )

print()
print("Generating validation predictions...")

val_logits,val_labels=collect_logits(
    model,
    val_loader
)

print(f"Validation predictions: {len(val_labels)}")

print()
print("Generating test predictions...")

test_logits,test_labels=collect_logits(
    model,
    test_loader
)

print(f"Test predictions: {len(test_labels)}")

def softmax_temperature(logits,temperature):
    return torch.softmax(
        logits/temperature,
        dim=1
    )

def expected_calibration_error(
    probabilities,
    labels,
    bins=15
):
    confidences,predictions=probabilities.max(
        dim=1
    )

    accuracies=predictions.eq(labels)

    ece=0.0

    bin_boundaries=torch.linspace(
        0.0,
        1.0,
        bins+1
    )

    for i in range(bins):
        lower=bin_boundaries[i]
        upper=bin_boundaries[i+1]

        mask=(
            (confidences>lower)
            &
            (confidences<=upper)
        )

        if mask.sum()==0:
            continue

        bin_accuracy=accuracies[mask].float().mean()
        bin_confidence=confidences[mask].mean()
        bin_fraction=mask.float().mean()

        ece+=(
            torch.abs(
                bin_accuracy-bin_confidence
            )*bin_fraction
        ).item()

    return ece

def multiclass_brier_score(
    probabilities,
    labels,
    num_classes
):
    targets=torch.zeros_like(probabilities)

    targets[
        torch.arange(len(labels)),
        labels
    ]=1.0

    return torch.mean(
        torch.sum(
            (probabilities-targets)**2,
            dim=1
        )
    ).item()

print()
print("="*60)
print("FITTING TEMPERATURE")
print("="*60)

temperature=torch.nn.Parameter(
    torch.ones(1,device=DEVICE)
)

optimizer=torch.optim.LBFGS(
    [temperature],
    lr=0.01,
    max_iter=100
)

criterion=torch.nn.CrossEntropyLoss()

val_logits_device=val_logits.to(DEVICE)
val_labels_device=val_labels.to(DEVICE)

def closure():
    optimizer.zero_grad()

    temp=torch.clamp(
        temperature,
        min=0.05,
        max=10.0
    )

    loss=criterion(
        val_logits_device/temp,
        val_labels_device
    )

    loss.backward()

    return loss

optimizer.step(closure)

temperature_value=float(
    torch.clamp(
        temperature,
        min=0.05,
        max=10.0
    ).item()
)

print(
    f"Learned temperature: {temperature_value:.4f}"
)

print()
print("="*60)
print("CALIBRATION RESULTS")
print("="*60)

val_prob_before=torch.softmax(
    val_logits,
    dim=1
)

val_prob_after=softmax_temperature(
    val_logits,
    temperature_value
)

test_prob_before=torch.softmax(
    test_logits,
    dim=1
)

test_prob_after=softmax_temperature(
    test_logits,
    temperature_value
)

val_predictions_before=val_prob_before.argmax(
    dim=1
)

val_predictions_after=val_prob_after.argmax(
    dim=1
)

accuracy_before=accuracy_score(
    val_labels.numpy(),
    val_predictions_before.numpy()
)

accuracy_after=accuracy_score(
    val_labels.numpy(),
    val_predictions_after.numpy()
)

f1_before=f1_score(
    val_labels.numpy(),
    val_predictions_before.numpy(),
    average="macro"
)

f1_after=f1_score(
    val_labels.numpy(),
    val_predictions_after.numpy(),
    average="macro"
)

ece_before=expected_calibration_error(
    val_prob_before,
    val_labels
)

ece_after=expected_calibration_error(
    val_prob_after,
    val_labels
)

brier_before=multiclass_brier_score(
    val_prob_before,
    val_labels,
    NUM_CLASSES
)

brier_after=multiclass_brier_score(
    val_prob_after,
    val_labels,
    NUM_CLASSES
)

print()
print("BEFORE CALIBRATION")
print(f"Accuracy:       {accuracy_before:.4f}")
print(f"Macro F1:       {f1_before:.4f}")
print(f"ECE:            {ece_before:.4f}")
print(f"Brier Score:    {brier_before:.4f}")

print()
print("AFTER CALIBRATION")
print(f"Accuracy:       {accuracy_after:.4f}")
print(f"Macro F1:       {f1_after:.4f}")
print(f"ECE:            {ece_after:.4f}")
print(f"Brier Score:    {brier_after:.4f}")

print()
print(f"Temperature:    {temperature_value:.4f}")

# ------------------------------------------------------------
# RELIABILITY DIAGRAM
# ------------------------------------------------------------

def reliability_data(
    probabilities,
    labels,
    bins=10
):
    confidences,predictions=probabilities.max(
        dim=1
    )

    correct=predictions.eq(labels)

    mean_confidences=[]
    mean_accuracies=[]

    for i in range(bins):
        lower=i/bins
        upper=(i+1)/bins

        if i==0:
            mask=(
                (confidences>=lower)
                &
                (confidences<=upper)
            )
        else:
            mask=(
                (confidences>lower)
                &
                (confidences<=upper)
            )

        if mask.sum()==0:
            continue

        mean_confidences.append(
            confidences[mask].mean().item()
        )

        mean_accuracies.append(
            correct[mask].float().mean().item()
        )

    return(
        mean_confidences,
        mean_accuracies
    )

conf_before,acc_before=reliability_data(
    val_prob_before,
    val_labels
)

conf_after,acc_after=reliability_data(
    val_prob_after,
    val_labels
)

plt.figure(figsize=(7,7))

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.plot(
    conf_before,
    acc_before,
    marker="o",
    label="Before Calibration"
)

plt.plot(
    conf_after,
    acc_after,
    marker="o",
    label="After Calibration"
)

plt.xlabel("Mean Predicted Confidence")
plt.ylabel("Observed Accuracy")
plt.title("Reliability Diagram")
plt.legend()
plt.grid(True)
plt.tight_layout()

reliability_path=os.path.join(
    OUTPUT_DIR,
    "reliability_diagram.png"
)

plt.savefig(
    reliability_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# SAVE CALIBRATION RESULTS
# ------------------------------------------------------------

results={
    "temperature":temperature_value,
    "validation_samples":len(val_labels),
    "test_samples":len(test_labels),
    "before_calibration":{
        "accuracy":accuracy_before,
        "macro_f1":f1_before,
        "ece":ece_before,
        "brier_score":brier_before
    },
    "after_calibration":{
        "accuracy":accuracy_after,
        "macro_f1":f1_after,
        "ece":ece_after,
        "brier_score":brier_after
    }
}

results_path=os.path.join(
    OUTPUT_DIR,
    "calibration_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=2
    )

# ------------------------------------------------------------
# SAVE TEMPERATURE
# ------------------------------------------------------------

temperature_path=os.path.join(
    OUTPUT_DIR,
    "temperature.json"
)

with open(
    temperature_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "temperature":temperature_value
        },
        f,
        indent=2
    )

print()
print("="*60)
print("CALIBRATION COMPLETED")
print("="*60)

print(
    f"Temperature: {temperature_value:.4f}"
)

print(
    f"ECE: {ece_before:.4f} -> {ece_after:.4f}"
)

print(
    f"Brier Score: {brier_before:.4f} -> {brier_after:.4f}"
)

print()
print(
    f"Reliability diagram saved to:"
)

print(
    reliability_path
)

print()
print(
    f"Results saved to:"
)

print(
    results_path
)

print()
print(
    f"Temperature saved to:"
)

print(
    temperature_path
)

print("="*60)