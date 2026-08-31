import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from dataset import SkinLesionDataset,train_transform,val_transform,build_mappings
from model import MultimodalSkinCancerModel

DATA_DIR="data"
METADATA_PATH=os.path.join(DATA_DIR,"HAM10000_metadata.csv")
IMAGE_DIRS=[
    os.path.join(DATA_DIR,"HAM10000_images_part_1"),
    os.path.join(DATA_DIR,"HAM10000_images_part_2")
]
OUTPUT_DIR="outputs"
CHECKPOINT_PATH=os.path.join(OUTPUT_DIR,"multimodal_best_model.pth")
HISTORY_PATH=os.path.join(OUTPUT_DIR,"multimodal_training_history.json")
SPLIT_PATH=os.path.join(OUTPUT_DIR,"data_splits.json")
CLASS_NAMES=["akiec","bcc","bkl","df","mel","nv","vasc"]
NUM_CLASSES=7
SEED=42
BATCH_SIZE=32
NUM_EPOCHS=25
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
PATIENCE=5
NUM_WORKERS=0
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

def calculate_metrics(labels,predictions):
    accuracy=accuracy_score(labels,predictions)
    balanced_accuracy=balanced_accuracy_score(labels,predictions)
    macro_f1=f1_score(labels,predictions,average="macro",zero_division=0)
    return accuracy,balanced_accuracy,macro_f1

set_seed(SEED)
os.makedirs(OUTPUT_DIR,exist_ok=True)

print("="*70)
print("FINAL MULTIMODAL SKIN CANCER TRAINING")
print("="*70)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading HAM10000 metadata...")
df=pd.read_csv(METADATA_PATH)
print(f"Total samples: {len(df)}")

df["sex"]=df["sex"].fillna("unknown").astype(str).str.lower().str.strip()
df["localization"]=df["localization"].fillna("unknown").astype(str).str.lower().str.strip()
df["age"]=pd.to_numeric(df["age"],errors="coerce")
df["age"]=df["age"].fillna(df["age"].median())

sex_mapping,location_mapping=build_mappings(df)

print("\nClinical mappings:")
print(f"Sex categories: {len(sex_mapping)}")
print(f"Location categories: {len(location_mapping)}")
print(f"Sex mapping: {sex_mapping}")
print(f"Location mapping: {location_mapping}")

print("\nCreating group-aware data splits...")
splitter=StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)
train_val_idx,test_idx=next(
    splitter.split(
        df,
        df["dx"],
        groups=df["lesion_id"]
    )
)
train_val_df=df.iloc[train_val_idx].reset_index(drop=True)
test_df=df.iloc[test_idx].reset_index(drop=True)

splitter_val=StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)
train_idx,val_idx=next(
    splitter_val.split(
        train_val_df,
        train_val_df["dx"],
        groups=train_val_df["lesion_id"]
    )
)
train_df=train_val_df.iloc[train_idx].reset_index(drop=True)
val_df=train_val_df.iloc[val_idx].reset_index(drop=True)

print("\n"+"="*70)
print("DATA SPLIT")
print("="*70)
print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples:       {len(test_df)}")

with open(SPLIT_PATH,"w",encoding="utf-8") as f:
    json.dump(
        {
            "seed":SEED,
            "train_ids":train_df["image_id"].tolist(),
            "val_ids":val_df["image_id"].tolist(),
            "test_ids":test_df["image_id"].tolist()
        },
        f,
        indent=2
    )

print(f"Splits saved to: {SPLIT_PATH}")

train_dataset=SkinLesionDataset(
    train_df,
    IMAGE_DIRS,
    sex_mapping,
    location_mapping,
    train_transform
)
val_dataset=SkinLesionDataset(
    val_df,
    IMAGE_DIRS,
    sex_mapping,
    location_mapping,
    val_transform
)

train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)
val_loader=DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nClass distribution:")
class_counts=train_df["dx"].value_counts()
counts=np.array(
    [class_counts.get(c,0) for c in CLASS_NAMES],
    dtype=np.float32
)
for name,count in zip(CLASS_NAMES,counts):
    print(f"{name.upper():6s}: {int(count)}")

if np.any(counts==0):
    raise ValueError("One or more classes have zero training samples.")

total_samples=counts.sum()
class_weights=total_samples/(NUM_CLASSES*counts)
class_weights=class_weights/class_weights.mean()
class_weights=torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("\nClass weights:")
for name,weight in zip(CLASS_NAMES,class_weights.cpu().numpy()):
    print(f"{name.upper():6s}: {weight:.4f}")

print("\nBuilding multimodal model...")
model=MultimodalSkinCancerModel(
    num_sex_categories=len(sex_mapping),
    num_location_categories=len(location_mapping),
    num_classes=NUM_CLASSES
).to(DEVICE)

print("Model: EfficientNet-B0 + Clinical Embeddings + Feature Fusion")
print("Inputs: IMAGE + AGE + SEX + LOCATION")

criterion=nn.CrossEntropyLoss(
    weight=class_weights
)
optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

def train_one_epoch():
    model.train()
    total_loss=0.0
    labels_all=[]
    predictions_all=[]
    for batch in train_loader:
        images=batch["image"].to(DEVICE,non_blocking=True)
        clinical=batch["clinical"].to(DEVICE,non_blocking=True)
        labels=batch["label"].to(DEVICE,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs=model(images,clinical)
        loss=criterion(outputs,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total_loss+=loss.item()*labels.size(0)
        predictions=outputs.argmax(dim=1)
        labels_all.extend(labels.detach().cpu().numpy())
        predictions_all.extend(predictions.detach().cpu().numpy())
    loss=total_loss/len(train_loader.dataset)
    acc,balanced_acc,f1=calculate_metrics(labels_all,predictions_all)
    return loss,acc,balanced_acc,f1

@torch.no_grad()
def validate():
    model.eval()
    total_loss=0.0
    labels_all=[]
    predictions_all=[]
    for batch in val_loader:
        images=batch["image"].to(DEVICE,non_blocking=True)
        clinical=batch["clinical"].to(DEVICE,non_blocking=True)
        labels=batch["label"].to(DEVICE,non_blocking=True)
        outputs=model(images,clinical)
        loss=criterion(outputs,labels)
        total_loss+=loss.item()*labels.size(0)
        predictions=outputs.argmax(dim=1)
        labels_all.extend(labels.cpu().numpy())
        predictions_all.extend(predictions.cpu().numpy())
    loss=total_loss/len(val_loader.dataset)
    acc,balanced_acc,f1=calculate_metrics(labels_all,predictions_all)
    return loss,acc,balanced_acc,f1

print("\n"+"="*70)
print("STARTING FINAL MULTIMODAL TRAINING")
print("="*70)

history=[]
best_val_f1=-1.0
epochs_without_improvement=0

for epoch in range(1,NUM_EPOCHS+1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print("-"*70)
    train_loss,train_acc,train_bal,train_f1=train_one_epoch()
    val_loss,val_acc,val_bal,val_f1=validate()
    scheduler.step(val_f1)
    lr=optimizer.param_groups[0]["lr"]
    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train Balanced Acc: {train_bal:.4f} | "
        f"Train F1: {train_f1:.4f}"
    )
    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Balanced Acc: {val_bal:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )
    print(f"Learning Rate: {lr:.7f}")
    history.append({
        "epoch":epoch,
        "train_loss":float(train_loss),
        "train_accuracy":float(train_acc),
        "train_balanced_accuracy":float(train_bal),
        "train_macro_f1":float(train_f1),
        "val_loss":float(val_loss),
        "val_accuracy":float(val_acc),
        "val_balanced_accuracy":float(val_bal),
        "val_macro_f1":float(val_f1),
        "learning_rate":float(lr)
    })
    if val_f1>best_val_f1:
        best_val_f1=val_f1
        epochs_without_improvement=0
        torch.save(
            {
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "epoch":epoch,
                "best_val_f1":float(best_val_f1),
                "sex_mapping":sex_mapping,
                "location_mapping":location_mapping,
                "class_names":CLASS_NAMES
            },
            CHECKPOINT_PATH
        )
        print(f"✓ Best model saved | Val Macro F1: {best_val_f1:.4f}")
    else:
        epochs_without_improvement+=1
        print(f"No improvement ({epochs_without_improvement}/{PATIENCE})")
    if epochs_without_improvement>=PATIENCE:
        print("\nEarly stopping triggered.")
        break

with open(HISTORY_PATH,"w",encoding="utf-8") as f:
    json.dump(history,f,indent=2)

print("\n"+"="*70)
print("MULTIMODAL TRAINING COMPLETED")
print("="*70)
print(f"Best Validation Macro F1: {best_val_f1:.4f}")
print(f"Model saved to: {CHECKPOINT_PATH}")
print(f"History saved to: {HISTORY_PATH}")
print(f"Splits saved to: {SPLIT_PATH}")
print("="*70)