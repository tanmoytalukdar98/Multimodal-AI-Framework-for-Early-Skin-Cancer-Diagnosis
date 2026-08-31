import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES=["akiec","bcc","bkl","df","mel","nv","vasc"]
CLASS_TO_IDX={name:idx for idx,name in enumerate(CLASS_NAMES)}

train_transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224,scale=(0.80,1.0),ratio=(0.90,1.10)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15,contrast=0.15,saturation=0.10,hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.20,scale=(0.02,0.10),ratio=(0.3,3.3))
])

val_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

def build_mappings(df):
    sex_values=sorted(
        df["sex"].fillna("unknown").astype(str).str.lower().str.strip().unique().tolist()
    )
    location_values=sorted(
        df["localization"].fillna("unknown").astype(str).str.lower().str.strip().unique().tolist()
    )
    sex_mapping={value:idx for idx,value in enumerate(sex_values)}
    location_mapping={value:idx for idx,value in enumerate(location_values)}
    return sex_mapping,location_mapping

def find_image(image_id,image_dirs):
    filename=f"{image_id}.jpg"
    for directory in image_dirs:
        path=os.path.join(directory,filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Image not found: {filename}\nSearched directories:\n"+"\n".join(image_dirs)
    )

class SkinLesionDataset(Dataset):
    def __init__(self,dataframe,image_dirs,sex_mapping,location_mapping,transform=None):
        self.df=dataframe.reset_index(drop=True)
        self.image_dirs=image_dirs
        self.sex_mapping=sex_mapping
        self.location_mapping=location_mapping
        self.transform=transform
        required_columns=["image_id","dx","age","sex","localization"]
        missing_columns=[c for c in required_columns if c not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required metadata columns: {missing_columns}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        row=self.df.iloc[index]
        image_id=str(row["image_id"])
        image_path=find_image(image_id,self.image_dirs)
        image=Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image=self.transform(image)

        age=pd.to_numeric(row["age"],errors="coerce")
        if pd.isna(age):
            age=0.0
        age=float(np.clip(float(age)/100.0,0.0,1.0))

        sex=str(row["sex"]).lower().strip()
        sex_index=self.sex_mapping.get(
            sex,
            self.sex_mapping.get("unknown",0)
        )

        location=str(row["localization"]).lower().strip()
        location_index=self.location_mapping.get(
            location,
            self.location_mapping.get("unknown",0)
        )

        clinical=torch.tensor(
            [age,float(sex_index),float(location_index)],
            dtype=torch.float32
        )

        diagnosis=str(row["dx"]).lower().strip()
        if diagnosis not in CLASS_TO_IDX:
            raise ValueError(
                f"Unknown diagnosis '{diagnosis}' for image {image_id}"
            )

        label=torch.tensor(
            CLASS_TO_IDX[diagnosis],
            dtype=torch.long
        )

        return {
            "image":image,
            "clinical":clinical,
            "label":label,
            "image_id":image_id
        }