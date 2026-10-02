import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0


class ClinicalEncoder(nn.Module):

    def __init__(
        self,
        num_sex_categories,
        num_location_categories,
        output_size=128
    ):
        super().__init__()

        self.sex_embedding = nn.Embedding(num_sex_categories, 16)
        self.location_embedding = nn.Embedding(num_location_categories, 32)

        self.network = nn.Sequential(
            nn.Linear(1 + 16 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

    def forward(self, x):
        age = x[:, 0:1]
        sex = x[:, 1].long()
        location = x[:, 2].long()

        sex_features = self.sex_embedding(sex)
        location_features = self.location_embedding(location)

        combined = torch.cat(
            [age, sex_features, location_features],
            dim=1
        )

        return self.network(combined)


class MultimodalSkinCancerModel(nn.Module):

    def __init__(
        self,
        num_sex_categories,
        num_location_categories,
        num_classes=7
    ):
        super().__init__()

        # The trained checkpoint already contains the EfficientNet-B0
        # weights, so deployment must not download ImageNet weights.
        self.image_encoder = efficientnet_b0(weights=None)

        image_features = self.image_encoder.classifier[1].in_features
        self.image_encoder.classifier = nn.Identity()

        self.clinical_encoder = ClinicalEncoder(
            num_sex_categories=num_sex_categories,
            num_location_categories=num_location_categories,
            output_size=128
        )

        fusion_size = image_features + 128

        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, clinical):
        image_features = self.image_encoder(image)
        clinical_features = self.clinical_encoder(clinical)

        combined = torch.cat(
            [image_features, clinical_features],
            dim=1
        )

        return self.classifier(combined)
