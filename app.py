import os
import hashlib

import torch

from flask import Flask, request, jsonify
from flask_cors import CORS

from PIL import Image

from torchvision import transforms

from model import MultimodalSkinCancerModel


# ==================================================
# Flask application
# ==================================================

app = Flask(__name__)

CORS(app)


# ==================================================
# Paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "multimodal_best_model.pth"
)


# ==================================================
# Model configuration
# ==================================================

TEMPERATURE = 1.2163183689


EXPECTED_MODEL_SHA256 = (
    "261a04172b8f6498e4b98f09baa10bdd0f8f382b81634abadfdad2d85f9ef7c4"
)


CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ==================================================
# CPU deployment
# ==================================================

DEVICE = torch.device("cpu")


# ==================================================
# Image preprocessing
# ==================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ==================================================
# Startup diagnostics
# ==================================================

print("=" * 50)

print("DermaVision AI API")

print("=" * 50)

print("Device:", DEVICE)

print(
    "Checkpoint path:",
    CHECKPOINT
)

print(
    "Checkpoint exists:",
    os.path.exists(CHECKPOINT)
)


# ==================================================
# Check checkpoint exists
# ==================================================

if not os.path.exists(CHECKPOINT):

    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )


# ==================================================
# Checkpoint size
# ==================================================

file_size = os.path.getsize(
    CHECKPOINT
)

print(
    "Checkpoint size:",
    file_size,
    "bytes"
)


# ==================================================
# SHA-256 verification
# ==================================================

print(
    "Calculating checkpoint SHA256..."
)


sha256 = hashlib.sha256()


with open(
    CHECKPOINT,
    "rb"
) as f:

    while True:

        chunk = f.read(
            1024 * 1024
        )

        if not chunk:
            break

        sha256.update(
            chunk
        )


file_hash = sha256.hexdigest()


print(
    "Checkpoint SHA256:",
    file_hash
)

print(
    "Expected SHA256:",
    EXPECTED_MODEL_SHA256
)


# ==================================================
# Verify checkpoint
# ==================================================

if file_hash != EXPECTED_MODEL_SHA256:

    raise RuntimeError(
        "Checkpoint SHA-256 mismatch. "
        f"Expected {EXPECTED_MODEL_SHA256}, "
        f"but received {file_hash}."
    )


print(
    "Checkpoint SHA-256 verified successfully."
)


# ==================================================
# Load checkpoint
# ==================================================

print("=" * 50)

print("Loading checkpoint...")

print("=" * 50)


checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)


# ==================================================
# Load mappings
# ==================================================

sex_mapping = checkpoint[
    "sex_mapping"
]

location_mapping = checkpoint[
    "location_mapping"
]


print(
    "Sex mapping:",
    sex_mapping
)

print(
    "Location mapping:",
    location_mapping
)


# ==================================================
# Create model
# ==================================================

model = MultimodalSkinCancerModel(

    num_sex_categories=
        len(sex_mapping),

    num_location_categories=
        len(location_mapping),

    num_classes=
        len(CLASS_NAMES)
)


# ==================================================
# Load trained weights
# ==================================================

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


print(
    "Model loaded successfully."
)

print(
    "Temperature:",
    TEMPERATURE
)

print("=" * 50)


# ==================================================
# Home
# ==================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status": "online",

        "service":
            "DermaVision AI API",

        "device":
            DEVICE.type,

        "temperature":
            TEMPERATURE
    })


# ==================================================
# Mappings
# ==================================================

@app.route(
    "/api/mappings",
    methods=["GET"]
)
def get_mappings():

    return jsonify({

        "sex_mapping":
            sex_mapping,

        "location_mapping":
            location_mapping
    })


# ==================================================
# Prediction
# ==================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        # ------------------------------------------
        # Image
        # ------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "error":
                    "No image was provided."

            }), 400


        image_file = (
            request.files["image"]
        )


        if image_file.filename == "":

            return jsonify({

                "error":
                    "No image was selected."

            }), 400


        # ------------------------------------------
        # Clinical information
        # ------------------------------------------

        age = request.form.get(
            "age"
        )

        sex = request.form.get(
            "sex"
        )

        location = request.form.get(
            "location"
        )


        if (
            age is None
            or sex is None
            or location is None
        ):

            return jsonify({

                "error":
                    "Age, sex, and location are required."

            }), 400


        # ------------------------------------------
        # Age
        # ------------------------------------------

        try:

            age = float(age)

        except (
            ValueError,
            TypeError
        ):

            return jsonify({

                "error":
                    "Age must be a valid number."

            }), 400


        if age < 0 or age > 120:

            return jsonify({

                "error":
                    "Age must be between 0 and 120."

            }), 400


        # ------------------------------------------
        # Normalize clinical information
        # ------------------------------------------

        age_value = (
            age / 100.0
        )


        sex = (
            sex
            .strip()
            .lower()
        )


        location = (
            location
            .strip()
            .lower()
        )


        # ------------------------------------------
        # Validate sex
        # ------------------------------------------

        if sex not in sex_mapping:

            return jsonify({

                "error":
                    f"Unknown sex category: {sex}"

            }), 400


        # ------------------------------------------
        # Validate location
        # ------------------------------------------

        if location not in location_mapping:

            return jsonify({

                "error":
                    f"Unknown location category: {location}"

            }), 400


        sex_value = sex_mapping[
            sex
        ]

        location_value = location_mapping[
            location
        ]


        # ------------------------------------------
        # Clinical tensor
        # ------------------------------------------

        clinical_tensor = torch.tensor(

            [[
                age_value,
                sex_value,
                location_value
            ]],

            dtype=torch.float32
        )


        # ------------------------------------------
        # Image
        # ------------------------------------------

        image = Image.open(
            image_file
        ).convert("RGB")


        image_tensor = (
            image_transform(
                image
            )
            .unsqueeze(0)
        )


        # ------------------------------------------
        # Inference
        # ------------------------------------------

        with torch.inference_mode():

            outputs = model(
                image_tensor,
                clinical_tensor
            )


            calibrated_probabilities = (
                torch.softmax(
                    outputs / TEMPERATURE,
                    dim=1
                )[0]
            )


        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        predicted_index = (
            torch.argmax(
                calibrated_probabilities
            ).item()
        )


        predicted_class = (
            CLASS_NAMES[
                predicted_index
            ]
        )


        confidence = (
            calibrated_probabilities[
                predicted_index
            ].item()
            * 100
        )


        probabilities = {

            CLASS_NAMES[i]:
                calibrated_probabilities[
                    i
                ].item() * 100

            for i in range(
                len(CLASS_NAMES)
            )
        }


        # ------------------------------------------
        # Response
        # ------------------------------------------

        return jsonify({

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "probabilities":
                probabilities,

            "temperature":
                TEMPERATURE,

            "device":
                DEVICE.type
        })


    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )


        return jsonify({

            "error":
                str(e)

        }), 500


# ==================================================
# Local development
# ==================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False
    )