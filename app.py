import os
import hashlib
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

from model import MultimodalSkinCancerModel


app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "multimodal_best_model.pth"
)

TEMPERATURE = 1.1264986991882324

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


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Checkpoint diagnostics
# --------------------------------------------------

print("========================================")
print("DermaVision AI API")
print("========================================")

print("Device:", DEVICE)
print("Checkpoint path:", CHECKPOINT)
print("Checkpoint exists:", os.path.exists(CHECKPOINT))


if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )


# Check file size
file_size = os.path.getsize(CHECKPOINT)

print("Checkpoint size:", file_size, "bytes")


# Calculate SHA-256
print("Calculating checkpoint SHA256...")

sha256 = hashlib.sha256()

with open(CHECKPOINT, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        sha256.update(chunk)

file_hash = sha256.hexdigest()

print("Checkpoint SHA256:", file_hash)

print("Expected SHA256:")
print(
    "3f6255cb7c94be9d526ec8c9d56d561e06552898431adaafbc11575a0fb34bfc"
)

print("========================================")
print("Loading checkpoint...")
print("========================================")


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


# --------------------------------------------------
# Load mappings saved with the model
# --------------------------------------------------

sex_mapping = checkpoint["sex_mapping"]
location_mapping = checkpoint["location_mapping"]


print("Sex mapping:", sex_mapping)
print("Location mapping:", location_mapping)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = MultimodalSkinCancerModel(
    num_sex_categories=len(sex_mapping),
    num_location_categories=len(location_mapping),
    num_classes=len(CLASS_NAMES)
).to(DEVICE)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print("Model loaded successfully.")
print("========================================")


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "online",
        "service": "DermaVision AI API"
    })


# --------------------------------------------------
# Mapping endpoint
# --------------------------------------------------

@app.route("/api/mappings", methods=["GET"])
def get_mappings():

    return jsonify({
        "sex_mapping": sex_mapping,
        "location_mapping": location_mapping
    })


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------------
        # Check image
        # ------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "error": "No image was provided."
            }), 400


        image_file = request.files["image"]


        if image_file.filename == "":

            return jsonify({
                "error": "No image was selected."
            }), 400


        # ------------------------------------------
        # Get clinical information
        # ------------------------------------------

        age = request.form.get("age")
        sex = request.form.get("sex")
        location = request.form.get("location")


        if age is None or sex is None or location is None:

            return jsonify({
                "error": "Age, sex, and location are required."
            }), 400


        # ------------------------------------------
        # Validate age
        # ------------------------------------------

        try:

            age = float(age)

        except ValueError:

            return jsonify({
                "error": "Age must be a valid number."
            }), 400


        if age < 0 or age > 120:

            return jsonify({
                "error": "Age must be between 0 and 120."
            }), 400


        # ------------------------------------------
        # Normalize clinical data
        # ------------------------------------------

        age_value = age / 100.0

        sex = sex.strip().lower()
        location = location.strip().lower()


        # ------------------------------------------
        # Validate mappings
        # ------------------------------------------

        if sex not in sex_mapping:

            return jsonify({
                "error": f"Unknown sex category: {sex}"
            }), 400


        if location not in location_mapping:

            return jsonify({
                "error": f"Unknown location category: {location}"
            }), 400


        sex_value = sex_mapping[sex]
        location_value = location_mapping[location]


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
        ).to(DEVICE)


        # ------------------------------------------
        # Process image
        # ------------------------------------------

        image = Image.open(
            image_file
        ).convert("RGB")


        image_tensor = image_transform(
            image
        ).unsqueeze(0).to(DEVICE)


        # ------------------------------------------
        # Model inference
        # ------------------------------------------

        with torch.no_grad():

            outputs = model(
                image_tensor,
                clinical_tensor
            )


            calibrated_probabilities = torch.softmax(
                outputs / TEMPERATURE,
                dim=1
            )[0]


        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        predicted_index = torch.argmax(
            calibrated_probabilities
        ).item()


        predicted_class = CLASS_NAMES[
            predicted_index
        ]


        confidence = (
            calibrated_probabilities[
                predicted_index
            ].item() * 100
        )


        probabilities = {
            CLASS_NAMES[i]:
            calibrated_probabilities[i].item() * 100

            for i in range(len(CLASS_NAMES))
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
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run server
# --------------------------------------------------

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