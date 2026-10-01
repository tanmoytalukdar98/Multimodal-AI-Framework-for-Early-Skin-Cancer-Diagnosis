import os
import sys
import json
import uuid
import tempfile
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs
import urllib.request
import torch
from PIL import Image
from torchvision import transforms

from model import MultimodalSkinCancerModel

# ============================================================
# CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 8765
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "outputs/multimodal_best_model.pth"
TEMPERATURE_FILE = "outputs/temperature.json"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# MODEL LOADING
# ============================================================

print("Loading model...")
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
sex_mapping = checkpoint["sex_mapping"]
location_mapping = checkpoint["location_mapping"]

model = MultimodalSkinCancerModel(
    num_sex_categories=len(sex_mapping),
    num_location_categories=len(location_mapping),
    num_classes=len(CLASS_NAMES)
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

temperature = 1.0
if os.path.exists(TEMPERATURE_FILE):
    with open(TEMPERATURE_FILE, "r", encoding="utf-8") as f:
        temperature = float(json.load(f)["temperature"])

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Temperature: {temperature:.4f}")
print(f"Sex mappings: {sex_mapping}")
print(f"Location mappings: {location_mapping}")

# ============================================================
# INFERENCE
# ============================================================

def run_inference(image_path, age, sex, location):
    image = Image.open(image_path).convert("RGB")
    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    age_value = float(age) / 100.0
    age_value = max(0.0, min(age_value, 1.0))
    sex_index = sex_mapping.get(sex, sex_mapping.get("unknown", 0))
    location_index = location_mapping.get(
        location, location_mapping.get("unknown", 0)
    )

    clinical_tensor = torch.tensor(
        [[age_value, sex_index, location_index]],
        dtype=torch.float32, device=DEVICE
    )

    with torch.no_grad():
        logits = model(image_tensor, clinical_tensor)
        calibrated = logits / temperature
        probs = torch.softmax(calibrated, dim=1)[0]

    predicted_class = int(probs.argmax().item())
    confidence = float(probs[predicted_class].item())

    all_probs = {}
    for i, name in enumerate(CLASS_NAMES):
        all_probs[name] = round(float(probs[i].item()) * 100, 4)

    return {
        "predicted_class": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "probabilities": all_probs,
        "temperature": round(temperature, 4),
        "device": str(DEVICE),
        "image_size": f"{image.size[0]}x{image.size[1]}"
    }

# ============================================================
# HTTP HANDLER
# ============================================================

class RequestHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.glb': 'model/gltf-binary', '.gltf': 'model/gltf+json'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=".", **kwargs)

    def end_headers(self):
        if self.path.split("?")[0].endswith(".html") or self.path == "/":
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()
        elif self.path == "/api/mappings":
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            self._json_response(200, {
                "sex_mapping": sex_mapping,
                "location_mapping": location_mapping,
                "device": str(DEVICE),
                "gpu_name": gpu_name,
                "temperature": temperature,
                "classes": CLASS_NAMES
            })
            return
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            self._handle_predict()
        elif self.path in ("/api/chat", "/api/assist"):
            self._handle_chat()
        else:
            self.send_error(404)

    def _handle_chat(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            content_length = 0
        try:
            data = json.loads(self.rfile.read(content_length).decode("utf-8") or "{}")
        except Exception:
            self._json_response(400, {"error": "Invalid JSON body"})
            return
        message = data.get("message", "")
        history = data.get("history", [])
        context = data.get("context", "")
        if not str(message).strip():
            self._json_response(400, {"error": "Empty message"})
            return
        self._json_response(200, gemini_chat(message, history, context))

    def _handle_predict(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json_response(400, {"error": "Expected multipart/form-data"})
            return

        boundary = content_type.split("boundary=")[1].strip()
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        parts = self._parse_multipart(body, boundary)

        if "image" not in parts:
            self._json_response(400, {"error": "No image provided"})
            return

        age = parts.get("age", ["45"])[0]
        sex = parts.get("sex", ["unknown"])[0]
        location = parts.get("location", ["unknown"])[0]

        try:
            age = float(age)
            if age < 0 or age > 120:
                raise ValueError
        except (ValueError, TypeError):
            self._json_response(400, {"error": "Invalid age"})
            return

        tmp_path = None
        try:
            image_data = parts["image"][1]
            ext = parts["image"][0].split(".")[-1].lower()
            if ext not in ("jpg", "jpeg", "png", "bmp", "webp"):
                ext = "jpg"
            tmp_path = os.path.join(
                tempfile.gettempdir(), f"skin_{uuid.uuid4().hex[:8]}.{ext}"
            )
            with open(tmp_path, "wb") as f:
                f.write(image_data)

            result = run_inference(tmp_path, age, sex, location)
            self._json_response(200, result)

        except Exception as e:
            self._json_response(500, {"error": str(e)})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _parse_multipart(self, body, boundary):
        parts = {}
        boundary_bytes = boundary.encode()
        splits = body.split(b"--" + boundary_bytes)

        for part in splits[1:]:
            if part.strip() == b"" or part.strip() == b"--":
                continue
            if b"\r\n\r\n" not in part:
                continue
            header_section, content = part.split(b"\r\n\r\n", 1)
            if content.endswith(b"\r\n"):
                content = content[:-2]
            header_text = header_section.decode("utf-8", errors="replace")
            name = None
            filename = None
            for line in header_text.split("\r\n"):
                if "Content-Disposition" in line:
                    for token in line.split(";"):
                        token = token.strip()
                        if token.startswith("name="):
                            name = token.split("=", 1)[1].strip('"')
                        elif token.startswith("filename="):
                            filename = token.split("=", 1)[1].strip('"')
            if name:
                if name not in parts:
                    parts[name] = []
                if filename:
                    parts[name] = (filename, content)
                else:
                    parts[name].append(content.decode("utf-8", errors="replace"))
        return parts

    def _json_response(self, code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        if "/api/" in str(args[0]):
            print(f"[API] {args[0]}")


# ============================================================
# START SERVER
# ============================================================

def open_browser():
    import time
    time.sleep(1.2)
    webbrowser.open(f"http://{HOST}:{PORT}")

if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), RequestHandler)
    print(f"\nServer running at http://{HOST}:{PORT}\n")
    threading.Thread(target=open_browser, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
