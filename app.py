import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image,ImageTk
import torch
from torchvision import transforms
from model import MultimodalSkinCancerModel

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT="outputs/multimodal_best_model.pth"
TEMPERATURE_FILE="outputs/temperature.json"

CLASS_NAMES=[
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

CLASS_DISPLAY={
    "akiec":"Actinic Keratoses",
    "bcc":"Basal Cell Carcinoma",
    "bkl":"Benign Keratosis",
    "df":"Dermatofibroma",
    "mel":"Melanoma",
    "nv":"Melanocytic Nevus",
    "vasc":"Vascular Lesion"
}

# Higher-level interpretation for UI only
CLASS_CATEGORY={
    "akiec":"Suspicious / Precancerous",
    "bcc":"Malignant",
    "bkl":"Benign",
    "df":"Benign",
    "mel":"Malignant",
    "nv":"Benign",
    "vasc":"Benign"
}

# Image preprocessing must match validation/inference
IMAGE_TRANSFORM=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{CHECKPOINT}"
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

# ============================================================
# LOAD CALIBRATION TEMPERATURE
# ============================================================

temperature=1.0

if os.path.exists(TEMPERATURE_FILE):
    with open(
        TEMPERATURE_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        temperature_data=json.load(f)

    temperature=float(
        temperature_data["temperature"]
    )

# ============================================================
# GLOBAL IMAGE
# ============================================================

selected_image_path=None
preview_image=None

# ============================================================
# PREDICTION
# ============================================================

def predict():
    global selected_image_path

    if selected_image_path is None:
        messagebox.showwarning(
            "Image Required",
            "Please upload a skin lesion image first."
        )
        return

    age_text=age_entry.get().strip()

    if not age_text:
        messagebox.showwarning(
            "Age Required",
            "Please enter the patient's age."
        )
        return

    try:
        age=float(age_text)

        if age<0 or age>120:
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Age",
            "Please enter a valid age between 0 and 120."
        )
        return

    sex=sex_var.get().strip()
    location=location_var.get().strip()

    if not sex:
        messagebox.showwarning(
            "Sex Required",
            "Please select sex."
        )
        return

    if not location:
        messagebox.showwarning(
            "Location Required",
            "Please select lesion location."
        )
        return

    try:
        image=Image.open(
            selected_image_path
        ).convert("RGB")

        image_tensor=IMAGE_TRANSFORM(
            image
        ).unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        age_value=max(
            0.0,
            min(
                age/100.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # SEX
        # ----------------------------------------------------

        sex_index=sex_mapping.get(
            sex,
            sex_mapping.get("unknown",0)
        )

        if len(sex_mapping)>1:
            sex_value=float(sex_index)/float(
                len(sex_mapping)-1
            )
        else:
            sex_value=0.0

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        location_index=location_mapping.get(
            location,
            location_mapping.get("unknown",0)
        )

        if len(location_mapping)>1:
            location_value=float(location_index)/float(
                len(location_mapping)-1
            )
        else:
            location_value=0.0

        clinical_tensor=torch.tensor(
            [
                age_value,
                sex_value,
                location_value
            ],
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        # ----------------------------------------------------
        # MODEL INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            logits=model(
                image_tensor,
                clinical_tensor
            )

            # Temperature scaling
            calibrated_logits=logits/temperature

            probabilities=torch.softmax(
                calibrated_logits,
                dim=1
            )

            predicted_class=int(
                probabilities.argmax(
                    dim=1
                ).item()
            )

            confidence=float(
                probabilities[
                    0,
                    predicted_class
                ].item()
            )

        predicted_name=CLASS_NAMES[
            predicted_class
        ]

        display_name=CLASS_DISPLAY[
            predicted_name
        ]

        category=CLASS_CATEGORY[
            predicted_name
        ]

        # ----------------------------------------------------
        # UPDATE RESULT
        # ----------------------------------------------------

        prediction_label.config(
            text=display_name.upper()
        )

        confidence_label.config(
            text=f"Confidence: {confidence*100:.2f}%"
        )

        category_label.config(
            text=f"Category: {category}"
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        for widget in probability_frame.winfo_children():
            widget.destroy()

        sorted_results=sorted(
            zip(
                CLASS_NAMES,
                probabilities[0].cpu().tolist()
            ),
            key=lambda x:x[1],
            reverse=True
        )

        for class_name,prob in sorted_results:

            row=tk.Frame(
                probability_frame,
                bg="#18202b"
            )

            row.pack(
                fill="x",
                padx=10,
                pady=3
            )

            name=tk.Label(
                row,
                text=CLASS_DISPLAY[class_name],
                font=("Segoe UI",10),
                fg="white",
                bg="#18202b",
                anchor="w"
            )

            name.pack(
                side="left",
                fill="x",
                expand=True
            )

            value=tk.Label(
                row,
                text=f"{prob*100:.2f}%",
                font=("Segoe UI",10,"bold"),
                fg="#55d6be",
                bg="#18202b"
            )

            value.pack(
                side="right"
            )

        status_label.config(
            text="Analysis completed successfully.",
            fg="#55d6be"
        )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )

# ============================================================
# IMAGE UPLOAD
# ============================================================

def upload_image():

    global selected_image_path
    global preview_image

    path=filedialog.askopenfilename(
        title="Select Skin Lesion Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            )
        ]
    )

    if not path:
        return

    selected_image_path=path

    try:

        image=Image.open(
            path
        ).convert("RGB")

        preview=image.copy()
        preview.thumbnail(
            (420,300)
        )

        preview_image=ImageTk.PhotoImage(
            preview
        )

        image_label.config(
            image=preview_image,
            text=""
        )

        filename=os.path.basename(path)

        image_name_label.config(
            text=filename
        )

        status_label.config(
            text="Image loaded. Enter clinical information and analyze.",
            fg="#ffffff"
        )

    except Exception as e:

        messagebox.showerror(
            "Image Error",
            str(e)
        )

# ============================================================
# CLEAR
# ============================================================

def clear_all():

    global selected_image_path
    global preview_image

    selected_image_path=None
    preview_image=None

    image_label.config(
        image="",
        text="No image selected"
    )

    image_name_label.config(
        text="No image selected"
    )

    age_entry.delete(
        0,
        tk.END
    )

    sex_var.set(
        ""
    )

    location_var.set(
        ""
    )

    prediction_label.config(
        text="—"
    )

    confidence_label.config(
        text="Confidence: —"
    )

    category_label.config(
        text="Category: —"
    )

    for widget in probability_frame.winfo_children():
        widget.destroy()

    status_label.config(
        text="Ready for analysis.",
        fg="#ffffff"
    )

# ============================================================
# MAIN WINDOW
# ============================================================

root=tk.Tk()

root.title(
    "Multimodal AI Framework for Early Skin Cancer Diagnosis"
)

root.geometry(
    "1100x750"
)

root.minsize(
    950,
    650
)

root.configure(
    bg="#0f1720"
)

# ============================================================
# HEADER
# ============================================================

header=tk.Frame(
    root,
    bg="#111c29",
    height=90
)

header.pack(
    fill="x"
)

title=tk.Label(
    header,
    text="Multimodal AI Framework",
    font=("Segoe UI",24,"bold"),
    fg="white",
    bg="#111c29"
)

title.pack(
    pady=(15,0)
)

subtitle=tk.Label(
    header,
    text="Early Skin Cancer Diagnosis",
    font=("Segoe UI",12),
    fg="#9caec1",
    bg="#111c29"
)

subtitle.pack()

# ============================================================
# MAIN CONTAINER
# ============================================================

main=tk.Frame(
    root,
    bg="#0f1720"
)

main.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=20
)

# ============================================================
# LEFT PANEL
# ============================================================

left=tk.Frame(
    main,
    bg="#18202b"
)

left.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0,10)
)

image_title=tk.Label(
    left,
    text="LESION IMAGE",
    font=("Segoe UI",13,"bold"),
    fg="white",
    bg="#18202b"
)

image_title.pack(
    pady=15
)

image_label=tk.Label(
    left,
    text="No image selected",
    font=("Segoe UI",12),
    fg="#728197",
    bg="#111820",
    width=45,
    height=15
)

image_label.pack(
    padx=20,
    pady=10
)

image_name_label=tk.Label(
    left,
    text="No image selected",
    font=("Segoe UI",9),
    fg="#9caec1",
    bg="#18202b"
)

image_name_label.pack(
    pady=5
)

upload_button=tk.Button(
    left,
    text="UPLOAD IMAGE",
    command=upload_image,
    font=("Segoe UI",11,"bold"),
    fg="white",
    bg="#2563eb",
    activebackground="#1d4ed8",
    activeforeground="white",
    relief="flat",
    padx=20,
    pady=10,
    cursor="hand2"
)

upload_button.pack(
    pady=15
)

# ============================================================
# CLINICAL INPUT
# ============================================================

clinical_title=tk.Label(
    left,
    text="CLINICAL INFORMATION",
    font=("Segoe UI",13,"bold"),
    fg="white",
    bg="#18202b"
)

clinical_title.pack(
    pady=(15,10)
)

clinical_frame=tk.Frame(
    left,
    bg="#18202b"
)

clinical_frame.pack(
    padx=30,
    fill="x"
)

# Age
tk.Label(
    clinical_frame,
    text="Age",
    font=("Segoe UI",10),
    fg="#d5deea",
    bg="#18202b"
).grid(
    row=0,
    column=0,
    sticky="w",
    pady=7
)

age_entry=tk.Entry(
    clinical_frame,
    font=("Segoe UI",10),
    bg="#111820",
    fg="white",
    insertbackground="white",
    relief="flat"
)

age_entry.grid(
    row=0,
    column=1,
    sticky="ew",
    padx=15
)

# Sex
tk.Label(
    clinical_frame,
    text="Sex",
    font=("Segoe UI",10),
    fg="#d5deea",
    bg="#18202b"
).grid(
    row=1,
    column=0,
    sticky="w",
    pady=7
)

sex_var=tk.StringVar()

sex_menu=tk.OptionMenu(
    clinical_frame,
    sex_var,
    "",
    *sorted(sex_mapping.keys())
)

sex_menu.config(
    font=("Segoe UI",10),
    bg="#111820",
    fg="white",
    activebackground="#263445",
    activeforeground="white",
    relief="flat",
    highlightthickness=0
)

sex_menu["menu"].config(
    bg="#111820",
    fg="white"
)

sex_menu.grid(
    row=1,
    column=1,
    sticky="ew",
    padx=15
)

# Location
tk.Label(
    clinical_frame,
    text="Location",
    font=("Segoe UI",10),
    fg="#d5deea",
    bg="#18202b"
).grid(
    row=2,
    column=0,
    sticky="w",
    pady=7
)

location_var=tk.StringVar()

location_menu=tk.OptionMenu(
    clinical_frame,
    location_var,
    "",
    *sorted(location_mapping.keys())
)

location_menu.config(
    font=("Segoe UI",10),
    bg="#111820",
    fg="white",
    activebackground="#263445",
    activeforeground="white",
    relief="flat",
    highlightthickness=0
)

location_menu["menu"].config(
    bg="#111820",
    fg="white"
)

location_menu.grid(
    row=2,
    column=1,
    sticky="ew",
    padx=15
)

clinical_frame.columnconfigure(
    1,
    weight=1
)

# ============================================================
# BUTTONS
# ============================================================

button_frame=tk.Frame(
    left,
    bg="#18202b"
)

button_frame.pack(
    pady=20
)

analyze_button=tk.Button(
    button_frame,
    text="ANALYZE LESION",
    command=predict,
    font=("Segoe UI",11,"bold"),
    fg="white",
    bg="#16a34a",
    activebackground="#15803d",
    activeforeground="white",
    relief="flat",
    padx=25,
    pady=10,
    cursor="hand2"
)

analyze_button.pack(
    side="left",
    padx=5
)

clear_button=tk.Button(
    button_frame,
    text="CLEAR",
    command=clear_all,
    font=("Segoe UI",11),
    fg="white",
    bg="#374151",
    activebackground="#4b5563",
    activeforeground="white",
    relief="flat",
    padx=25,
    pady=10,
    cursor="hand2"
)

clear_button.pack(
    side="left",
    padx=5
)

# ============================================================
# RIGHT PANEL
# ============================================================

right=tk.Frame(
    main,
    bg="#18202b"
)

right.pack(
    side="right",
    fill="both",
    expand=True,
    padx=(10,0)
)

result_title=tk.Label(
    right,
    text="AI ANALYSIS RESULT",
    font=("Segoe UI",16,"bold"),
    fg="white",
    bg="#18202b"
)

result_title.pack(
    pady=20
)

prediction_label=tk.Label(
    right,
    text="—",
    font=("Segoe UI",25,"bold"),
    fg="#55d6be",
    bg="#18202b"
)

prediction_label.pack(
    pady=5
)

confidence_label=tk.Label(
    right,
    text="Confidence: —",
    font=("Segoe UI",13),
    fg="white",
    bg="#18202b"
)

confidence_label.pack(
    pady=5
)

category_label=tk.Label(
    right,
    text="Category: —",
    font=("Segoe UI",12,"bold"),
    fg="#f0c674",
    bg="#18202b"
)

category_label.pack(
    pady=5
)

# ============================================================
# PROBABILITIES
# ============================================================

probability_title=tk.Label(
    right,
    text="CLASS PROBABILITIES",
    font=("Segoe UI",12,"bold"),
    fg="white",
    bg="#18202b"
)

probability_title.pack(
    pady=(30,10)
)

probability_frame=tk.Frame(
    right,
    bg="#18202b"
)

probability_frame.pack(
    fill="x",
    padx=15
)

# ============================================================
# MODEL INFORMATION
# ============================================================

info_frame=tk.Frame(
    right,
    bg="#111820"
)

info_frame.pack(
    fill="x",
    padx=20,
    pady=25
)

tk.Label(
    info_frame,
    text="MODEL",
    font=("Segoe UI",9,"bold"),
    fg="#728197",
    bg="#111820"
).pack(
    pady=(10,0)
)

tk.Label(
    info_frame,
    text="EfficientNet-B0 + Clinical Encoder",
    font=("Segoe UI",10),
    fg="white",
    bg="#111820"
).pack()

tk.Label(
    info_frame,
    text="IMAGE + AGE + SEX + LOCATION",
    font=("Segoe UI",9),
    fg="#9caec1",
    bg="#111820"
).pack(
    pady=(3,10)
)

# ============================================================
# STATUS
# ============================================================

status_label=tk.Label(
    root,
    text="Ready for analysis.",
    font=("Segoe UI",9),
    fg="white",
    bg="#0f1720"
)

status_label.pack(
    pady=(0,10)
)

# ============================================================
# DISCLAIMER
# ============================================================

disclaimer=tk.Label(
    root,
    text="AI-assisted classification only. This result is not a definitive medical diagnosis.",
    font=("Segoe UI",9),
    fg="#728197",
    bg="#0f1720"
)

disclaimer.pack(
    pady=(0,10)
)

# ============================================================
# START APPLICATION
# ============================================================

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Calibration temperature: {temperature:.4f}"
)

root.mainloop()