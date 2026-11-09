
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "model"

EFFICIENTNET_PATH = MODEL_DIR / "efficientnet_ai_detector.pth"
RESNET_PATH = MODEL_DIR / "resnet50_ai_detector.pth"

CLASS_NAMES = ["fake", "real"]
IMAGE_SIZE = 224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

app = FastAPI(title="ImageTruth - Explainable AI Image Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://image-truth-firy.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 2. MODEL ARCHITECTURES
# --------------------------------------------------

def create_efficientnet():
    network = models.efficientnet_b0(weights=None)
    network.classifier[1] = nn.Linear(
        network.classifier[1].in_features, 2
    )
    return network


def create_resnet():
    network = models.resnet50(weights=None)
    network.fc = nn.Linear(network.fc.in_features, 2)
    return network


# --------------------------------------------------
# 3. LOAD CHECKPOINTS
# --------------------------------------------------

def load_checkpoint(network, path):

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    # Use weights_only=False only for your own trusted
    # checkpoints created during your Colab training.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    network.load_state_dict(state_dict)

    # Check saved class order when checkpoint metadata exists.
    if isinstance(checkpoint, dict):
        saved_classes = checkpoint.get("class_names")
        if saved_classes is not None and list(saved_classes) != CLASS_NAMES:
            raise ValueError(
                f"Class order mismatch in {path.name}: "
                f"{saved_classes}. Expected {CLASS_NAMES}."
            )

    network.to(device)
    network.eval()

    return network


efficientnet = load_checkpoint(
    create_efficientnet(),
    EFFICIENTNET_PATH
)

resnet = load_checkpoint(
    create_resnet(),
    RESNET_PATH
)

print("EfficientNet-B0 loaded successfully.")
print("ResNet50 loaded successfully.")
print("Running on:", device)


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 7. PREDICTION HELPER
# --------------------------------------------------

def get_prediction(network, input_tensor):

    with torch.no_grad():
        output = network(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    scores = {
        CLASS_NAMES[i]: round(
            float(probabilities[i].item() * 100), 2
        )
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, scores


# --------------------------------------------------
# 8. MODEL RESULT (PREDICTION ONLY)
# --------------------------------------------------

def analyze_model(network, input_tensor):
    predicted_class, scores = get_prediction(
        network,
        input_tensor
    )

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "scores": scores
    }


# --------------------------------------------------
# 9. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "models_loaded": {
            "efficientnet_b0": True,
            "resnet50": True
        },
        "device": str(device),
        "classes": CLASS_NAMES
    }


# --------------------------------------------------
# 10. IMAGE PREDICTION + ENSEMBLE
# --------------------------------------------------

@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    if (
        image.content_type
        and not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:
        raw_bytes = await image.read()

        if not raw_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        original_image = Image.open(
            io.BytesIO(raw_bytes)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    except OSError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image could not be opened."
        )

    try:
        input_tensor = (
            preprocess(original_image)
            .unsqueeze(0)
            .to(device)
        )

        print("Starting EfficientNet analysis...", flush=True)

        efficientnet_result = analyze_model(
            efficientnet,
            input_tensor
        )

        print("EfficientNet prediction completed.", flush=True)

        print("Starting ResNet50 prediction...", flush=True)

        resnet_result = analyze_model(
            resnet,
            input_tensor
        )

        print("ResNet50 prediction completed.", flush=True)

        # ------------------------------------------
        # ENSEMBLE: EQUAL-WEIGHT SOFT VOTING
        # ------------------------------------------

        fake_probability = (
            efficientnet_result["scores"]["fake"]
            + resnet_result["scores"]["fake"]
        ) / 2

        real_probability = (
            efficientnet_result["scores"]["real"]
            + resnet_result["scores"]["real"]
        ) / 2

        ensemble_scores = {
            "fake": round(fake_probability, 2),
            "real": round(real_probability, 2)
        }

        ensemble_prediction = (
            "fake"
            if fake_probability >= real_probability
            else "real"
        )

        # ------------------------------------------
        # AGREEMENT CHECK
        # ------------------------------------------

        agreement = (
            efficientnet_result["prediction"]
            == resnet_result["prediction"]
        )

        explanation = (
            f"The ensemble classified this image as "
            f"{ensemble_prediction.upper()}. "
            f"EfficientNet-B0 predicted "
            f"{efficientnet_result['prediction'].upper()}, "
            f"while ResNet50 predicted "
            f"{resnet_result['prediction'].upper()}. "
            f"The final result is calculated using equal-weight "
            f"averaging of both models' softmax scores."
        )

        return {
            "prediction": ensemble_prediction,
            "scores": ensemble_scores,
            "explanation": explanation,

            # Individual model predictions and scores.
            "efficientnet": efficientnet_result,
            "resnet50": resnet_result,

            # Ensemble information.
            "ensemble": {
                "prediction": ensemble_prediction,
                "scores": ensemble_scores,
                "method": "Equal-weight soft voting"
            },

            "model_agreement": agreement,

            "warning": (
                "Model scores are not necessarily calibrated "
                "probabilities. Ensemble scores are the average "
                "of the two models' softmax outputs."
            )
        }

    except Exception as exc:
        print(
            f"PREDICTION ERROR: {type(exc).__name__}: {exc}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(exc)}"
        ) from exc
