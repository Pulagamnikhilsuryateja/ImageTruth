

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# --------------------------------------------------
# 1. PATHS AND MODEL CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "efficientnet_ai_detector.pth"
# ImageFolder sorts folder names alphabetically:
# fake -> 0, real -> 1
CLASS_NAMES = ["fake", "real"]
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Explainable AI Image Detection API")


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
# 2. CREATE THE SAME EFFICIENTNET-B0 ARCHITECTURE
# --------------------------------------------------

def create_model():
    # No pretrained download is needed here: the checkpoint
    # supplies the trained weights.
    network = models.efficientnet_b0(weights=None)

    num_features = network.classifier[1].in_features
    network.classifier[1] = nn.Linear(num_features, 2)

    return network


def load_checkpoint(network, path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    # Only load a checkpoint file you trust.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    # Support a raw state_dict or a common checkpoint dictionary.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format. "
            "Expected a model state_dict."
        )

    # Handle checkpoints saved with DataParallel.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    network.load_state_dict(state_dict)
    return network


model = create_model()
model = load_checkpoint(model, MODEL_PATH)
model.to(device)
model.eval()


# --------------------------------------------------
# 3. IMAGE PREPROCESSING
# --------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 4. HELPER: ENCODE AN IMAGE AS BASE64 PNG
# --------------------------------------------------

def image_to_base64(image_array):
    image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# --------------------------------------------------
# 5. GENERATE GRAD-CAM
# --------------------------------------------------

def generate_gradcam(input_tensor, original_image, target_class):
    target_layer = model.features[-1]
    captured_activations = []
    captured_gradients = []

    def forward_hook(module, inputs, output):
        captured_activations.append(output)
        output.register_hook(
            lambda grad: captured_gradients.append(grad)
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output = model(input_tensor)
            output[0, target_class].backward()

        activations = captured_activations[0].detach()
        gradients = captured_gradients[0].detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = torch.relu(
            (weights * activations).sum(dim=1, keepdim=True)
        )

        cam = torch.nn.functional.interpolate(
            cam,
            size=(original_image.height, original_image.width),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

    finally:
        handle.remove()

    original = np.array(original_image.convert("RGB"))

    heatmap_bgr = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET,
    )
    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    overlay = np.clip(
        0.6 * original + 0.4 * heatmap_rgb,
        0,
        255,
    ).astype(np.uint8)

    return cam, heatmap_rgb, overlay


# --------------------------------------------------
# 6. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
        "classes": CLASS_NAMES,
    }

# --------------------------------------------------
# 7. IMAGE PREDICTION + EXPLANATION
# --------------------------------------------------

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    print("PREDICT: request received", flush=True)

    # 1. Validate uploaded file
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:
        # 2. Read and validate image
        raw_bytes = await image.read()

        print(
            f"PREDICT: image received, size={len(raw_bytes)} bytes",
            flush=True,
        )

        if not raw_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        original_image = Image.open(
            io.BytesIO(raw_bytes)
        ).convert("RGB")

        print(
            f"PREDICT: image decoded, size={original_image.size}",
            flush=True,
        )

        # 3. Preprocess image
        input_tensor = preprocess(original_image).unsqueeze(0).to(device)

        print("PREDICT: preprocessing finished", flush=True)

        # 5. Run model inference
        print("PREDICT: starting model inference", flush=True)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        print("PREDICT: model inference finished", flush=True)

        predicted_class = int(torch.argmax(probabilities).item())
        predicted_label = CLASS_NAMES[predicted_class]

        scores = {
            CLASS_NAMES[i]: round(
                float(probabilities[i].item() * 100), 2
            )
            for i in range(len(CLASS_NAMES))
        }

        # 5. Generate real Grad-CAM explanation images
        print("PREDICT: starting Grad-CAM", flush=True)

        cam, heatmap, overlay = generate_gradcam(
            input_tensor,
            original_image,
            predicted_class,
        )

        print("PREDICT: Grad-CAM finished", flush=True)

        highlighted_ratio = float(np.mean(cam >= 0.6) * 100)

        explanation = (
            f"The model classified this image as "
            f"{predicted_label.upper()} based on patterns learned "
            f"during training. Its score for this class is "
            f"{scores[predicted_label]:.2f}%. "
            f"Grad-CAM highlights regions that influenced this "
            f"prediction; approximately {highlighted_ratio:.2f}% "
            f"of pixels have a normalized heatmap value of at least "
            f"0.6. This is a visualization statistic, not a measure "
            f"of the percentage of AI-generated content. The heatmap "
            f"does not prove the image's origin."
        )

        # 7. Encode Grad-CAM images
        print("PREDICT: encoding explanation images", flush=True)

        heatmap_base64 = image_to_base64(heatmap)
        overlay_base64 = image_to_base64(overlay)

        print("PREDICT: preparing response", flush=True)

        # 8. Return prediction
        return {
            "prediction": predicted_label,
            "scores": scores,
            "explanation": explanation,
            "gradcam_heatmap": heatmap_base64,
            "gradcam_overlay": overlay_base64,
            "warning": (
                "Scores are model outputs, not necessarily calibrated "
                "probabilities. Performance on image types outside "
                "the training data is not guaranteed."
            ),
        }

    except HTTPException:
        raise

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except Exception as exc:
        print(
            f"PREDICT ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(exc)}",
        ) from exc
