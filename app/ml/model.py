from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from app.ml.preprocess import transform

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "weights" / "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES_RU = {
    "gremyachaya_tower": "Гремячая башня",
    "mirozhsky_monastery": "Мирожский монастырь",
    "pokrovskaya_tower": "Покровская башня",
    "troitsky_cathedral": "Троицкий собор",
    "varlaamovskaya_tower": "Варлаамовская башня",
}

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    classes = checkpoint["classes"]

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    model.classifier[1] = nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=len(classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, classes

model, classes = load_model()

def predict_image(image_path: str):
    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    predicted_index = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_index].item()

    class_name = classes[predicted_index]
    class_name_ru = CLASS_NAMES_RU.get(class_name, class_name)

    top_predictions = []

    top_values, top_indices = torch.topk(probabilities, k=min(3, len(classes)))

    for value, index in zip(top_values, top_indices):
        technical_name = classes[index.item()]
        ru_name = CLASS_NAMES_RU.get(technical_name, technical_name)

        top_predictions.append({
            "class": ru_name,
            "technical_class": technical_name,
            "confidence": round(value.item() * 100, 2)
        })


    return {
        "class": class_name_ru,
        "technical_class": class_name,
        "confidence": round(confidence * 100, 2),
        "top_predictions": top_predictions
    }