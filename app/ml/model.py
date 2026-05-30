import torch

from PIL import Image

from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights

from app.ml.preprocess import transform

weights = EfficientNet_B0_Weights.DEFAULT

categories = weights.meta["categories"]

model = efficientnet_b0(weights=weights)

model.eval()

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image_tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    predicted_class = outputs.argmax(1).item()

    confidence = probabilities[predicted_class].item()

    class_name = categories[predicted_class]

    return {
        "class": class_name,
        "confidence": round(confidence * 100, 2)
    }