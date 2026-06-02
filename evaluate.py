from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DATASET_DIR = Path("dataset")
WEIGHTS_PATH = Path("weights") / "best_model.pth"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLASS_NAMES_RU = {
    "gremyachaya_tower": "Гремячая башня",
    "mirozhsky_monastery": "Мирожский монастырь",
    "pokrovskaya_tower": "Покровская башня",
    "troitsky_cathedral": "Троицкий собор",
    "varlaamovskaya_tower": "Варлаамовская башня",
}


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)

classes = checkpoint["classes"]

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

model.classifier[1] = nn.Linear(
    in_features=model.classifier[1].in_features,
    out_features=len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()


all_labels = []
all_predictions = []


with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.cpu().numpy())


accuracy = accuracy_score(all_labels, all_predictions)

target_names = [
    CLASS_NAMES_RU.get(class_name, class_name)
    for class_name in classes
]


print("\n=== ОБЩАЯ ТОЧНОСТЬ ===")
print(f"Accuracy: {accuracy:.4f}")


print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=target_names,
        digits=4
    )
)


print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(all_labels, all_predictions))