import json
from pathlib import Path

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


DATASET_DIR = Path("dataset")
EXPERIMENTS_DIR = Path("experiments")
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EXPERIMENTS = {
    "efficientnet_b0": {
        "title": "EfficientNet-B0",
        "path": EXPERIMENTS_DIR / "efficientnet_b0" / "best_model.pth",
    },
    "resnet50": {
        "title": "ResNet-50",
        "path": EXPERIMENTS_DIR / "resnet50" / "best_model.pth",
    },
    "vit_b_16": {
        "title": "ViT-Base",
        "path": EXPERIMENTS_DIR / "vit_b_16" / "best_model.pth",
    },
}


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=val_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


def create_model(model_name: str, num_classes: int):
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=num_classes,
        )

        return model

    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=num_classes,
        )

        return model

    if model_name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        model.heads.head = nn.Linear(
            in_features=model.heads.head.in_features,
            out_features=num_classes,
        )

        return model

    raise ValueError(f"Неизвестная модель: {model_name}")


def evaluate_model(model_name: str, model_info: dict):
    checkpoint_path = model_info["path"]

    if not checkpoint_path.exists():
        print(f"Файл модели не найден: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = create_model(model_name, num_classes)

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

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "model": model_info["title"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def main():
    results = {}

    print("\n=== СВОДНЫЕ МЕТРИКИ ПО МОДЕЛЯМ ===")
    print(f"{'Модель':<18} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    print("-" * 62)

    for model_name, model_info in EXPERIMENTS.items():
        result = evaluate_model(model_name, model_info)

        if result is not None:
            results[model_name] = result

            print(
                f"{result['model']:<18} "
                f"{result['accuracy']:<10.4f} "
                f"{result['precision']:<10.4f} "
                f"{result['recall']:<10.4f} "
                f"{result['f1_score']:<10.4f}"
            )

    summary_path = EXPERIMENTS_DIR / "metrics_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\nИтоговые результаты сохранены:")
    print(summary_path)


if __name__ == "__main__":
    main()