import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader


# ====== НАСТРОЙКИ ======

DATASET_DIR = Path("dataset")

EXPERIMENT_DIR = Path("experiments") / "resnet50"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.00003
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЙ ======

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ====== ЗАГРУЗКА ДАТАСЕТА ======

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("Классы:", train_dataset.classes)
print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))
print("Device:", DEVICE)


# ====== МОДЕЛЬ RESNET-50 ======

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Сначала замораживаем все параметры
for param in model.parameters():
    param.requires_grad = False

# Размораживаем последний блок ResNet
for param in model.layer4.parameters():
    param.requires_grad = True

# Меняем последний классификационный слой
model.fc = nn.Linear(
    in_features=model.fc.in_features,
    out_features=NUM_CLASSES
)

# Новый classifier должен обучаться
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)


# ====== LOSS И OPTIMIZER ======

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)


# ====== ИСТОРИЯ ОБУЧЕНИЯ ======

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


# ====== ФУНКЦИЯ ОЦЕНКИ ======

def evaluate(model, dataloader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


# ====== ОБУЧЕНИЕ ======

best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    val_loss, val_accuracy = evaluate(model, val_loader)

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "classes": train_dataset.classes,
                "val_accuracy": best_val_accuracy
            },
            EXPERIMENT_DIR / "best_model.pth"
        )

        print(f"Лучшая модель сохранена. Val Acc: {best_val_accuracy:.4f}")


# ====== СОХРАНЕНИЕ ИСТОРИИ ======

with open(EXPERIMENT_DIR / "history.json", "w", encoding="utf-8") as file:
    json.dump(history, file, indent=4, ensure_ascii=False)


print("Обучение ResNet-50 завершено.")
print(f"Лучшая Val Accuracy: {best_val_accuracy:.4f}")
print(f"История обучения сохранена в: {EXPERIMENT_DIR / 'history.json'}")
print(f"Лучшая модель сохранена в: {EXPERIMENT_DIR / 'best_model.pth'}")