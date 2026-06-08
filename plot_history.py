import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENT_NAME = "vit_b_16"

MODEL_TITLES = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-Base"
}

MODEL_TITLE = MODEL_TITLES.get(EXPERIMENT_NAME, EXPERIMENT_NAME)

HISTORY_PATH = Path("experiments") / EXPERIMENT_NAME / "history.json"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


with open(HISTORY_PATH, "r", encoding="utf-8") as file:
    history = json.load(file)


epochs = range(1, len(history["train_loss"]) + 1)


# ====== ГРАФИК LOSS ======

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.title(f"Динамика функции потерь {MODEL_TITLE}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"{EXPERIMENT_NAME}_loss.png", dpi=300)
plt.close()


# ====== ГРАФИК ACCURACY ======

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.title(f"Динамика точности {MODEL_TITLE}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"{EXPERIMENT_NAME}_accuracy.png", dpi=300)
plt.close()


print("Графики сохранены в папку plots/")
print(f"- {PLOTS_DIR / f'{EXPERIMENT_NAME}_loss.png'}")
print(f"- {PLOTS_DIR / f'{EXPERIMENT_NAME}_accuracy.png'}")