import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENTS = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-Base",
}

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def load_history(experiment_name):
    history_path = Path("experiments") / experiment_name / "history.json"

    with open(history_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ====== СРАВНЕНИЕ VALIDATION ACCURACY ======

plt.figure(figsize=(9, 5))

for experiment_name, model_title in EXPERIMENTS.items():
    history = load_history(experiment_name)
    epochs = range(1, len(history["val_accuracy"]) + 1)

    plt.plot(
        epochs,
        history["val_accuracy"],
        marker="o",
        label=model_title
    )

plt.xlabel("Эпоха")
plt.ylabel("Validation Accuracy")
plt.title("Сравнение точности моделей на валидационной выборке")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "models_val_accuracy_comparison.png", dpi=300)
plt.close()


# ====== СРАВНЕНИЕ VALIDATION LOSS ======

plt.figure(figsize=(9, 5))

for experiment_name, model_title in EXPERIMENTS.items():
    history = load_history(experiment_name)
    epochs = range(1, len(history["val_loss"]) + 1)

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label=model_title
    )

plt.xlabel("Эпоха")
plt.ylabel("Validation Loss")
plt.title("Сравнение функции потерь моделей на валидационной выборке")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "models_val_loss_comparison.png", dpi=300)
plt.close()


print("Сравнительные графики сохранены:")
print(f"- {PLOTS_DIR / 'models_val_accuracy_comparison.png'}")
print(f"- {PLOTS_DIR / 'models_val_loss_comparison.png'}")