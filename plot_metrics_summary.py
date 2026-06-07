import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS_PATH = Path("experiments") / "metrics_summary.json"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


with open(METRICS_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)


models = []
accuracy = []
precision = []
recall = []
f1_score = []

for model_key, metrics in results.items():
    models.append(metrics["model"])
    accuracy.append(metrics["accuracy"])
    precision.append(metrics["precision"])
    recall.append(metrics["recall"])
    f1_score.append(metrics["f1_score"])


x = range(len(models))
width = 0.2

plt.figure(figsize=(10, 6))

plt.bar([i - 1.5 * width for i in x], accuracy, width, label="Accuracy")
plt.bar([i - 0.5 * width for i in x], precision, width, label="Precision")
plt.bar([i + 0.5 * width for i in x], recall, width, label="Recall")
plt.bar([i + 1.5 * width for i in x], f1_score, width, label="F1-score")

plt.xticks(list(x), models) 
plt.ylim(0, 1.05)
plt.ylabel("Значение метрики")
plt.title("Сравнение моделей по основным метрикам качества")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

output_path = PLOTS_DIR / "models_metrics_comparison.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"График сохранен: {output_path}")