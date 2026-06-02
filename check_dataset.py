from pathlib import Path
from PIL import Image


DATASET_DIR = Path ("dataset")

SPLITS = ["train", "val"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def is_valid_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False
    
def check_dataset():
    if not DATASET_DIR.exists():
        print("Папка dataset не найдена")
        return
    
    total_images = 0

    for split in SPLITS:
        split_path = DATASET_DIR / split

        if not split_path.exists():
            print(f"Папка {split_path} не найдена")
            continue

        print(f"\n==={split.upper()} ===")

        class_dirs = [p for p in split_path.iterdir() if p.is_dir()]

        for class_dir in class_dirs:
            images = [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]

            valid_count = 0
            invalid_count = 0

            for image_path in images:
                if is_valid_image(image_path):
                    valid_count +=1
                else:
                    invalid_count +=1

            total_images += valid_count

            print(
                f"{class_dir.name}: "
                f"{valid_count} корректных изображений"
                + (f", {invalid_count} поврежденных" if invalid_count else "")
            )

    print(f"\n Всего корректных изображений: {total_images}")

if __name__ == "__main__":
    check_dataset()
