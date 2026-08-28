import os
import random
import shutil
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
VAL_DIR = "/home/malaika/imagenet/imagenet/val"          # original ImageNet val folder
OUT_DIR = "./imagenet_val_100cls_50imgs"

NUM_CLASSES = 100
IMAGES_PER_CLASS = 50
SEED = 42

COPY_IMAGES = True   # True = copy files, False = symlink files

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}


# --------------------------------------------------
# SCRIPT
# --------------------------------------------------
def get_image_files(class_dir):
    return [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    ]


def main():
    random.seed(SEED)

    val_path = Path(VAL_DIR)
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    if not val_path.exists():
        raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

    # Get class folders
    class_dirs = sorted([
        p for p in val_path.iterdir()
        if p.is_dir()
    ])

    print(f"Total classes found: {len(class_dirs)}")

    if len(class_dirs) < NUM_CLASSES:
        raise ValueError(
            f"Only found {len(class_dirs)} classes, but requested {NUM_CLASSES}"
        )

    # Randomly select 100 classes
    selected_classes = random.sample(class_dirs, NUM_CLASSES)

    summary = []

    for class_dir in selected_classes:
        class_name = class_dir.name
        images = get_image_files(class_dir)

        if len(images) < IMAGES_PER_CLASS:
            print(
                f"Skipping {class_name}: only {len(images)} images found"
            )
            continue

        selected_images = random.sample(images, IMAGES_PER_CLASS)

        target_class_dir = out_path / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in selected_images:
            target_path = target_class_dir / img_path.name

            if COPY_IMAGES:
                shutil.copy2(img_path, target_path)
            else:
                if target_path.exists():
                    target_path.unlink()
                os.symlink(img_path.resolve(), target_path)

        summary.append({
            "class": class_name,
            "num_images": len(selected_images)
        })

        print(f"Saved {len(selected_images)} images from class {class_name}")

    print("\nDone.")
    print(f"Selected classes: {len(summary)}")
    print(f"Images per class: {IMAGES_PER_CLASS}")
    print(f"Total images: {len(summary) * IMAGES_PER_CLASS}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()