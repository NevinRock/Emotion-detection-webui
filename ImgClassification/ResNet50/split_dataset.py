import os
import random
import shutil


# =============================================================================
# Dataset split configuration
# =============================================================================
source_dir = "./dataset/Facial_Emotion_Recognition_Dataset"

train_dir = os.path.join(source_dir, "train")
val_dir = os.path.join(source_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

train_ratio = 0.8


# =============================================================================
# Split and copy images into train/val folders
# =============================================================================
for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    images = [
        f
        for f in os.listdir(cls_path)
        if os.path.isfile(os.path.join(cls_path, f))
    ]
    if not images:
        continue

    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(train_dir, cls, img),
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(val_dir, cls, img),
        )

print("ok: train/val")
