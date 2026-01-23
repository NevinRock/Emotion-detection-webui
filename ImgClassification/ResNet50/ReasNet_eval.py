import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchvision import models, transforms
from tqdm import tqdm


# =============================================================================
# Device configuration
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Class order must be consistent with training (and folder names)
class_names = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
num_classes = len(class_names)


# =============================================================================
# 1. Load ResNet50 model
# =============================================================================
def load_model(weight_path):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# =============================================================================
# 2. Image preprocessing
# =============================================================================
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# =============================================================================
# 3. Predict a single image
# =============================================================================
def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    idx = torch.argmax(probs).item()
    return class_names[idx], idx


# =============================================================================
# 4. Evaluate per-class + overall accuracy + confusion matrix
# =============================================================================
def evaluate_per_class_with_cm(model, val_root, save_path="confusion_matrix.png"):
    results = {cls: {"correct": 0, "total": 0} for cls in class_names}

    total_correct = 0
    total_images = 0

    y_true = []
    y_pred = []

    for true_idx, cls in enumerate(class_names):
        cls_folder = os.path.join(val_root, cls)
        images = [
            f
            for f in os.listdir(cls_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        print(f"\nEvaluating class: {cls} ({len(images)} images)")
        for imgname in tqdm(images, leave=False):
            img_path = os.path.join(cls_folder, imgname)
            pred_name, pred_idx = predict_image(model, img_path)

            results[cls]["total"] += 1
            total_images += 1

            y_true.append(true_idx)
            y_pred.append(pred_idx)

            if pred_name == cls:
                results[cls]["correct"] += 1
                total_correct += 1

    print("\n==================== Evaluation Results ====================\n")

    overall_acc = total_correct / total_images if total_images > 0 else 0.0
    print(
        f"Overall Accuracy: {overall_acc * 100:.2f}%   "
        f"({total_correct}/{total_images})\n"
    )

    print("Per-class Accuracy:")
    for cls in class_names:
        c = results[cls]["correct"]
        t = results[cls]["total"]
        acc = c / t if t > 0 else 0.0
        print(f"{cls:<10}: {acc * 100:.2f}%    ({c}/{t})")

    print("\n============================================================\n")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 7))
    disp.plot(values_format="d")
    plt.title("Confusion Matrix (Counts)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm, display_labels=class_names
    )

    plt.figure(figsize=(8, 7))
    disp_norm.plot(values_format=".2f")
    plt.title("Confusion Matrix (Row-normalized)")
    plt.tight_layout()
    norm_path = os.path.splitext(save_path)[0] + "_norm.png"
    plt.savefig(norm_path, dpi=300)
    plt.show()

    print(f"Confusion matrix saved to: {save_path}")
    print(f"Normalized confusion matrix saved to: {norm_path}")


# =============================================================================
# 5. Main entry point
# =============================================================================
if __name__ == "__main__":
    weight_path = r".\ckpt_facial\ResNet50_facial.pth"
    val_root = r"..\dataset\Facial_Emotion_Recognition_Dataset/val"

    model = load_model(weight_path)
    evaluate_per_class_with_cm(
        model,
        val_root,
        save_path="confusion_matrix_facial.png",
    )
