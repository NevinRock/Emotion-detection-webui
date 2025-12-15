from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# =============================================================================
# Device configuration
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =============================================================================
# MyCNN model definition (must match training)
# =============================================================================
class MyCNN(nn.Module):
    """A small CNN for image classification."""

    def __init__(self, num_classes: int):
        super(MyCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Validation dataset & transform
# =============================================================================
val_transform = transforms.Compose(
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
# Load trained MyCNN
# =============================================================================
def load_model(weight_path):
    """Load a trained MyCNN from a checkpoint path."""
    model = MyCNN(num_classes=num_classes)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# =============================================================================
# Evaluate overall & per-class accuracy
# =============================================================================
def evaluate(model, loader, num_classes):
    """Evaluate model accuracy and collect predictions for confusion matrices."""
    total_correct = 0
    total_samples = 0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # For confusion matrix
    y_true = []
    y_pred = []

    progress = tqdm(loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for imgs, labels in progress:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Collect labels for confusion matrix
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i].item() == label:
                    class_correct[label] += 1

    overall_acc = total_correct / total_samples

    per_class_acc = {
        class_names[i]: (class_correct[i] / class_total[i]) if class_total[i] > 0 else 0.0
        for i in range(num_classes)
    }

    return overall_acc, per_class_acc, total_correct, total_samples, y_true, y_pred


def plot_confusion_matrices(y_true, y_pred, class_names, prefix="confusion_matrix"):
    """Plot and save confusion matrices (counts + row-normalized)."""
    # ---- Counts ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 7))
    disp.plot(values_format="d")
    plt.title("Confusion Matrix (Counts)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(prefix + ".png", dpi=300)
    plt.show()

    # ---- Row-normalized ----
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)

    plt.figure(figsize=(8, 7))
    disp_norm.plot(values_format=".2f")
    plt.title("Confusion Matrix (Row-normalized)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(prefix + "_norm.png", dpi=300)
    plt.show()

    print(f"✅ Saved: {prefix}.png")
    print(f"✅ Saved: {prefix}_norm.png")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    weight_path = "./ckpt_facial/mycnn_facial.pth"
    val_dir = "../dataset/Facial_Emotion_Recognition_Dataset/val"

    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    class_names = val_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Validation samples:", len(val_dataset))

    model = load_model(weight_path)
    overall_acc, per_class_acc, c, t, y_true, y_pred = evaluate(
        model, val_loader, num_classes
    )

    print("\n==================== Evaluation Results ====================")
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%   ({c}/{t})\n")

    print("Per-class Accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"{cls:<10}: {acc * 100:.2f}%")

    print("============================================================\n")

    plot_confusion_matrices(
        y_true,
        y_pred,
        class_names,
        prefix="mycnn_confusion_matrix_facial",
    )