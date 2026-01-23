import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


# =============================================================================
# Device configuration
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =============================================================================
# 1. Dataset / transforms
# =============================================================================
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

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

train_dir = "../dataset/Human_Face_Emotions/train"
val_dir = "../dataset/Human_Face_Emotions/val"

print("Loading datasets...")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))
print("Classes:", train_dataset.classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)


# =============================================================================
# 2. Load ResNet50 (pretrained)
# =============================================================================
print("Loading pretrained ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)

for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

model = model.to(device)

print("==== Trainable parameters ====")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name)
print("==============================")


# =============================================================================
# 3. Loss & optimizer
# =============================================================================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)


# =============================================================================
# 4. Training / evaluation functions
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0

    progress = tqdm(loader, desc="Training", leave=False)

    for imgs, labels in progress:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def eval_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0

    progress = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for imgs, labels in progress:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


# =============================================================================
# 5. Run training
# =============================================================================
EPOCHS = 30

print("Starting training...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
    )
    val_loss, val_acc = eval_model(
        model,
        val_loader,
        criterion,
    )

    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")


# =============================================================================
# 6. Save model
# =============================================================================
os.makedirs("ckpt_human", exist_ok=True)
torch.save(model.state_dict(), "ckpt_human/resnet50_best.pth")
print("Model saved: resnet50_best.pth")
