import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# 类别顺序需与训练一致（按字典序）
class_names = ["Angry", "Fear", "Happy", "Sad", "Suprise"]
num_classes = len(class_names)

# --------------------------
# 1. Load model
# --------------------------
def load_model(weight_path):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# --------------------------
# 2. Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# --------------------------
# 3. Predict single image
# --------------------------
def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, 1)[0]

    idx = torch.argmax(probs).item()
    return class_names[idx]


# --------------------------
# 4. Per-class accuracy
# --------------------------
def evaluate_per_class(model, val_root):
    results = {cls: {"correct": 0, "total": 0} for cls in class_names}

    for cls in class_names:
        cls_folder = os.path.join(val_root, cls)
        images = [f for f in os.listdir(cls_folder)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        print(f"\n📂 Evaluating class: {cls} ({len(images)} images)")
        for imgname in tqdm(images):
            img_path = os.path.join(cls_folder, imgname)
            pred = predict_image(model, img_path)

            results[cls]["total"] += 1
            if pred == cls:
                results[cls]["correct"] += 1

    # ---- Print summary ----
    print("\n==================== Per-class Accuracy ====================\n")
    for cls in class_names:
        c = results[cls]["correct"]
        t = results[cls]["total"]
        acc = c / t if t > 0 else 0
        print(f"{cls:<10}: {acc*100:.2f}%    ({c}/{t})")

    print("\n============================================================\n")


# --------------------------
# 5. Main
# --------------------------
if __name__ == "__main__":
    weight_path = r"ckpt_human\resnet50_rc.pth"
    val_root = r"D:..\dataset\Human_Face_Emotions\val"  # ← 修改这里 !!!

    model = load_model(weight_path)
    evaluate_per_class(model, val_root)
