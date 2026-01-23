import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


class_names = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
num_classes = len(class_names)


# pic transfrom
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# load model
def load_model(weight_path: str):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model

# return category & probability
def predict_image(model, img_path: str):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]  # [num_classes]

    # Final Prediction Category
    idx = torch.argmax(probs).item()
    pred_label = class_names[idx]

    # Category Probability
    prob_dict = {
        class_names[i]: float(probs[i]) * 100
        for i in range(num_classes)
    }

    return pred_label, prob_dict

# mian output
def classify_emotion(img_path: str, weight_path: str):
    model = load_model(weight_path)
    return predict_image(model, img_path)


# 示例
if __name__ == "__main__":
    weight_path = r"../../ckpt/ResNet50/ckpt_facial/ResNet50_facial.pth"
    img_path = r"../../media/input_pic.jpg"

    label, probs = classify_emotion(img_path, weight_path)

    print("预测表情：", label)
    print("\n各类别百分比：")
    for k, v in probs.items():
        print(f"{k:<10}: {v:.2f}%")
