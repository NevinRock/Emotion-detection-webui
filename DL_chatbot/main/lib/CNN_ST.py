import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class_names = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
num_classes = len(class_names)


# model calss
class MyCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
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


# tranform pic
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# load weight
def load_mycnn(weight_path: str):
    model = MyCNN(num_classes=num_classes).to(device)

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=True)

    model.eval()
    return model

# predict category
def predict_image(model, img_path: str):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    idx = torch.argmax(probs).item()
    pred_label = class_names[idx]

    prob_dict = {
        class_names[i]: float(probs[i] * 100)
        for i in range(num_classes)
    }

    return pred_label, prob_dict


# main output
def classify_emotion(img_path: str, weight_path: str):
    model = load_mycnn(weight_path)
    return predict_image(model, img_path)


# ================================
# main 用于测试
# ================================
if __name__ == "__main__":
    weight_path = "../../ckpt/CNN_ST/ckpt_facial/CNN_ST_facial.pth"
    img_path = "../../media/input_pic.jpg"

    label, probs = classify_emotion(img_path, weight_path)

    print("预测表情：", label)
    print("\n各类别概率：")
    for k, v in probs.items():
        print(f"{k:<10}: {v:.2f}%")
