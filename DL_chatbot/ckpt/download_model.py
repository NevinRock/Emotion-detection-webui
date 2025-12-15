"""
This file is for pretrain ResNet50 weight download, the model is pack in pytorch
"""

import urllib.request

url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
save_path = "./resnet50.pth"

print("Downloading...")
urllib.request.urlretrieve(url, save_path)
print("Saved to:", save_path)
