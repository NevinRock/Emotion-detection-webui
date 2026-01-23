import sys
from pathlib import Path
import torch
import numpy as np

# main/lib/detect_face.py
import sys
from pathlib import Path
import torch
import numpy as np

# ==================================================
# 1. 用 .. 计算路径（不搞复杂）
# ==================================================
FILE = Path(__file__).resolve()
# FILE = .../Use_model/main/lib/detect_face.py

# YOLOv5-face 代码目录
YOLO_ROOT = FILE.parent / "yolov5_face"
# .../Use_model/main/lib/yolov5_face

sys.path.insert(0, str(YOLO_ROOT))

# YOLO 权重（用 .. 回到 Use_model）
YOLO_WEIGHT = FILE.parent.parent.parent / "ckpt" / "YOLO_face" / "yolov5s-face.pt"
YOLO_WEIGHT = str(YOLO_WEIGHT)

# --------------------------------------------------
# 2️⃣ 使用 YOLO 官方实现（不要复制代码）
# --------------------------------------------------
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_face, scale_coords
from utils.torch_utils import select_device



class FaceDetectorYOLO:
    def __init__(
        self,
        weights=YOLO_WEIGHT,
        device=None,
        img_size=640,
        conf_thres=0.5,
        iou_thres=0.5
    ):
        self.device = device or select_device('')
        self.model = attempt_load(weights, map_location=self.device)
        self.model.eval()

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    @torch.no_grad()
    def detect(self, pil_img):
        """
        Args:
            pil_img: PIL.Image
        Returns:
            list of (x1, y1, x2, y2)
        """
        img0 = np.array(pil_img)
        img = letterbox(img0, self.img_size)[0]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        img = img.unsqueeze(0)

        pred = self.model(img)[0]
        pred = non_max_suppression_face(
            pred, self.conf_thres, self.iou_thres
        )[0]

        boxes = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(
                img.shape[2:], pred[:, :4], img0.shape
            ).round()

            for det in pred:
                x1, y1, x2, y2 = map(int, det[:4])
                boxes.append((x1, y1, x2, y2))

        return boxes
