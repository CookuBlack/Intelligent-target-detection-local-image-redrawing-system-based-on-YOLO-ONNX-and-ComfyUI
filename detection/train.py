import torch
from ultralytics import YOLO


if __name__ == '__main__':

    # 加载YOLOv8，直接用coco128微调（无需下载大数据集）
    model = YOLO("yolov8n.pt")  # 加载nano版预训练模型

    # 直接用YOLOv8自带的coco128数据集微调（自动下载，仅500MB）
    results = model.train(
        data="coco128.yaml",  # 内置小数据集，无需额外准备
        epochs=50,
        imgsz=640,
        batch=8,
        device="0" if torch.cuda.is_available() else "cpu",
        workers=0
    )