from ultralytics import YOLO


model = YOLO(r"detection/runs/detect/train/weights/best.pt")
# ----------------------
# 3. 导出ONNX（端侧部署）
# ----------------------
model.export(format="onnx", imgsz=640, simplify=True)
print("ONNX模型已导出：yolov8n.onnx")