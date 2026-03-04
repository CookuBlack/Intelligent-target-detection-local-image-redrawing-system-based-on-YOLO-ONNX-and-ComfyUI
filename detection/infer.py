from ultralytics import YOLO
from PIL import Image
import torch
import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告

from ultralytics.nn import tasks

# 重写torch_safe_load函数，固定weights_only=False
def custom_torch_safe_load(file):
    return torch.load(file, map_location='cpu', weights_only=False), file

# 替换原函数
tasks.torch_safe_load = custom_torch_safe_load

# 推理测试
model = YOLO(r"detection/runs/detect/train/weights/best.pt")

# 测试图片路径
test_image_path = r"detection/girl.jpg"
results = model(test_image_path)

# 可视化检测结果
for r in results:
    im_array = r.plot()  # 绘制检测框、类别、置信度
    im = Image.fromarray(im_array[..., ::-1])  # BGR转RGB适配PIL
    im.save("detection_result.jpg")
    print("检测结果已保存：detection_result.jpg")
    im.show()