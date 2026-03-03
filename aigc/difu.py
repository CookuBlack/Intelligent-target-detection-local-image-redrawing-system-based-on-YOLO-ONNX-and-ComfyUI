import os

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/Study/Controllable-image-generation/hf_cache'  # 自定义缓存路径，避免重复下载


# ----------------------
# 1. 加载预训练模型（无需自己训练）
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载ControlNet（Canny边缘检测版）
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
).to(device)

# 加载Stable Diffusion 1.5
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

# 加载刚才训好的YOLOv8
yolo_model = YOLO(r"D:\Study\Controllable-image-generation\detection\runs\detect\train\weights\best.pt.pt")


# ----------------------
# 2. 完整流程：检测→生成→贴回
# ----------------------
def detect_and_generate(image_path, prompt):
    # Step 1: YOLOv8检测物体
    image = cv2.imread(image_path)
    results = yolo_model(image)

    if len(results[0].boxes) == 0:
        print("未检测到物体，用整张图做生成")
        h, w = image.shape[:2]
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        bbox = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)

    # Step 2: 生成Canny边缘图
    roi = image[y1:y2, x1:x2]
    canny = cv2.Canny(roi, 100, 200)
    canny = np.stack([canny] * 3, axis=-1)
    canny_image = Image.fromarray(canny)

    # Step 3: ControlNet可控生成
    output = pipe(
        prompt=prompt,
        image=canny_image,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]

    # Step 4: 贴回原图
    output = output.resize((x2 - x1, y2 - y1))
    image[y1:y2, x1:x2] = np.array(output)
    final_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return final_image


# ----------------------
# 3. 测试（用你自己的图片）
# ----------------------
test_image = "kid.jpg"  # 替换为你的图片
prompt = "a cute robot, cyberpunk style, neon lights"  # 你的生成指令

result = detect_and_generate(test_image, prompt)
result.save("final_result.jpg")
print("最终结果已保存：final_result.jpg")
result.show()