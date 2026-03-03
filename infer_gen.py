import os
import cv2
import numpy as np
import requests
import time
import torch
from PIL import Image
from ultralytics import YOLO

# ---------------------- 仅需修改这里的配置 ----------------------
YOLO_MODEL_PATH = r"D:\Study\Controllable-image-generation\detection\runs\detect\train\weights\best.pt"
# ⚠️ 务必确认：模型名和ComfyUI的CheckpointLoaderSimple下拉框里的完全一致
COMFYUI_MODEL_NAME = "majicMIX realistic 麦橘写实_v7.safetensors"
# 临时文件保存目录（默认在代码同目录，无需修改）
TEMP_DIR = "./temp_comfyui"
# -------------------------------------------------------------------

# 本地ComfyUI API配置（固定）
COMFYUI_API_URL = "http://127.0.0.1:8188/prompt"
COMFYUI_HISTORY_URL = "http://127.0.0.1:8188/history"
COMFYUI_VIEW_URL = "http://127.0.0.1:8188/view"
HEADERS = {"Content-Type": "application/json"}

# 创建临时目录
os.makedirs(TEMP_DIR, exist_ok=True)


def main():
    # 1. 加载YOLO检测模型
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO模型不存在！路径：{YOLO_MODEL_PATH}")

    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"⚠️ 直接加载YOLO失败，尝试解除安全限制: {e}")
        original_load = torch.load

        def custom_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = custom_torch_load
        yolo_model = YOLO(YOLO_MODEL_PATH)

    print("✅ YOLO目标检测模型加载成功")

    # -------------------------- 工具函数 --------------------------
    def build_comfyui_workflow(img_path, mask_path, width, height, prompt, negative_prompt):
        """重构ComfyUI工作流：使用本地路径加载图片"""
        workflow = {
            # 1. 加载原图（使用本地绝对路径）
            "103": {
                "inputs": {"image": img_path},
                "class_type": "LoadImage"
            },
            # 2. 加载掩码图（使用本地绝对路径）
            "104": {
                "inputs": {"image": mask_path},
                "class_type": "LoadImage"
            },
            # 3. 加载模型
            "105": {
                "inputs": {"ckpt_name": COMFYUI_MODEL_NAME},
                "class_type": "CheckpointLoaderSimple"
            },
            # 4. 正向提示词
            "106": {
                "inputs": {"text": prompt, "clip": ["105", 1]},
                "class_type": "CLIPTextEncode"
            },
            # 5. 反向提示词
            "107": {
                "inputs": {"text": negative_prompt, "clip": ["105", 1]},
                "class_type": "CLIPTextEncode"
            },
            # 6. VAE编码
            "108": {
                "inputs": {"pixels": ["103", 0], "vae": ["105", 2]},
                "class_type": "VAEEncode"
            },
            # 7. 掩码转换
            "110": {
                "inputs": {"image": ["104", 0], "channel": "red", "invert": False},
                "class_type": "ImageToMask"
            },
            # 8. 设置Latent掩码（参数名确认：mask）
            "113": {
                "inputs": {"samples": ["108", 0], "mask": ["110", 0]},
                "class_type": "SetLatentNoiseMask"
            },
            # 9. KSampler采样
            "109": {
                "inputs": {
                    "seed": int(time.time()),
                    "steps": 20,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.8,
                    "model": ["105", 0],
                    "positive": ["106", 0],
                    "negative": ["107", 0],
                    "latent_image": ["113", 0]
                },
                "class_type": "KSampler"
            },
            # 10. VAE解码
            "111": {
                "inputs": {"samples": ["109", 0], "vae": ["105", 2]},
                "class_type": "VAEDecode"
            },
            # 11. 保存图片
            "112": {
                "inputs": {"filename_prefix": "YOLO_Inpaint", "images": ["111", 0]},
                "class_type": "SaveImage"
            }
        }
        return workflow

    def get_comfyui_result(prompt_id):
        """轮询获取结果（优化超时逻辑）"""
        max_retries = 60  # 延长等待时间到60秒
        for retry_count in range(max_retries):
            try:
                history_resp = requests.get(f"{COMFYUI_HISTORY_URL}/{prompt_id}", timeout=10)
                history_data = history_resp.json()
                if prompt_id in history_data:
                    outputs = history_data[prompt_id]["outputs"]
                    for node_out in outputs.values():
                        if "images" in node_out:
                            img_info = node_out["images"][0]
                            img_resp = requests.get(
                                f"{COMFYUI_VIEW_URL}?filename={img_info['filename']}&subfolder={img_info['subfolder']}&type={img_info['type']}"
                            )
                            return Image.open(Image.io.BytesIO(img_resp.content))
                time.sleep(1)
                print(f"⏳ 等待生成中... ({retry_count + 1}/{max_retries})")
            except Exception as e:
                time.sleep(1)
        return None

    # -------------------------- 核心流程 --------------------------
    def detect_and_inpaint_local(
            image_path="test.jpg",
            prompt="cyberpunk dog with sunglasses, high detail, realistic lighting, 8k",
            negative_prompt="blurry, low quality, distorted, text, watermark"
    ):
        # 1. 读取输入图片
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ 图片不存在！路径：{image_path}")
            return None
        img = cv2.resize(img, (512, 512))  # 固定尺寸避免报错
        img_height, img_width = img.shape[:2]
        print(f"✅ 输入图片加载完成，尺寸：{img_width}×{img_height}")

        # 2. YOLO目标检测
        results = yolo_model(img, verbose=False)
        if len(results[0].boxes) == 0:
            print("⚠️ 未检测到目标，使用整张图片")
            x1, y1, x2, y2 = 0, 0, img_width, img_height
        else:
            best_box_idx = np.argmax(results[0].boxes.conf.cpu().numpy())
            best_box = results[0].boxes[best_box_idx]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
            print(f"✅ 检测到目标，区域：[{x1},{y1},{x2},{y2}]")

        # 3. 生成掩码并保存临时文件
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 保存临时图片（关键：用绝对路径传给ComfyUI）
        temp_img_path = os.path.abspath(os.path.join(TEMP_DIR, "temp_input.jpg"))
        temp_mask_path = os.path.abspath(os.path.join(TEMP_DIR, "temp_mask.jpg"))
        cv2.imencode('.jpg', img)[1].tofile(temp_img_path)
        cv2.imencode('.jpg', mask_3ch)[1].tofile(temp_mask_path)
        print(f"✅ 临时文件已保存：\n  原图：{temp_img_path}\n  掩码：{temp_mask_path}")

        # 4. 构建并调用ComfyUI API
        print("🖼️ 正在调用ComfyUI API...")
        try:
            workflow = build_comfyui_workflow(
                temp_img_path, temp_mask_path, img_width, img_height, prompt, negative_prompt
            )
            payload = {"prompt": workflow, "client_id": "yolo_comfyui_client"}

            resp = requests.post(COMFYUI_API_URL, headers=HEADERS, json=payload, timeout=10)
            print(f"📝 ComfyUI响应状态码：{resp.status_code}")
            if resp.status_code != 200:
                print(f"📝 错误详情：{resp.text}")
                return None
            resp.raise_for_status()

            prompt_id = resp.json()["prompt_id"]
            print(f"✅ 任务已提交 (ID: {prompt_id[:8]}...)")

            # 5. 获取生成结果
            generated_pil = get_comfyui_result(prompt_id)
            if not generated_pil:
                print("❌ 获取结果超时")
                return None
            print("✅ ComfyUI生成完成")

            # 6. 结果合成与保存
            generated_np = cv2.cvtColor(np.array(generated_pil), cv2.COLOR_RGB2BGR)
            final_img = img.copy()
            final_img[y1:y2, x1:x2] = generated_np[y1:y2, x1:x2]

            # 保存结果
            final_pil = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
            final_pil.save("final_comfyui_result.jpg")
            # 保存检测框可视化
            img_with_box = img.copy()
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
            Image.fromarray(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)).save("detection_visualization.jpg")

            print("🎉 全流程完成！")
            print("📄 最终结果：final_comfyui_result.jpg")
            print("📄 检测可视化：detection_visualization.jpg")
            return final_pil

        except requests.exceptions.ConnectionError:
            print("❌ 请先启动ComfyUI！确保http://127.0.0.1:8188可访问")
            return None
        except Exception as e:
            print(f"❌ 流程异常：{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    # 运行入口
    TEST_IMAGE_PATH = "kid.jpg"  # 确保图片在代码同目录
    GENERATE_PROMPT = "A small child is playing with the camera"
    detect_and_inpaint_local(TEST_IMAGE_PATH, GENERATE_PROMPT)


if __name__ == "__main__":
    main()