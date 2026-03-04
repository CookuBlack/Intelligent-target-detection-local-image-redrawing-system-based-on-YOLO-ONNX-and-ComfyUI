import os
import cv2
import numpy as np
import requests
import time
import torch
import io
from PIL import Image
from ultralytics import YOLO

# ====================== 可配置参数 ======================
YOLO_MODEL_PATH = r"detection\runs\detect\train\weights\best.pt"
COMFYUI_MODEL_NAME = "majicMIX_realistic_v7.safetensors"    # ComfyUI模型文件名
TEMP_DIR = "./temp_comfyui"

# 软扩大与掩码
BOX_EXPAND_RATIO = 1.2
MASK_BLUR_KERNEL = (9, 9)

# 可视化（绿=原始框，红=软扩大框）
VIS_CONFIG = {
    "orig_color": (0, 255, 0),
    "soft_color": (0, 0, 255),
    "box_thick": 2,
    "text_font": cv2.FONT_HERSHEY_SIMPLEX,
    "text_size": 0.5,
    "text_thick": 1
}

# ComfyUI API
COMFYUI_API = {
    "prompt": "http://127.0.0.1:8188/prompt",
    "history": "http://127.0.0.1:8188/history",
    "view": "http://127.0.0.1:8188/view"
}
# ========================================================

os.makedirs(TEMP_DIR, exist_ok=True)


def load_yolo_model(model_path):
    """加载YOLO模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO模型不存在：{model_path}")

    try:
        return YOLO(model_path)
    except Exception:
        original_load = torch.load
        torch.load = lambda *a, **k: original_load(*a, **{**k, "weights_only": False})
        return YOLO(model_path)


def soft_expand_box(x1, y1, x2, y2, img_w, img_h, expand_ratio=1.2):
    """软扩大检测框"""
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    new_w, new_h = (x2 - x1) * expand_ratio, (y2 - y1) * expand_ratio
    return (
        max(0, int(center_x - new_w / 2)),
        max(0, int(center_y - new_h / 2)),
        min(img_w, int(center_x + new_w / 2)),
        min(img_h, int(center_y + new_h / 2))
    )


def create_soft_mask(img_h, img_w, box, blur_kernel):
    """生成软边缘掩码（高斯模糊）"""
    x1, y1, x2, y2 = box
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def save_temp_files(img, mask, temp_dir):
    """保存临时原图和掩码"""
    temp_img = os.path.abspath(os.path.join(temp_dir, "temp_input.jpg"))
    temp_mask = os.path.abspath(os.path.join(temp_dir, "temp_mask.jpg"))
    cv2.imencode('.jpg', img)[1].tofile(temp_img)
    cv2.imencode('.jpg', mask)[1].tofile(temp_mask)
    return temp_img, temp_mask


def build_comfyui_workflow(img_path, mask_path, prompt, negative_prompt):
    """构建ComfyUI工作流"""
    return {
        "103": {"inputs": {"image": img_path}, "class_type": "LoadImage"},
        "104": {"inputs": {"image": mask_path}, "class_type": "LoadImage"},
        "105": {"inputs": {"ckpt_name": COMFYUI_MODEL_NAME}, "class_type": "CheckpointLoaderSimple"},
        "106": {"inputs": {"text": prompt, "clip": ["105", 1]}, "class_type": "CLIPTextEncode"},
        "107": {"inputs": {"text": negative_prompt, "clip": ["105", 1]}, "class_type": "CLIPTextEncode"},
        "108": {"inputs": {"pixels": ["103", 0], "vae": ["105", 2]}, "class_type": "VAEEncode"},
        "110": {"inputs": {"image": ["104", 0], "channel": "red", "invert": False}, "class_type": "ImageToMask"},
        "113": {"inputs": {"samples": ["108", 0], "mask": ["110", 0]}, "class_type": "SetLatentNoiseMask"},
        "109": {
            "inputs": {
                "seed": int(time.time()), "steps": 25, "cfg": 8.0,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 0.7,
                "model": ["105", 0], "positive": ["106", 0], "negative": ["107", 0],
                "latent_image": ["113", 0]
            },
            "class_type": "KSampler"
        },
        "111": {"inputs": {"samples": ["109", 0], "vae": ["105", 2]}, "class_type": "VAEDecode"},
        "112": {"inputs": {"filename_prefix": "YOLO_Inpaint", "images": ["111", 0]}, "class_type": "SaveImage"}
    }


def get_comfyui_result(prompt_id):
    """轮询获取ComfyUI生成结果（最多90秒）"""
    for i in range(90):
        try:
            resp = requests.get(f"{COMFYUI_API['history']}/{prompt_id}", timeout=10)
            data = resp.json()
            if prompt_id in data:
                outputs = data[prompt_id]["outputs"]
                for node_out in outputs.values():
                    if "images" in node_out:
                        img_info = node_out["images"][0]
                        img_resp = requests.get(
                            f"{COMFYUI_API['view']}?filename={img_info['filename']}&subfolder={img_info['subfolder']}&type={img_info['type']}"
                        )
                        return Image.open(io.BytesIO(img_resp.content))
            if (i + 1) % 10 == 0:
                print(f"生成中... ({i + 1}/90)")
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return None


def blend_and_save_result(original_img, generated_pil, soft_box):
    """合成生成图与原图（修复尺寸不匹配）并保存"""
    x1, y1, x2, y2 = soft_box
    generated_np = cv2.cvtColor(np.array(generated_pil), cv2.COLOR_RGB2BGR)

    # 缩放生成区域至原图尺寸
    target_h, target_w = y2 - y1, x2 - x1
    generated_region = cv2.resize(generated_np[y1:y2, x1:x2], (target_w, target_h))

    # 合成
    final_img = original_img.copy()
    final_img[y1:y2, x1:x2] = generated_region

    # 保存
    Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)).save("final_comfyui_result.jpg")
    return final_img


def draw_dual_boxes(img, orig_box, soft_box, vis_config):
    """在同一张图上绘制原始框+软扩大框（带文字标注）"""
    vis_img = img.copy()
    x1_o, y1_o, x2_o, y2_o = orig_box
    x1_s, y1_s, x2_s, y2_s = soft_box

    # 绘制原始框（绿色）
    cv2.rectangle(vis_img, (x1_o, y1_o), (x2_o, y2_o), vis_config["orig_color"], vis_config["box_thick"])
    cv2.putText(vis_img, "Original Box", (x1_o, y1_o - 10),
                vis_config["text_font"], vis_config["text_size"], vis_config["orig_color"], vis_config["text_thick"])

    # 绘制软扩大框（红色）
    cv2.rectangle(vis_img, (x1_s, y1_s), (x2_s, y2_s), vis_config["soft_color"], vis_config["box_thick"])
    cv2.putText(vis_img, "Soft Expand Box", (x1_s, y1_s - 25),
                vis_config["text_font"], vis_config["text_size"], vis_config["soft_color"], vis_config["text_thick"])

    # 保存
    Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)).save("detection_visualization.jpg")


def detect_and_inpaint_local(image_path, prompt, negative_prompt):
    """核心流程：检测→软扩大→生成→合成→保存"""
    # 1. 读取图片
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"图片不存在：{image_path}")
        return None
    img_h, img_w = img.shape[:2]
    print(f"输入图片：{img_w}×{img_h}")

    # 2. YOLO检测
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    results = yolo_model(img, verbose=False)
    orig_box = (0, 0, img_w, img_h)
    if len(results[0].boxes) > 0:
        best_box = results[0].boxes[np.argmax(results[0].boxes.conf.cpu().numpy())]
        orig_box = tuple(map(int, best_box.xyxy[0].cpu().numpy()))
    print(f"原始框：{orig_box}")

    # 3. 软扩大框
    soft_box = soft_expand_box(*orig_box, img_w, img_h, BOX_EXPAND_RATIO)
    print(f"软扩大框：{soft_box}")

    # 4. 生成软掩码并保存临时文件
    mask = create_soft_mask(img_h, img_w, soft_box, MASK_BLUR_KERNEL)
    temp_img, temp_mask = save_temp_files(img, mask, TEMP_DIR)
    print(f"临时文件已保存")

    # 5. 调用ComfyUI生成
    print("提交生成任务...")
    try:
        workflow = build_comfyui_workflow(temp_img, temp_mask, prompt, negative_prompt)
        resp = requests.post(COMFYUI_API["prompt"], json={"prompt": workflow, "client_id": "yolo_client"}, timeout=10)
        if resp.status_code != 200:
            print(f"ComfyUI错误：{resp.text}")
            return None
        prompt_id = resp.json()["prompt_id"]
        print(f"任务ID：{prompt_id[:8]}...")

        # 6. 获取结果
        generated_pil = get_comfyui_result(prompt_id)
        if not generated_pil:
            print("生成超时")
            return None
        print("生成完成")

        # 7. 合成并保存
        blend_and_save_result(img, generated_pil, soft_box)
        draw_dual_boxes(img, orig_box, soft_box, VIS_CONFIG)

        print("全流程完成！")
        print("最终结果：final_comfyui_result.jpg")
        print("检测可视化：detection_visualization.jpg（绿=原始框，红=软扩大框）")
        return generated_pil

    except requests.exceptions.ConnectionError:
        print("请先启动ComfyUI！")
        return None
    except Exception as e:
        print(f"流程异常：{str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    detect_and_inpaint_local(
        image_path=r"kid.jpg",
        prompt="A small child is playing with the camera, realistic, high detail, natural lighting",
        negative_prompt="blurry, low quality, distorted, text, watermark, cartoon, anime"
    )