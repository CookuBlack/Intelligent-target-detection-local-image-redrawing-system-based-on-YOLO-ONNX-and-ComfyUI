import os
import sys
import cv2
import numpy as np
import requests
import time
import io
from PIL import Image
from ultralytics import YOLO
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QFileDialog, QGroupBox,
    QDoubleSpinBox, QSpinBox, QFormLayout, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QPalette, QColor

# ====================== 核心配置 ======================
YOLO_MODEL_PATH = "weights/best.onnx"  # 修改为ONNX模型路径
COMFYUI_MODEL_NAME = "majicMIX_realistic_v7.safetensors"
TEMP_DIR = "./temp_comfyui"
wait_time = 120
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("weights", exist_ok=True)  # 确保weights目录存在

# ComfyUI API
COMFYUI_API = {
    "prompt": "http://127.0.0.1:8188/prompt",
    "history": "http://127.0.0.1:8188/history",
    "view": "http://127.0.0.1:8188/view"
}
# ================================================================


class InpaintWorker(QThread):
    """后台生成线程"""
    log_signal = Signal(str)  # 日志信号
    result_signal = Signal(QPixmap)  # 结果图片信号
    error_signal = Signal(str)  # 错误信号
    finished_signal = Signal()  # 完成信号

    def __init__(self, image_path, prompt, negative_prompt, box_expand_ratio, mask_blur_kernel, steps, cfg, denoise):
        super().__init__()
        self.image_path = image_path
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.box_expand_ratio = box_expand_ratio
        self.mask_blur_kernel = mask_blur_kernel
        self.steps = steps
        self.cfg = cfg
        self.denoise = denoise

    def log(self, msg):
        self.log_signal.emit(msg)

    def load_yolo_model(self, model_path):
        """加载YOLO模型（支持ONNX）"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO模型不存在：{model_path}\n请确保模型已放在 {model_path}")
        try:
            # Ultralytics YOLO直接支持加载ONNX模型
            return YOLO(model_path)
        except Exception as e:
            raise ValueError(f"加载ONNX模型失败：{str(e)}\n请确保已安装 onnxruntime 库")

    def soft_expand_box(self, x1, y1, x2, y2, img_w, img_h, expand_ratio):
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        new_w, new_h = (x2 - x1) * expand_ratio, (y2 - y1) * expand_ratio
        return (
            max(0, int(center_x - new_w / 2)),
            max(0, int(center_y - new_h / 2)),
            min(img_w, int(center_x + new_w / 2)),
            min(img_h, int(center_y + new_h / 2))
        )

    def create_soft_mask(self, img_h, img_w, box, blur_kernel):
        x1, y1, x2, y2 = box
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        mask = cv2.GaussianBlur(mask, blur_kernel, 0)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def save_temp_files(self, img, mask):
        temp_img = os.path.abspath(os.path.join(TEMP_DIR, "temp_input.jpg"))
        temp_mask = os.path.abspath(os.path.join(TEMP_DIR, "temp_mask.jpg"))
        cv2.imencode('.jpg', img)[1].tofile(temp_img)
        cv2.imencode('.jpg', mask)[1].tofile(temp_mask)
        return temp_img, temp_mask

    def build_comfyui_workflow(self, img_path, mask_path):
        return {
            "103": {"inputs": {"image": img_path}, "class_type": "LoadImage"},
            "104": {"inputs": {"image": mask_path}, "class_type": "LoadImage"},
            "105": {"inputs": {"ckpt_name": COMFYUI_MODEL_NAME}, "class_type": "CheckpointLoaderSimple"},
            "106": {"inputs": {"text": self.prompt, "clip": ["105", 1]}, "class_type": "CLIPTextEncode"},
            "107": {"inputs": {"text": self.negative_prompt, "clip": ["105", 1]}, "class_type": "CLIPTextEncode"},
            "108": {"inputs": {"pixels": ["103", 0], "vae": ["105", 2]}, "class_type": "VAEEncode"},
            "110": {"inputs": {"image": ["104", 0], "channel": "red", "invert": False}, "class_type": "ImageToMask"},
            "113": {"inputs": {"samples": ["108", 0], "mask": ["110", 0]}, "class_type": "SetLatentNoiseMask"},
            "109": {
                "inputs": {
                    "seed": int(time.time()), "steps": self.steps, "cfg": self.cfg,
                    "sampler_name": "euler", "scheduler": "normal", "denoise": self.denoise,
                    "model": ["105", 0], "positive": ["106", 0], "negative": ["107", 0],
                    "latent_image": ["113", 0]
                },
                "class_type": "KSampler"
            },
            "111": {"inputs": {"samples": ["109", 0], "vae": ["105", 2]}, "class_type": "VAEDecode"},
            "112": {"inputs": {"filename_prefix": "YOLO_Inpaint", "images": ["111", 0]}, "class_type": "SaveImage"}
        }

    def get_comfyui_result(self, prompt_id):
        for i in range(wait_time):
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
                    self.log(f"生成中... ({i + 1}/{wait_time})")
                time.sleep(1)
            except Exception:
                time.sleep(1)
        return None

    def run(self):
        try:
            # 读取图片
            self.log("正在读取图片...")
            img = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("图片读取失败，请检查路径")
            img_h, img_w = img.shape[:2]
            self.log(f"输入图片：{img_w}×{img_h}")

            # YOLO检测（使用ONNX模型）
            self.log("正在加载YOLO ONNX模型...")
            yolo_model = self.load_yolo_model(YOLO_MODEL_PATH)
            self.log("正在进行目标检测...")
            results = yolo_model(img, verbose=False)
            orig_box = (0, 0, img_w, img_h)
            if len(results[0].boxes) > 0:
                best_box = results[0].boxes[np.argmax(results[0].boxes.conf.cpu().numpy())]
                orig_box = tuple(map(int, best_box.xyxy[0].cpu().numpy()))
            self.log(f"原始检测框：{orig_box}")

            # 软扩大框
            soft_box = self.soft_expand_box(*orig_box, img_w, img_h, self.box_expand_ratio)
            self.log(f"软扩大框：{soft_box}")

            # 生成软掩码
            self.log("正在生成软边缘掩码...")
            mask = self.create_soft_mask(img_h, img_w, soft_box, self.mask_blur_kernel)
            temp_img, temp_mask = self.save_temp_files(img, mask)
            self.log("临时文件已保存")

            # 调用ComfyUI
            self.log("正在提交生成任务...")
            workflow = self.build_comfyui_workflow(temp_img, temp_mask)
            resp = requests.post(COMFYUI_API["prompt"], json={"prompt": workflow, "client_id": "yolo_client"},
                                 timeout=10)
            if resp.status_code != 200:
                raise ValueError(f"ComfyUI错误：{resp.text}")
            prompt_id = resp.json()["prompt_id"]
            self.log(f"任务ID：{prompt_id[:8]}...")

            # 获取结果
            generated_pil = self.get_comfyui_result(prompt_id)
            if not generated_pil:
                raise ValueError("生成超时，请检查ComfyUI")
            self.log("生成完成")

            # 合成结果
            self.log("正在合成结果...")
            x1, y1, x2, y2 = soft_box
            generated_np = cv2.cvtColor(np.array(generated_pil), cv2.COLOR_RGB2BGR)
            target_h, target_w = y2 - y1, x2 - x1
            generated_region = cv2.resize(generated_np[y1:y2, x1:x2], (target_w, target_h))
            final_img = img.copy()
            final_img[y1:y2, x1:x2] = generated_region

            # 转换为QPixmap并发送
            final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            h, w, ch = final_rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(final_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            result_pixmap = QPixmap.fromImage(qt_img)

            # 保存结果
            Image.fromarray(final_rgb).save("final_comfyui_result.jpg")
            self.log("全流程完成！结果已保存为 final_comfyui_result.jpg")

            self.result_signal.emit(result_pixmap)
            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))
            self.finished_signal.emit()


class InpaintUI(QMainWindow):
    """主界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能目标检测与局部重绘系统（ONNX版）")
        self.setGeometry(100, 100, 1000, 700)
        self.current_image_path = None
        self.worker = None
        self.result_pixmap = None

        # 设置暗色主题
        self.setup_dark_theme()

        self.init_ui()

    def setup_dark_theme(self):
        """设置暗色主题"""
        app = QApplication.instance()
        app.setStyle("Fusion")

        # 暗色调色板
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(45, 45, 45))
        palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(60, 60, 60))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        app.setPalette(palette)

        # 额外的样式表
        self.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                font-size: 13px; 
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px;
                color: #4a9eff;
            }
            QPushButton {
                background-color: #4a9eff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                min-height: 25px;
            }
            QPushButton:hover { 
                background-color: #6ab0ff; 
            }
            QPushButton:pressed { 
                background-color: #3a8eef; 
            }
            QPushButton:disabled { 
                background-color: #555; 
                color: #888;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
                background-color: #2d2d2d;
                color: #e0e0e0;
                font-size: 13px;
            }
            QLineEdit:focus, QTextEdit:focus { 
                border: 1px solid #4a9eff; 
            }
            QLabel { 
                color: #e0e0e0; 
                font-size: 13px;
            }
            QTextEdit { 
                background-color: #252525;
                border: 1px solid #444;
            }
            QDoubleSpinBox, QSpinBox {
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border: 1px solid #4a9eff;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

    def init_ui(self):
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 顶部：图片预览区
        preview_group = QGroupBox("图片预览")
        preview_layout = QHBoxLayout()

        # 原图预览
        self.orig_label = QLabel("请选择图片")
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setStyleSheet("border: 2px dashed #555; border-radius: 6px; background-color: #2d2d2d;")
        self.orig_label.setMinimumSize(350, 300)

        # 结果预览
        self.result_label = QLabel("等待生成...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 2px dashed #555; border-radius: 6px; background-color: #2d2d2d;")
        self.result_label.setMinimumSize(350, 300)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.orig_label)
        splitter.addWidget(self.result_label)
        splitter.setSizes([450, 450])
        splitter.setHandleWidth(10)

        preview_layout.addWidget(splitter)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group, 3)

        # 中间：参数配置区
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setMaximumHeight(200)

        param_group = QGroupBox("参数配置")
        param_layout = QFormLayout()
        param_layout.setSpacing(8)

        # 提示词
        self.prompt_edit = QLineEdit(
            "A small child is playing with the camera, realistic, high detail, natural lighting")
        self.prompt_edit.setPlaceholderText("输入正向提示词")
        param_layout.addRow("正向提示词：", self.prompt_edit)

        # 反向提示词
        self.negative_prompt_edit = QLineEdit("blurry, low quality, distorted, text, watermark, cartoon, anime")
        self.negative_prompt_edit.setPlaceholderText("输入反向提示词")
        param_layout.addRow("反向提示词：", self.negative_prompt_edit)

        # 参数行（两列布局）
        param_row_layout = QHBoxLayout()

        # 左侧参数
        left_param_layout = QFormLayout()
        self.expand_spin = QDoubleSpinBox()
        self.expand_spin.setRange(1.0, 2.0)
        self.expand_spin.setSingleStep(0.1)
        self.expand_spin.setValue(1.2)
        left_param_layout.addRow("软扩大比例：", self.expand_spin)

        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(3, 21)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(9)
        left_param_layout.addRow("掩码模糊核：", self.blur_spin)

        # 右侧参数
        right_param_layout = QFormLayout()
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 50)
        self.steps_spin.setValue(25)
        right_param_layout.addRow("采样步数：", self.steps_spin)

        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 20.0)
        self.cfg_spin.setSingleStep(0.5)
        self.cfg_spin.setValue(8.0)
        right_param_layout.addRow("CFG强度：", self.cfg_spin)

        self.denoise_spin = QDoubleSpinBox()
        self.denoise_spin.setRange(0.1, 1.0)
        self.denoise_spin.setSingleStep(0.1)
        self.denoise_spin.setValue(0.7)
        right_param_layout.addRow("重绘强度：", self.denoise_spin)

        param_row_layout.addLayout(left_param_layout, 1)
        param_row_layout.addLayout(right_param_layout, 1)
        param_layout.addRow(param_row_layout)

        param_group.setLayout(param_layout)
        param_scroll.setWidget(param_group)
        main_layout.addWidget(param_scroll, 1)

        # 底部：操作按钮和日志
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        # 按钮区
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)

        self.select_btn = QPushButton("📷 选择图片")
        self.select_btn.clicked.connect(self.select_image)
        self.select_btn.setMinimumHeight(35)

        self.generate_btn = QPushButton("🚀 开始生成")
        self.generate_btn.clicked.connect(self.start_generate)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setMinimumHeight(35)

        self.save_btn = QPushButton("💾 保存结果")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumHeight(35)

        btn_layout.addWidget(self.select_btn)
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()

        # 日志区
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        self.log_edit.setFont(QFont("Consolas", 9))

        bottom_layout.addLayout(btn_layout, 1)
        bottom_layout.addWidget(self.log_edit, 4)

        main_layout.addLayout(bottom_layout, 1)

    def log(self, msg):
        self.log_edit.append(msg)
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.orig_label.setPixmap(scaled_pixmap)
            self.generate_btn.setEnabled(True)
            self.log(f"已选择图片：{file_path}")

    def start_generate(self):
        if not self.current_image_path:
            self.log("请先选择图片")
            return

        self.generate_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        prompt = self.prompt_edit.text()
        negative_prompt = self.negative_prompt_edit.text()
        box_expand_ratio = self.expand_spin.value()
        mask_blur_kernel = (self.blur_spin.value(), self.blur_spin.value())
        steps = self.steps_spin.value()
        cfg = self.cfg_spin.value()
        denoise = self.denoise_spin.value()

        self.worker = InpaintWorker(
            self.current_image_path, prompt, negative_prompt,
            box_expand_ratio, mask_blur_kernel, steps, cfg, denoise
        )
        self.worker.log_signal.connect(self.log)
        self.worker.result_signal.connect(self.show_result)
        self.worker.error_signal.connect(lambda e: self.log(f"错误：{e}"))
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def show_result(self, pixmap):
        self.result_pixmap = pixmap
        scaled_pixmap = pixmap.scaled(
            self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.result_label.setPixmap(scaled_pixmap)
        self.save_btn.setEnabled(True)

    def on_finished(self):
        self.generate_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

    def save_result(self):
        if self.result_pixmap:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存结果", "", "图片文件 (*.jpg *.png)"
            )
            if file_path:
                self.result_pixmap.save(file_path)
                self.log(f"结果已保存到：{file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InpaintUI()
    window.show()
    sys.exit(app.exec())