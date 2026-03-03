from ultralytics import YOLO
from PIL import Image


# ----------------------
# 2. 推理测试
# 先找一张测试图片（比如自己手机拍的，命名为test.jpg）
model = YOLO(r"D:\Study\Controllable-image-generation\detection\runs\detect\train\weights\best.pt")
test_image_path = "test.jpg"  # 替换为你的图片路径
results = model(test_image_path)

# 可视化检测结果
for r in results:
    im_array = r.plot()  # 自动画检测框
    im = Image.fromarray(im_array[..., ::-1])
    im.save("detection_result.jpg")
    print("检测结果已保存：detection_result.jpg")
    im.show()

