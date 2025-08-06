import sys
import os

# 直接导入上一级目录下的ddddocr文件夹中的代码
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ddddocr import DdddOcr

import cv2

# 定义输入输出目录
src_dir = os.path.join(os.getcwd(), "src")
dist_dir = os.path.join(os.getcwd(), "dist")

# 创建输出目录（如果不存在）
os.makedirs(dist_dir, exist_ok=True)

# 支持的图片后缀
img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

det = DdddOcr(det=True)

# 遍历src目录下所有图片
for filename in os.listdir(src_dir):
    if filename.lower().endswith(img_exts):
        src_path = os.path.join(src_dir, filename)
        dist_path = os.path.join(dist_dir, filename)

        with open(src_path, 'rb') as f:
            image = f.read()

        bboxes = det.detection(image)
        print(f"{filename} 检测到的框: {bboxes}")

        im = cv2.imread(src_path)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            im = cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        cv2.imwrite(dist_path, im)
