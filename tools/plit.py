import os
import shutil
import random

# 源文件夹和目标文件夹路径
src_dir = r"C:\Users\Administrator\Desktop\coal_classification_project\data\train"
dst_dir = r"C:\Users\Administrator\Desktop\coal_classification_project\data\test"

# 确保目标文件夹存在
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# 遍历源文件夹下的所有子文件夹
for class_folder in os.listdir(src_dir):
    src_class_path = os.path.join(src_dir, class_folder)
    dst_class_path = os.path.join(dst_dir, class_folder)

    # 如果是文件夹才处理
    if os.path.isdir(src_class_path):
        # 创建目标文件夹中对应的类别文件夹
        if not os.path.exists(dst_class_path):
            os.makedirs(dst_class_path)

        # 获取该类别下所有图片文件
        images = [f for f in os.listdir(src_class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 计算需要移动的图片数量（20%）
        num_to_move = int(len(images) * 0.2)

        # 随机选择图片
        selected_images = random.sample(images, num_to_move)

        # 移动选中的图片到目标文件夹
        for image in selected_images:
            src_path = os.path.join(src_class_path, image)
            dst_path = os.path.join(dst_class_path, image)
            shutil.copy2(src_path, dst_path)  # 使用copy2保留文件的元数据

print("完成！已将20%的图片复制到测试集文件夹。")
