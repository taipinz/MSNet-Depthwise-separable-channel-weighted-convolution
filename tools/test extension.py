#随机裁剪操作
# import os
# import random
# from PIL import Image
#
# # 设置源文件夹路径
# source_dir = r"C:\Users\Administrator\Desktop\coal_classification_project\data\test"
# # 获取五个子文件夹
# subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]
#
# # 遍历每个子文件夹
# for folder in subfolders:
#     # 获取文件夹中的所有图片
#     images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
#
#     # 遍历每张图片
#     for img_name in images:
#         img_path = os.path.join(folder, img_name)
#         img = Image.open(img_path)
#         width, height = img.size
#
#         # 确保原图大小足够进行裁剪
#         if width >= 224 and height >= 224:
#             # 为每张图片生成10张随机裁剪的图片
#             for i in range(10):
#                 # 随机选择裁剪起点
#                 x = random.randint(0, width - 224)
#                 y = random.randint(0, height - 224)
#
#                 # 裁剪图片
#                 cropped_img = img.crop((x, y, x + 224, y + 224))
#
#                 # 生成新文件名并保存
#                 base_name = os.path.splitext(img_name)[0]
#                 new_name = f"{base_name}_crop_{i + 1}.jpg"
#                 save_path = os.path.join(folder, new_name)
#                 cropped_img.save(save_path)
#
#         img.close()
#
# print("裁剪完成！")



# 重命名操作
import os

# 定义源文件夹路径
source_dir = r'C:\Users\Administrator\Desktop\coal_classification_project\data\test'

# 获取所有子文件夹
categories = os.listdir(source_dir)

# 遍历每个类别文件夹
for category in categories:
    category_path = os.path.join(source_dir, category)

    # 确保是文件夹
    if os.path.isdir(category_path):
        # 获取该类别下的所有图片
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # 重命名每张图片
        for i, image in enumerate(images, 1):
            old_path = os.path.join(category_path, image)
            # 获取文件扩展名
            extension = os.path.splitext(image)[1]
            # 新文件名格式: 类别 (序号)
            new_name = f"{category} ({i}){extension}"
            new_path = os.path.join(category_path, new_name)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f"重命名: {image} -> {new_name}")

print("重命名完成!")
