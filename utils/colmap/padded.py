from PIL import Image
import os

def padding(root_images_dir,target_size,fill_color):
        

    # 定义要处理的图片文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png','.webp')

    print(f"开始扫描并填充 {root_images_dir} 目录下的所有图片...")
    print(f"目标尺寸: {target_size}, 填充颜色: {fill_color}")
    print("-" * 30)

    processed_count = 0
    skipped_count = 0

    # --- 2. 使用 os.walk 遍历所有子文件夹和文件 ---
    # dirpath 是当前文件夹 (例如: '.../images/Cam127')
    # dirnames 是该文件夹中的子文件夹列表
    # filenames 是该文件夹中的文件列表 (例如: ['460.jpg', '461.jpg'])
    for dirpath, dirnames, filenames in os.walk(root_images_dir):
        for filename in filenames:
            
            # 检查文件是否为我们想要处理的图片类型
            if not filename.lower().endswith(image_extensions):
                continue # 如果不是图片，跳过

            # 构建当前图片的完整路径
            current_image_path = os.path.join(dirpath, filename)
            
            # --- 3. 应用你的原始填充逻辑 ---
            
            # --- 3.1. 打开原始图片 ---
            try:
                original_img = Image.open(current_image_path)
                original_width, original_height = original_img.size
                
                # (可选) 如果图片已经是目标尺寸，则跳过，避免不必要的处理
                if (original_width, original_height) == target_size:
                    # print(f"已跳过 (尺寸已匹配): {current_image_path}")
                    skipped_count += 1
                    continue

            except Exception as e:
                print(f"错误: 打开图片失败 {current_image_path}. 原因: {e}")
                continue # 跳到下一个文件

            # --- 3.2. 创建一个纯黑色的新画布 ---
            # 使用 target_size 创建一个纯黑色的RGB图像
            # 确保原始图像是RGB模式，以防粘贴时出错 (例如，如果是P模式或RGBA)
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
                
            padded_img = Image.new('RGB', target_size, fill_color)

            # --- 3.3. 计算粘贴位置 (居中) ---
            offset_x = (target_size[0] - original_width) // 2
            offset_y = (target_size[1] - original_height) // 2
            offset_x = max(0, offset_x)
            offset_y = max(0, offset_y)

            # --- 3.4. 将原始图片粘贴到新画布的中心 ---
            padded_img.paste(original_img, (offset_x, offset_y))

            # --- 3.5. 保存填充后的图片 (覆盖原始文件) ---
            try:
                # 使用与打开时相同的路径来保存，实现覆盖
                padded_img.save(current_image_path)
                print(f"已处理并覆盖: {current_image_path} (原尺寸: {original_width}x{original_height})")
                processed_count += 1
            except Exception as e:
                print(f"错误: 保存图片失败 {current_image_path}. 原因: {e}")

    print("-" * 30)
    print(f"处理完成。")
    print(f"成功填充并覆盖: {processed_count} 张图片。")
    print(f"跳过 (尺寸已匹配): {skipped_count} 张图片。")

import os
from PIL import Image

def batch_convert_to_webp(root_directory, delete_original=False):
    """
    递归地查找 root_directory 中的所有 .jpg 和 .jpeg 文件，
    将它们转换为 .webp 格式，并选择性地删除原始文件。

    Args:
        root_directory (str): 要搜索的根文件夹路径。
        quality (int): WebP 的压缩质量 (0-100)。默认值为 85。
        delete_original (bool): 如果为 True，将在转换成功后删除原始的 .jpg 文件。
                                默认为 False (更安全)。
    """
    
    if not os.path.isdir(root_directory):
        print(f"错误: 目录未找到: {root_directory}")
        return

    print(f"开始在 {root_directory} 中进行 WebP 转换...")
    print("-" * 30)

    converted_count = 0
    failed_count = 0
    
    # os.walk 会遍历所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            # 检查文件是否为 JPG 或 JPEG (不区分大小写)
            if not filename.lower().endswith(('.jpg', '.jpeg')):
                continue

            jpg_path = os.path.join(dirpath, filename)
            
            # 创建新的 .webp 文件路径 (例如 "image.jpg" -> "image.webp")
            webp_path = os.path.splitext(jpg_path)[0] + ".webp"

            try:
                # 使用 'with' 语句确保文件被正确关闭
                with Image.open(jpg_path) as img:
                    
                    # 确保图像模式为 'RGB' 或 'RGBA'
                    # JPG 通常是 'RGB'，但此步骤可防止模式问题
                    if img.mode not in ['RGB', 'RGBA']:
                        img = img.convert('RGB')
                    
                    # 以指定的质量保存为 WebP 格式
                    img.save(webp_path, 'webp', lossless=True)
                
                print(f"  已转换: {webp_path}")
                converted_count += 1

                # 如果用户明确要求，则删除原始文件
                if delete_original:
                    os.remove(jpg_path)
                    print(f"  已删除: {jpg_path}")

            except Exception as e:
                print(f"  [!] 转换 {jpg_path} 失败: {e}")
                failed_count += 1

    print("-" * 30)
    print("批量转换完成。")
    print(f"成功转换: {converted_count} 个文件")
    print(f"失败: {failed_count} 个文件")

# --- 主程序入口 ---
if __name__ == "__main__":
    
    # 1. 设置你的根目录
    TARGET_DIR = "/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio/images"
    
    batch_convert_to_webp(TARGET_DIR, delete_original=True)

