import os
from pathlib import Path
from PIL import Image
import fire
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def convert_one_image(args):
    """
    单个图片转换任务函数
    """
    file_path, quality, remove_original = args
    try:
        # 构造输出路径
        output_path = file_path.with_suffix('.webp')
        
        # 如果目标文件已存在，跳过 (可选逻辑，这里选择覆盖)
        # if output_path.exists(): return
        
        with Image.open(file_path) as img:
            # 保存为 webp
            # lossless=False 表示有损压缩，对于大多数照片素材建议使用默认有损
            # 如果需要无损，可以设置 lossless=True
            img.save(output_path, 'webp', lossless=True)
            
        # 如果转换成功且需要删除源文件
        if remove_original:
            os.remove(file_path)
            
        return True
    except Exception as e:
        return f"Error converting {file_path}: {str(e)}"

def convert_to_webp(
    root_dir: str, 
    quality: int = 80, 
    remove_original: bool = False, 
    num_workers: int = 8
):
    """
    将目录下所有的 jpg/png 图片转换为 webp 格式。

    Args:
        root_dir (str): 根目录路径
        quality (int): webp 质量 (0-100)，默认 80
        remove_original (bool): 是否删除原始 jpg/png 图片，默认 False (不删除)
        num_workers (int): 并行进程数，默认 8
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Directory {root_dir} does not exist.")
        return

    print(f">> Scanning files in {root_dir} ...")
    
    # 查找所有符合后缀的文件 (不区分大小写)
    target_extensions = {'.jpg', '.jpeg', '.png'}
    # 使用 rglob 进行递归查找
    files = [
        p for p in root_path.rglob('*') 
        if p.suffix.lower() in target_extensions and p.is_file()
    ]

    if not files:
        print("No jpg or png files found.")
        return

    print(f">> Found {len(files)} images. Starting conversion with {num_workers} workers...")

    # 准备参数列表
    tasks = [(f, quality, remove_original) for f in files]

    # 并行执行
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 使用 tqdm 显示进度条
        results = list(tqdm(executor.map(convert_one_image, tasks), total=len(tasks), unit="img"))

    # 统计错误
    errors = [r for r in results if r is not True]
    if errors:
        print(f"\nCompleted with {len(errors)} errors:")
        for err in errors:
            print(err)
    else:
        print("\n>> All images converted successfully.")

if __name__ == '__main__':
    fire.Fire(convert_to_webp)