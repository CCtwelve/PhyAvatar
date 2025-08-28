import sqlite3
import struct


def import_images_to_db(images_path, db_path):
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建images表（如果不存在）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        camera_id INTEGER NOT NULL,
        prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
        prior_tx REAL, prior_ty REAL, prior_tz REAL,
        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
    )
    ''')

    # 读取并解析images.txt
    with open(images_path, 'r') as f:
        lines = f.readlines()

    # 跳过注释行，每两行一组数据
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        # 解析第一行（图像元数据）
        parts = line.split()
        try:
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = ' '.join(parts[9:]).strip()  # 处理可能含空格的文件名

            # 跳过第二行（2D点数据）
            i += 1
            # 插入数据库
            cursor.execute('''
            INSERT OR REPLACE INTO images 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz))

        except (IndexError, ValueError) as e:
            print(f"解析错误 (行 {i + 1}): {line}")
            print(f"错误详情: {str(e)}")
            i += 1

    conn.commit()
    conn.close()
    print("image 导入完成！")


def get_cameras(cameras_txt_path):

    # 读取cameras.txt文件
    with open(cameras_txt_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]
    return lines

def update_cameras_data(db_path, cameras_data):
    """
    更新相机数据到数据库（基于camera_id）
    :param db_path: 数据库文件路径
    :param cameras_data: 相机数据列表，每个元素是字符串，格式为：
        "camera_id MODEL width height fx fy cx cy"
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 准备更新语句
    update_sql = '''
    UPDATE cameras SET
        model = ?,
        width = ?,
        height = ?,
        params = ?,
        prior_focal_length = ?
    WHERE camera_id = ?
    '''

    # 处理每条相机数据
    success_count = 0
    for i, cam_str in enumerate(cameras_data):
        try:
            print(f"处理第 {i + 1} 条数据: {cam_str}")

            # 分割字符串并转换类型
            parts = cam_str.split()
            if len(parts) != 8:
                raise ValueError(f"需要8个参数，但得到 {len(parts)} 个")

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            fx = float(parts[4])
            fy = float(parts[5])
            cx = float(parts[6])
            cy = float(parts[7])
            prior_focal =(fx + fy) / 2
            # 将参数打包为二进制BLOB
            params = struct.pack('dddd', fx, fy, cx, cy)
            # print(struct.unpack('dddd',params))
            # print(struct.unpack('dddd', bytes(b'\x99\x99\x99\x99\x99)\x93@\x99\x99\x99\x99\x99)\x93@\x00\x00\x00\x00\x00Xw@\x00\x00\x00\x00\x00\xf0\x7f@')) )
            # 设置prior_focal_length为fx的整数值
            print(camera_id,model, width, height, params, prior_focal)
            # 执行更新（注意参数顺序与SQL语句中的顺序匹配）
            cursor.execute(update_sql,
                           (1, width, height, params, prior_focal,camera_id))

            # 检查是否实际更新了记录
            if cursor.rowcount > 0:
                success_count += 1
            else:
                print(f"警告: camera_id={camera_id} 不存在，未更新任何记录")

        except Exception as e:
            print(f"处理第 {i + 1} 条数据时出错: {cam_str}")
            print(f"错误详情: {str(e)}")
            continue

    conn.commit()
    conn.close()
    print(f"成功更新 {success_count}/{len(cameras_data)} 条相机数据")

# 使用示例
if __name__ == '__main__':
    num= '17'
    root = '/mnt/cvda/cvda_phava/code/Han/3DGS/colmap/'

    db_path = root+ f"colmap_0_{num}/database.db"
    cameras_path = root + f"data/cam_txt_{num}/cameras.txt"

    # import_images_to_db(
    #     images_path=root + f'data/cam_txt_{num}/images.txt',
    #     db_path=db_path
    # )

    cameras_data = get_cameras(cameras_path)

    update_cameras_data(db_path, cameras_data)

