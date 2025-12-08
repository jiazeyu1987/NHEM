import os

import cv2

from green_detector import detect_green_intersection


def find_green_interaction(picturepath: str):
    """
    外部封装函数：
    - 负责从磁盘读取图像，调用检测模块得到交点坐标；
    - 不再输出 / 保存 mask 图像，而是在控制台打印坐标。
    """
    image = cv2.imread(picturepath)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {picturepath}")

    point = detect_green_intersection(image)

    if point is None:
        print(f"{picturepath}: 未检测到有效交点")
    else:
        x, y = point
        print(f"{picturepath}: 交点坐标 (x={x}, y={y})")

    return point


def test_find_green_interaction() -> None:
    """
    测试函数：
    读取 dataset 目录下的 1.png ~ 5.png，
    依次调用 find_green_interaction。
    """
    base_dir = "dataset"
    for i in range(1, 6):
        img_path = os.path.join(base_dir, f"{i}.png")
        find_green_interaction(img_path)


if __name__ == "__main__":
    test_find_green_interaction()
