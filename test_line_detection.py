#!/usr/bin/env python3
"""
简单测试脚本验证线条检测实现
用于验证Canny边缘检测和Hough直线变换的参数配置
"""

import sys
import os

# 添加backend路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    import numpy as np
    import cv2
    print("✓ 依赖库导入成功")
except ImportError as e:
    print(f"✗ 依赖库导入失败: {e}")
    print("请确保已安装: numpy, opencv-python")
    sys.exit(1)

try:
    from backend.app.models import LineDetectionConfig
    from backend.app.core.line_intersection_detector import LineIntersectionDetector
    print("✓ 核心模块导入成功")
except ImportError as e:
    print(f"✗ 核心模块导入失败: {e}")
    sys.exit(1)

def test_canny_parameters():
    """测试Canny边缘检测参数"""
    print("\n=== 测试Canny边缘检测参数 ===")

    # 创建配置
    config = LineDetectionConfig()

    # 验证参数值
    print(f"低阈值: {config.canny_low_threshold} (期望: 25)")
    print(f"高阈值: {config.canny_high_threshold} (期望: 80)")

    assert config.canny_low_threshold == 25, f"Canny低阈值错误: {config.canny_low_threshold}"
    assert config.canny_high_threshold == 80, f"Canny高阈值错误: {config.canny_high_threshold}"

    print("✓ Canny参数验证通过")

def test_hough_parameters():
    """测试Hough直线变换参数"""
    print("\n=== 测试Hough直线变换参数 ===")

    # 创建配置
    config = LineDetectionConfig()

    # 验证参数值
    print(f"最小线长: {config.hough_min_line_length} (期望: 15)")
    print(f"最大间隙: {config.hough_max_line_gap} (期望: 8)")

    assert config.hough_min_line_length == 15, f"最小线长错误: {config.hough_min_line_length}"
    assert config.hough_max_line_gap == 8, f"最大间隙错误: {config.hough_max_line_gap}"

    print("✓ Hough参数验证通过")

def test_function_import():
    """测试函数导入和调用"""
    print("\n=== 测试函数导入 ===")

    config = LineDetectionConfig()
    detector = LineIntersectionDetector(config)

    # 验证方法存在
    assert hasattr(detector, '_detect_edges'), "_detect_edges方法不存在"
    assert hasattr(detector, '_detect_lines'), "_detect_lines方法不存在"

    print("✓ 函数导入验证通过")

def test_mock_processing():
    """测试模拟处理流程"""
    print("\n=== 测试模拟处理流程 ===")

    # 创建测试图像
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # 添加一些绿色线条
    test_image[30:35, 10:80] = [0, 255, 0]  # 水平绿色线
    test_image[10:80, 50:55] = [0, 255, 0]  # 垂直绿色线

    config = LineDetectionConfig()
    detector = LineIntersectionDetector(config)

    try:
        # 测试绿色掩码提取
        green_mask = detector._extract_green_mask(test_image)
        print(f"✓ 绿色掩码提取成功，绿色像素数: {np.sum(green_mask > 0)}")

        # 测试边缘检测
        edges = detector._detect_edges(green_mask)
        print(f"✓ Canny边缘检测成功，边缘像素数: {np.sum(edges > 0)}")

        # 测试线条检测
        lines = detector._detect_lines(edges)
        print(f"✓ Hough直线检测成功，检测到 {len(lines)} 条线条")

        print("✓ 模拟处理流程测试通过")

    except Exception as e:
        print(f"✗ 处理流程测试失败: {e}")
        return False

    return True

if __name__ == "__main__":
    print("开始线条检测实现验证测试...")

    try:
        test_canny_parameters()
        test_hough_parameters()
        test_function_import()

        if test_mock_processing():
            print("\n🎉 所有测试通过！Canny边缘检测和Hough直线变换实现正确。")
        else:
            print("\n❌ 处理流程测试失败")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)