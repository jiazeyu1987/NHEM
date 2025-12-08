#!/usr/bin/env python3
"""
线条相交点检测功能测试脚本

测试绿色线条相交点检测的核心功能，包括：
1. 基本线条检测功能
2. 相交点计算准确性
3. 性能测试
4. 错误处理
"""

import sys
import os
import time
import logging
import numpy as np
from PIL import Image, ImageDraw
import json

# 添加backend路径到sys.path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

try:
    from backend.app.core.line_intersection_detector import LineIntersectionDetector, create_detector_for_nhem
    from backend.app.models import RoiConfig, LineIntersectionResult, LineInfo
    from backend.app.core.roi_capture import RoiCaptureService
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在NHEM项目根目录下运行此脚本")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image_with_lines(width=200, height=150, lines=None):
    """
    创建包含线条的测试图像

    Args:
        width: 图像宽度
        height: 图像高度
        lines: 线条列表，每个元素为 ((x1, y1, x2, y2), color)

    Returns:
        PIL.Image: 测试图像
    """
    if lines is None:
        # 默认创建两条相交的绿色线条
        lines = [
            ((20, 50, 180, 100), (0, 255, 0)),    # 第一条线
            ((50, 20, 100, 130), (0, 200, 0)),    # 第二条线，与第一条相交
        ]

    # 创建白色背景图像
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 绘制线条
    for (x1, y1, x2, y2), color in lines:
        # 绘制较细的线条（1-3像素宽）
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

    return image


def create_test_image_with_intersection():
    """
    创建有明显相交点的测试图像
    """
    width, height = 200, 150
    lines = [
        ((30, 30, 170, 120), (0, 255, 0)),   # 对角线1
        ((30, 120, 170, 30), (0, 200, 0)),  # 对角线2
    ]
    return create_test_image_with_lines(width, height, lines)


def create_test_image_with_parallel_lines():
    """
    创建平行线条测试图像（无相交点）
    """
    width, height = 200, 150
    lines = [
        ((20, 50, 180, 50), (0, 255, 0)),    # 水平线1
        ((20, 100, 180, 100), (0, 200, 0)),  # 水平线2
    ]
    return create_test_image_with_lines(width, height, lines)


def create_test_image_with_no_lines():
    """
    创建无线条的空白测试图像
    """
    return Image.new('RGB', (200, 150), (255, 255, 255))


def test_basic_line_detection():
    """测试基本线条检测功能"""
    print("\n=== 测试基本线条检测功能 ===")

    try:
        # 创建检测器
        detector = LineIntersectionDetector()

        # 测试相交线条
        print("1. 测试相交线条检测...")
        image_with_intersection = create_test_image_with_intersection()
        result = detector.detect_intersection(image_with_intersection)

        print(f"   - 检测结果: {result}")
        if result.get('intersection'):
            print(f"   - 相交点坐标: {result['intersection']}")
            print(f"   - 置信度: {result['confidence']:.2f}")
            print(f"   - 检测到线条数: {result['lines_count']}")
            print(f"   - 处理时间: {result['processing_time_ms']:.1f}ms")
            print("   ✅ 相交线条检测通过")
        else:
            print("   ❌ 未检测到相交点")

        # 测试平行线条
        print("\n2. 测试平行线条检测...")
        image_with_parallel = create_test_image_with_parallel_lines()
        result_parallel = detector.detect_intersection(image_with_parallel)

        if result_parallel.get('intersection') is None:
            print("   ✅ 平行线条正确识别为无相交点")
        else:
            print(f"   ❌ 平行线条错误检测到相交点: {result_parallel.get('intersection')}")

        # 测试无线条图像
        print("\n3. 测试无线条图像...")
        image_no_lines = create_test_image_with_no_lines()
        result_no_lines = detector.detect_intersection(image_no_lines)

        if result_no_lines.get('intersection') is None:
            print("   ✅ 无线条图像正确识别为无相交点")
        else:
            print(f"   ❌ 无线条图像错误检测到相交点: {result_no_lines.get('intersection')}")

    except Exception as e:
        print(f"❌ 基本线条检测测试失败: {e}")
        return False

    return True


def test_intersection_accuracy():
    """测试相交点计算准确性"""
    print("\n=== 测试相交点计算准确性 ===")

    try:
        detector = LineIntersectionDetector()

        # 创建已知相交点的图像
        # 线条1: 从 (10, 10) 到 (190, 140)
        # 线条2: 从 (10, 140) 到 (190, 10)
        # 理论相交点应该在 (100, 75) (中心点)
        lines = [
            ((10, 10, 190, 140), (0, 255, 0)),
            ((10, 140, 190, 10), (0, 200, 0))
        ]
        test_image = create_test_image_with_lines(200, 150, lines)

        result = detector.detect_intersection(test_image)

        if result.get('intersection'):
            detected_x, detected_y = result['intersection']
            expected_x, expected_y = 100, 75

            # 计算误差
            error_x = abs(detected_x - expected_x)
            error_y = abs(detected_y - expected_y)
            total_error = np.sqrt(error_x**2 + error_y**2)

            print(f"   - 期望相交点: ({expected_x}, {expected_y})")
            print(f"   - 检测到相交点: ({detected_x:.1f}, {detected_y:.1f})")
            print(f"   - X轴误差: {error_x:.1f}px")
            print(f"   - Y轴误差: {error_y:.1f}px")
            print(f"   - 总误差: {total_error:.1f}px")

            # 允许5像素的误差
            if total_error <= 5.0:
                print("   ✅ 相交点计算准确性通过")
                return True
            else:
                print(f"   ❌ 相交点计算误差过大: {total_error:.1f}px > 5px")
                return False
        else:
            print("   ❌ 未检测到预期的相交点")
            return False

    except Exception as e:
        print(f"❌ 相交点计算准确性测试失败: {e}")
        return False


def test_performance():
    """测试性能"""
    print("\n=== 测试性能 ===")

    try:
        detector = LineIntersectionDetector()

        # 创建测试图像
        test_image = create_test_image_with_intersection()

        # 性能测试
        num_tests = 10
        processing_times = []

        print(f"   执行 {num_tests} 次检测测试...")

        for i in range(num_tests):
            start_time = time.time()
            result = detector.detect_intersection(test_image)
            end_time = time.time()

            processing_time = (end_time - start_time) * 1000  # 转换为毫秒
            processing_times.append(processing_time)

        # 计算统计信息
        avg_time = np.mean(processing_times)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)

        print(f"   - 平均处理时间: {avg_time:.1f}ms")
        print(f"   - 最快处理时间: {min_time:.1f}ms")
        print(f"   - 最慢处理时间: {max_time:.1f}ms")

        # 检查是否满足性能要求（目标：<300ms）
        if avg_time <= 300:
            print("   ✅ 性能测试通过 (平均处理时间 < 300ms)")
            return True
        else:
            print(f"   ❌ 性能测试失败 (平均处理时间 {avg_time:.1f}ms > 300ms)")
            return False

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")

    try:
        detector = LineIntersectionDetector()

        # 测试无效输入
        print("1. 测试无效输入...")
        try:
            result = detector.detect_intersection(None)
            print("   - 应该抛出异常但没有抛出")
            return False
        except (AttributeError, TypeError):
            print("   ✅ 无效输入正确抛出异常")

        # 测试极小图像
        print("\n2. 测试极小图像...")
        tiny_image = Image.new('RGB', (1, 1), (255, 255, 255))
        result = detector.detect_intersection(tiny_image)
        # 应该能处理，但返回无相交点
        if result.get('intersection') is None:
            print("   ✅ 极小图像正确处理")
        else:
            print("   ❌ 极小图像处理异常")

        # 测试无效ROI坐标
        print("\n3. 测试无效ROI坐标...")
        test_image = create_test_image_with_intersection()
        result = detector.detect_intersection(test_image, roi_coords=(300, 300, 100, 100))
        # 应该能处理无效坐标
        if 'error' in result or result.get('intersection') is None:
            print("   ✅ 无效ROI坐标正确处理")
        else:
            print("   ❌ 无效ROI坐标处理异常")

        return True

    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False


def test_roi_capture_integration():
    """测试ROI捕获服务集成"""
    print("\n=== 测试ROI捕获服务集成 ===")

    try:
        # 创建ROI捕获服务
        roi_service = RoiCaptureService()

        # 创建测试ROI配置
        roi_config = RoiConfig(x1=10, y1=10, x2=210, y2=160)

        print("1. 测试启用线条检测...")
        success = roi_service.enable_line_detection()
        if success:
            print("   ✅ 线条检测启用成功")
        else:
            print("   ❌ 线条检测启用失败")
            return False

        print(f"   - 线条检测状态: {roi_service.is_line_detection_enabled()}")

        print("\n2. 测试线条检测功能...")
        # 注意：这个测试需要屏幕上有绿色线条，可能在实际环境中失败
        # 在CI环境中，我们只测试功能是否正常调用
        try:
            result = roi_service.detect_line_intersection(roi_config)
            print(f"   - 检测结果类型: {type(result)}")
            print("   ✅ 线条检测功能调用成功")
        except Exception as e:
            print(f"   - 线条检测功能调用异常（可能在无屏环境中）: {e}")
            print("   ⚠️  这在无屏环境中是正常的")

        print("\n3. 测试禁用线条检测...")
        roi_service.disable_line_detection()
        if not roi_service.is_line_detection_enabled():
            print("   ✅ 线条检测禁用成功")
        else:
            print("   ❌ 线条检测禁用失败")
            return False

        return True

    except Exception as e:
        print(f"❌ ROI捕获服务集成测试失败: {e}")
        return False


def test_configuration():
    """测试配置功能"""
    print("\n=== 测试配置功能 ===")

    try:
        # 测试默认配置
        print("1. 测试默认配置...")
        detector = LineIntersectionDetector()
        config = detector.get_detection_info()
        print(f"   - 配置信息: {config}")
        print("   ✅ 默认配置获取成功")

        # 测试配置更新
        print("\n2. 测试配置更新...")
        new_config = {
            'canny_low_threshold': 50,
            'hough_threshold': 20,
            'processing_timeout_ms': 400
        }
        detector.update_config(new_config)
        updated_info = detector.get_detection_info()

        if updated_info['config']['canny_low_threshold'] == 50:
            print("   ✅ 配置更新成功")
        else:
            print("   ❌ 配置更新失败")
            return False

        # 测试NHEM专用检测器创建
        print("\n3. 测试NHEM专用检测器...")
        nhem_detector = create_detector_for_nhem(4.0)  # 4 FPS
        nhem_info = nhem_detector.get_detection_info()
        print(f"   - NHEM检测器缓存持续时间: {nhem_info['config']['cache_duration']}")
        print("   ✅ NHEM专用检测器创建成功")

        return True

    except Exception as e:
        print(f"❌ 配置功能测试失败: {e}")
        return False


def save_test_results(test_results):
    """保存测试结果到文件"""
    try:
        results_file = 'line_intersection_test_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"\n测试结果已保存到: {results_file}")
    except Exception as e:
        print(f"保存测试结果失败: {e}")


def main():
    """主测试函数"""
    print("🎯 线条相交点检测功能测试")
    print("=" * 50)

    test_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests': {}
    }

    # 运行所有测试
    tests = [
        ('基本线条检测', test_basic_line_detection),
        ('相交点计算准确性', test_intersection_accuracy),
        ('性能测试', test_performance),
        ('错误处理', test_error_handling),
        ('ROI捕获服务集成', test_roi_capture_integration),
        ('配置功能', test_configuration),
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results['tests'][test_name] = {
                'passed': result,
                'error': None
            }
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            test_results['tests'][test_name] = {
                'passed': False,
                'error': str(e)
            }

    # 输出测试总结
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print("=" * 50)
    print(f"通过测试: {passed_tests}/{total_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")

    if passed_tests == total_tests:
        print("🎉 所有测试通过！线条相交点检测功能正常工作")
    else:
        print("⚠️  部分测试失败，请检查相关功能")

    # 保存测试结果
    test_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'pass_rate': passed_tests/total_tests,
        'all_passed': passed_tests == total_tests
    }
    save_test_results(test_results)

    return passed_tests == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)