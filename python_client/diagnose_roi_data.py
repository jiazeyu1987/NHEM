#!/usr/bin/env python3
"""
诊断ROI图像数据问题的调试工具
"""

import os
import sys
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def diagnose_widget_state():
    """诊断widget的ROI数据状态"""
    print("=" * 60)
    print("ROI图像数据诊断工具")
    print("=" * 60)

    try:
        import line_detection_widget

        print("📋 检查LineDetectionWidget类...")
        print(f"   LINE_DETECTION_API_AVAILABLE = {line_detection_widget.LINE_DETECTION_API_AVAILABLE}")

        # 创建widget实例
        print("\n🔧 创建LineDetectionWidget实例...")
        config = {
            'enable_api_integration': True,
            'api_base_url': 'http://localhost:8421',
            'api_password': '31415',
        }

        widget = line_detection_widget.LineDetectionWidget(None, config)

        print("📊 Widget状态检查:")
        print(f"   enable_api_integration = {widget.enable_api_integration}")
        print(f"   api_client = {widget.api_client}")
        print(f"   current_roi1_data = {type(widget.current_roi1_data)}")
        print(f"   image_shape = {widget.image_shape}")
        print(f"   hasattr('_last_roi1_data') = {hasattr(widget, '_last_roi1_data')}")

        if hasattr(widget, '_last_roi1_data'):
            print(f"   _last_roi1_data length = {len(widget._last_roi1_data) if widget._last_roi1_data else 'None'}")
            if widget._last_roi1_data:
                print(f"   _last_roi1_data type = {type(widget._last_roi1_data)}")
                print(f"   _last_roi1_data prefix = {widget._last_roi1_data[:50]}...")

        # 检查回调注册
        print("\n🔄 检查回调注册...")
        print(f"   Callbacks: {widget.callbacks}")

        # 测试手动检测前状态
        print("\n🧪 测试手动检测前的状态...")
        widget._simulate_manual_detection()

        return True, widget

    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_api_connection():
    """测试与后端的API连接"""
    print("\n" + "=" * 60)
    print("测试API连接")
    print("=" * 60)

    try:
        import requests

        # 测试基本连接
        response = requests.get("http://localhost:8421/health", timeout=5)
        if response.status_code == 200:
            print("✅ 后端连接正常")

            # 测试实时数据API
            response = requests.get("http://localhost:8421/data/realtime/enhanced?count=1&include_line_intersection=false", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ 实时数据API连接正常")
                print(f"   数据类型: {data.get('type', 'unknown')}")

                if data.get('type') == 'dual_realtime_data':
                    dual_roi_data = data.get('dual_roi_data', {})
                    roi1_data = dual_roi_data.get('roi1_data', {})

                    print(f"   ROI1数据存在: {bool(roi1_data)}")
                    print(f"   ROI1数据键: {list(roi1_data.keys())}")

                    if roi1_data and 'pixels' in roi1_data:
                        pixels = roi1_data['pixels']
                        print(f"   Pixels数据存在: {bool(pixels)}")
                        print(f"   Pixels数据长度: {len(pixels) if pixels else 'None'}")
                        print(f"   Pixels数据类型: {type(pixels)}")

                        # 检查是否是有效的base64图像数据
                        if isinstance(pixels, str) and pixels.startswith('data:image/'):
                            print("✅ 有效的base64图像数据格式")
                        else:
                            print("❌ 无效的图像数据格式")
                    else:
                        print("❌ ROI1数据中缺少pixels字段")
                else:
                    print(f"❌ 响应类型不是dual_realtime_data: {data.get('type')}")
            else:
                print(f"❌ 实时数据API失败: HTTP {response.status_code}")
        else:
            print(f"❌ 后端连接失败: HTTP {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端服务器 (http://localhost:8421)")
        print("   请确保后端已启动:")
        print("   cd D:\\ProjectPackage\\NHEM\\backend")
        print("   python run.py")
    except Exception as e:
        print(f"❌ API连接测试失败: {e}")

def test_image_processing():
    """测试图像数据处理"""
    print("\n" + "=" * 60)
    print("测试图像数据处理")
    print("=" * 60)

    try:
        import requests
        import base64
        from PIL import Image
        import io

        # 获取实时数据
        response = requests.get("http://localhost:8421/data/realtime/enhanced?count=1&include_line_intersection=false", timeout=5)
        if response.status_code == 200:
            data = response.json()

            if data.get('type') == 'dual_realtime_data':
                dual_roi_data = data.get('dual_roi_data', {})
                roi1_data = dual_roi_data.get('roi1_data', {})

                if roi1_data and 'pixels' in roi1_data:
                    pixels = roi1_data['pixels']

                    print(f"📸 测试图像数据处理...")
                    print(f"   原始数据长度: {len(pixels)}")

                    # 解析base64
                    if pixels.startswith('data:image/'):
                        # 提取base64部分
                        comma_pos = pixels.find(',')
                        if comma_pos != -1:
                            base64_data = pixels[comma_pos + 1:]
                            print(f"   Base64数据长度: {len(base64_data)}")

                            try:
                                # 解码
                                image_bytes = base64.b64decode(base64_data)
                                print(f"   解码后长度: {len(image_bytes)}")

                                # 打开图像
                                image = Image.open(io.BytesIO(image_bytes))
                                print(f"   图像尺寸: {image.size}")
                                print(f"   图像模式: {image.mode}")
                                print(f"   图像格式: {image.format}")

                                # 转换为numpy数组
                                import numpy as np
                                np_array = np.array(image)
                                print(f"   NumPy形状: {np_array.shape}")

                                print("✅ 图像数据处理测试成功")

                            except Exception as e:
                                print(f"❌ 图像处理失败: {e}")
                    else:
                        print("❌ 无效的数据URI格式")
                else:
                    print("❌ ROI1数据中无pixels字段")
            else:
                print("❌ 响应类型不是dual_realtime_data")
        else:
            print(f"❌ 获取实时数据失败: HTTP {response.status_code}")

    except Exception as e:
        print(f"❌ 图像处理测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主诊断函数"""
    print("🔍 ROI图像数据问题诊断工具")
    print("这个工具帮助诊断'无图像数据'错误的具体原因")
    print()

    # 测试1: API连接
    test_api_connection()

    # 测试2: 图像数据处理
    test_image_processing()

    # 测试3: Widget状态
    success, widget = diagnose_widget_state()

    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)

    if success:
        print("✅ LineDetectionWidget创建成功")

        # 检查具体状态
        if hasattr(widget, '_last_roi1_data') and widget._last_roi1_data:
            print("✅ _last_roi1_data已设置")
        else:
            print("❌ _last_roi1_data未设置")

        if widget.image_shape:
            print("✅ image_shape已设置")
        else:
            print("❌ image_shape未设置")

        if widget.current_roi1_data is not None:
            print("✅ current_roi1_data已设置")
        else:
            print("❌ current_roi1_data未设置")

    print("\n💡 建议:")
    print("1. 如果API连接失败，请启动后端服务")
    print("2. 如果图像处理失败，请检查后端ROI配置")
    print("3. 如果Widget状态异常，请检查数据回调注册")

if __name__ == "__main__":
    try:
        main()
        print("\n按Enter键退出...")
        input()
    except KeyboardInterrupt:
        print("\n诊断被用户中断")
    except Exception as e:
        print(f"\n诊断工具异常: {e}")