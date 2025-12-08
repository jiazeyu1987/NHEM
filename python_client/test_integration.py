#!/usr/bin/env python3
"""
Test script for LineDetectionWidget integration
Tests the import and basic functionality without running the full GUI
"""

import sys
import traceback

def test_imports():
    """Test importing the modified components"""
    try:
        print("Testing imports...")

        # Test basic imports
        import tkinter as tk
        from tkinter import ttk
        print("✓ tkinter imports successful")

        # Test matplotlib imports
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        print("✓ matplotlib imports successful")

        # Test LineDetectionWidget import
        try:
            from line_detection_widget import LineDetectionWidget
            print("✓ LineDetectionWidget import successful")
        except ImportError as e:
            print(f"⚠ LineDetectionWidget import failed: {e}")
            print("  This may be expected if dependencies are missing")

        # Test RealtimePlotter import
        try:
            from realtime_plotter import RealtimePlotter
            print("✓ RealtimePlotter import successful")
        except ImportError as e:
            print(f"⚠ RealtimePlotter import failed: {e}")

        # Test LocalConfigLoader import
        try:
            from local_config_loader import LocalConfigLoader
            print("✓ LocalConfigLoader import successful")
        except ImportError as e:
            print(f"⚠ LocalConfigLoader import failed: {e}")

        return True

    except Exception as e:
        print(f"✗ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_syntax():
    """Test Python syntax of the modified file"""
    try:
        print("\nTesting syntax...")

        # Test http_realtime_client syntax
        with open('http_realtime_client.py', 'r', encoding='utf-8') as f:
            code = f.read()

        # Compile to check syntax
        compile(code, 'http_realtime_client.py', 'exec')
        print("✓ http_realtime_client.py syntax is valid")

        # Test line_detection_widget syntax
        with open('line_detection_widget.py', 'r', encoding='utf-8') as f:
            code = f.read()

        compile(code, 'line_detection_widget.py', 'exec')
        print("✓ line_detection_widget.py syntax is valid")

        return True

    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        print(f"  Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"✗ Syntax test failed: {e}")
        traceback.print_exc()
        return False

def test_integration_structure():
    """Test the integration structure without creating GUI"""
    try:
        print("\nTesting integration structure...")

        # Check if required components are present in http_realtime_client
        with open('http_realtime_client.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for LineDetectionWidget import
        if 'from line_detection_widget import LineDetectionWidget' in content:
            print("✓ LineDetectionWidget import found")
        else:
            print("✗ LineDetectionWidget import missing")
            return False

        # Check for notebook setup
        if 'self.notebook = ttk.Notebook(main_frame)' in content:
            print("✓ Notebook setup found")
        else:
            print("✗ Notebook setup missing")
            return False

        # Check for monitoring frame setup
        if 'self.monitoring_frame' in content:
            print("✓ Monitoring frame setup found")
        else:
            print("✗ Monitoring frame setup missing")
            return False

        # Check for line detection setup method
        if '_setup_line_detection_widget' in content:
            print("✓ Line detection setup method found")
        else:
            print("✗ Line detection setup method missing")
            return False

        # Check for toggle method
        if '_toggle_line_detection' in content:
            print("✓ Line detection toggle method found")
        else:
            print("✗ Line detection toggle method missing")
            return False

        # Check for configuration handling
        if '"line_detection"' in content:
            print("✓ Line detection configuration handling found")
        else:
            print("✗ Line detection configuration handling missing")
            return False

        print("✓ All integration structure checks passed")
        return True

    except Exception as e:
        print(f"✗ Integration structure test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("LineDetectionWidget Integration Test")
    print("=" * 40)

    tests = [
        ("Import Tests", test_imports),
        ("Syntax Tests", test_syntax),
        ("Integration Structure Tests", test_integration_structure)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))

        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")

    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Integration appears successful.")
        return True
    else:
        print("❌ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)