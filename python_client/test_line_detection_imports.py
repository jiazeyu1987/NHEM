#!/usr/bin/env python3
"""
Test script to verify that all line_detection package components can be imported correctly.
This helps validate that the refactoring was successful and all dependencies are properly structured.
"""

import sys
import traceback
from pathlib import Path

def test_import(module_path: str, description: str) -> bool:
    """Test importing a module and report success/failure."""
    try:
        __import__(module_path)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}")
        print(f"   Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ {description}")
        print(f"   Unexpected Error: {e}")
        return False

def main():
    """Run import tests for all line_detection components."""
    print("Testing Line Detection Package Imports")
    print("=" * 50)

    success_count = 0
    total_tests = 0

    # Test main package import
    total_tests += 1
    if test_import("line_detection", "Main package import"):
        success_count += 1

    # Test configuration modules
    print("\n📁 Configuration Modules:")
    config_tests = [
        ("line_detection.config", "Config package"),
        ("line_detection.config.widget_config", "Widget configuration"),
        ("line_detection.config.visualization_config", "Visualization configuration"),
        ("line_detection.config.api_config", "API configuration"),
    ]

    for module, desc in config_tests:
        total_tests += 1
        if test_import(module, desc):
            success_count += 1

    # Test core components
    print("\n🔧 Core Components:")
    core_tests = [
        ("line_detection.core", "Core package"),
        ("line_detection.core.image_visualizer", "Image visualizer"),
        ("line_detection.core.interaction_handler", "Interaction handler"),
        ("line_detection.core.overlay_manager", "Overlay manager"),
        ("line_detection.core.coordinate_system", "Coordinate system"),
    ]

    for module, desc in core_tests:
        total_tests += 1
        if test_import(module, desc):
            success_count += 1

    # Test business logic
    print("\n💼 Business Logic:")
    business_tests = [
        ("line_detection.business", "Business package"),
        ("line_detection.business.line_detection_manager", "Line detection manager"),
        ("line_detection.business.api_integration", "API integration"),
        ("line_detection.business.data_processor", "Data processor"),
    ]

    for module, desc in business_tests:
        total_tests += 1
        if test_import(module, desc):
            success_count += 1

    # Test UI components
    print("\n🖥️  UI Components:")
    ui_tests = [
        ("line_detection.ui", "UI package"),
        ("line_detection.ui.status_display", "Status display"),
        ("line_detection.ui.controls_manager", "Controls manager"),
    ]

    for module, desc in ui_tests:
        total_tests += 1
        if test_import(module, desc):
            success_count += 1

    # Test utilities
    print("\n🛠️  Utilities:")
    utils_tests = [
        ("line_detection.utils", "Utils package"),
        ("line_detection.utils.error_handling", "Error handling"),
        ("line_detection.utils.geometry_utils", "Geometry utilities"),
        ("line_detection.utils.display_utils", "Display utilities"),
    ]

    for module, desc in utils_tests:
        total_tests += 1
        if test_import(module, desc):
            success_count += 1

    # Test specific imports from main package
    print("\n📦 Main Package Components:")
    main_import_tests = [
        ("from line_detection import ImageVisualizer", "ImageVisualizer import"),
        ("from line_detection import LineDetectionManager", "LineDetectionManager import"),
        ("from line_detection import StatusDisplay", "StatusDisplay import"),
        ("from line_detection import ControlsManager", "ControlsManager import"),
        ("from line_detection import get_widget_config", "get_widget_config import"),
        ("from line_detection import get_visualization_config", "get_visualization_config import"),
    ]

    for import_statement, desc in main_import_tests:
        total_tests += 1
        try:
            exec(import_statement)
            print(f"✅ {desc}")
            success_count += 1
        except Exception as e:
            print(f"❌ {desc}")
            print(f"   Error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print(f"Import Test Summary:")
    print(f"✅ Successful: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    print(f"📊 Success Rate: {(success_count/total_tests)*100:.1f}%")

    if success_count == total_tests:
        print("\n🎉 All imports successful! Refactoring completed successfully.")
        return True
    else:
        print(f"\n⚠️  {total_tests - success_tests} imports failed. Check dependencies and module structure.")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n🧪 Testing Basic Functionality:")

    try:
        # Test configuration
        from line_detection import get_widget_config
        config = get_widget_config()
        print("✅ Configuration loading works")

        # Test geometry utilities
        from line_detection.utils import Point2D, Line2D
        p1 = Point2D(0, 0)
        p2 = Point2D(10, 10)
        line = Line2D(p1, p2)
        distance = p1.distance_to(p2)
        print(f"✅ Geometry calculations work (distance: {distance:.2f})")

        # Test error handling
        from line_detection.utils import ErrorHandlingSystem, ErrorSeverity
        error_handler = ErrorHandlingSystem()
        error_record = error_handler.handle_error(
            Exception("Test error"),
            component="test",
            operation="basic_functionality_test"
        )
        print(f"✅ Error handling works (ID: {error_record.error_id})")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Change to the correct directory
    script_dir = Path(__file__).parent
    python_client_dir = script_dir
    line_detection_dir = python_client_dir / "line_detection"

    if line_detection_dir.exists():
        sys.path.insert(0, str(python_client_dir))
        print(f"Added {python_client_dir} to Python path")
    else:
        print(f"❌ line_detection directory not found at {line_detection_dir}")
        sys.exit(1)

    # Run import tests
    import_success = main()

    # Run basic functionality tests
    if import_success:
        functionality_success = test_basic_functionality()

        if import_success and functionality_success:
            print("\n🎉 ALL TESTS PASSED! The refactoring is complete and working correctly.")
        else:
            print("\n⚠️  Some tests failed. Please review the issues above.")
            sys.exit(1)
    else:
        sys.exit(1)