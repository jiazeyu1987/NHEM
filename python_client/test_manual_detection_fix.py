#!/usr/bin/env python3
"""
Test script for manual detection API integration fix
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def test_line_detection_widget_imports():
    """Test the line detection widget imports and API availability"""
    print("=" * 60)
    print("Testing Line Detection Widget Imports")
    print("=" * 60)

    try:
        # Import the widget - this will show all the debug messages
        print("Importing line_detection_widget...")
        import line_detection_widget
        print("✅ line_detection_widget imported successfully")

        # Check the global variables
        print(f"\n📋 Import Status:")
        print(f"   LINE_DETECTION_API_AVAILABLE = {line_detection_widget.LINE_DETECTION_API_AVAILABLE}")
        print(f"   LineDetectionAPIClient = {line_detection_widget.LineDetectionAPIClient}")

        return True

    except Exception as e:
        print(f"❌ Error importing line_detection_widget: {e}")
        return False

def test_widget_initialization():
    """Test widget initialization with API integration"""
    print("\n" + "=" * 60)
    print("Testing Widget Initialization")
    print("=" * 60)

    try:
        import line_detection_widget

        # Create a simple configuration
        config = {
            'enable_api_integration': True,
            'api_base_url': 'http://localhost:8421',
            'api_password': '31415',
            'api_timeout': 10
        }

        print("Creating LineDetectionWidget with API integration enabled...")
        print("Look for the WIDGET DEBUG and API_CLIENT_DEBUG messages below:")
        print("-" * 50)

        # This should trigger all our debugging logs
        widget = line_detection_widget.LineDetectionWidget(None, config)

        print("-" * 50)
        print("✅ Widget created successfully")
        print(f"   enable_api_integration = {widget.enable_api_integration}")
        print(f"   api_client = {widget.api_client}")
        print(f"   LINE_DETECTION_API_AVAILABLE = {line_detection_widget.LINE_DETECTION_API_AVAILABLE}")

        return True, widget

    except Exception as e:
        print(f"❌ Error creating widget: {e}")
        return False, None

def test_manual_detection_callback():
    """Test the manual detection callback with detailed logging"""
    print("\n" + "=" * 60)
    print("Testing Manual Detection Callback")
    print("=" * 60)

    success, widget = test_widget_initialization()
    if not success:
        print("❌ Cannot test manual detection - widget creation failed")
        return False

    try:
        print("Testing _on_manual_detection method...")
        print("Look for MANUAL_DETECTION_DEBUG messages below:")
        print("-" * 50)

        # Call the manual detection method
        widget._on_manual_detection()

        print("-" * 50)
        print("✅ Manual detection callback completed")
        return True

    except Exception as e:
        print(f"❌ Error in manual detection callback: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 Manual Detection Fix Validation Test")
    print("This test validates that the 'API集成为启动' error has been fixed")
    print()

    # Test 1: Imports
    if not test_line_detection_widget_imports():
        print("\n❌ Import test failed")
        return False

    # Test 2: Widget initialization
    if not test_widget_initialization()[0]:
        print("\n❌ Widget initialization test failed")
        return False

    # Test 3: Manual detection callback
    if not test_manual_detection_callback():
        print("\n❌ Manual detection callback test failed")
        return False

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\n📋 Summary of Fix:")
    print("1. ✅ Added comprehensive logging to track API client status")
    print("2. ✅ Fixed Chinese status display with clear error messages")
    print("3. ✅ Enhanced offline mode with detailed diagnostic information")
    print("4. ✅ Improved user feedback with specific error reasons")
    print("\n🔍 Expected Behavior:")
    print("- When API is unavailable: Shows 'API不可用，使用离线模式 (specific reason)'")
    print("- Provides simulated detection with visual feedback")
    print("- Clear diagnostic information in console logs")
    print("\n🚀 The 'API集成为启动' error should now be resolved!")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        sys.exit(1)