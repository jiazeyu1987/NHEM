#!/usr/bin/env python3
"""
Test script to verify the ROI split display implementation in Python client
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the python_client directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_client'))

try:
    from http_realtime_client import HTTPRealtimeClientUI
    print("✅ Successfully imported HTTPRealtimeClientUI")
except ImportError as e:
    print(f"❌ Failed to import HTTPRealtimeClientUI: {e}")
    sys.exit(1)

def test_ui_creation():
    """Test that the UI can be created with dual ROI display"""
    try:
        print("🔄 Creating HTTPRealtimeClientUI instance...")

        # Create the UI instance
        app = HTTPRealtimeClientUI()

        # Check if dual ROI widgets were created
        if hasattr(app, 'roi_label_left') and hasattr(app, 'roi_label_right'):
            print("✅ Dual ROI widgets created successfully")
            print(f"   - Left ROI widget: {type(app.roi_label_left)}")
            print(f"   - Right ROI widget: {type(app.roi_label_right)}")
        else:
            print("❌ Dual ROI widgets not found")
            return False

        # Check if original roi_label reference is maintained
        if hasattr(app, 'roi_label'):
            print("✅ Original roi_label reference maintained")
            print(f"   - roi_label refers to: {app.roi_label}")
        else:
            print("❌ Original roi_label reference missing")
            return False

        # Check if helper methods exist
        if hasattr(app, '_update_roi_displays') and hasattr(app, '_update_roi_displays_error'):
            print("✅ Helper methods created successfully")
        else:
            print("❌ Helper methods missing")
            return False

        # Check if image cache attribute exists
        if hasattr(app, '_last_image'):
            print("✅ Image cache attribute created")
        else:
            print("❌ Image cache attribute missing")
            return False

        print("✅ All UI components created successfully")

        # Test error handling method
        try:
            app._update_roi_displays_error("Test error message")
            print("✅ Error handling method works")
        except Exception as e:
            print(f"❌ Error handling method failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ UI creation failed: {e}")
        return False

def test_layout_structure():
    """Test the layout structure of the dual ROI display"""
    try:
        print("\n🔄 Testing layout structure...")

        # Create the UI instance
        app = HTTPRealtimeClientUI()

        # Get the ROI frame
        roi_frame = None
        for child in app.info_frame.winfo_children():
            if isinstance(child, ttk.LabelFrame) and "ROI" in child.cget("text"):
                roi_frame = child
                break

        if roi_frame:
            print("✅ ROI frame found")

            # Check for ROI container
            roi_container = None
            for child in roi_frame.winfo_children():
                if isinstance(child, ttk.Frame):
                    roi_container = child
                    break

            if roi_container:
                print("✅ ROI container frame found")

                # Count child widgets (should be 3: left label, separator, right label)
                children = roi_container.winfo_children()
                print(f"✅ Found {len(children)} child widgets in ROI container")

                if len(children) >= 3:
                    left_label = children[0]
                    separator = children[1]
                    right_label = children[2]

                    print(f"   - Left label text: '{left_label.cget('text')}'")
                    print(f"   - Separator text: '{separator.cget('text')}'")
                    print(f"   - Right label text: '{right_label.cget('text')}'")

                    if separator.cget('text') == '|':
                        print("✅ Separator displays '|' character")
                    else:
                        print("❌ Separator does not display '|' character")
                        return False

                else:
                    print(f"❌ Expected 3 child widgets, found {len(children)}")
                    return False
            else:
                print("❌ ROI container frame not found")
                return False
        else:
            print("❌ ROI frame not found")
            return False

        return True

    except Exception as e:
        print(f"❌ Layout structure test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Python Client ROI Split Display Implementation")
    print("=" * 60)

    # Run tests
    tests = [
        ("UI Creation Test", test_ui_creation),
        ("Layout Structure Test", test_layout_structure)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        print("-" * 40)

        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! ROI split display implementation is working.")

        # Optional: Show the UI for visual verification
        try:
            print("\n👀 Showing UI for visual verification (close window to exit)...")
            app = HTTPRealtimeClientUI()
            app.mainloop()
        except KeyboardInterrupt:
            print("\n👋 Test interrupted by user")
        except Exception as e:
            print(f"⚠️  UI display error: {e}")

        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)