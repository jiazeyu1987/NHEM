#!/usr/bin/env python3
"""
Simple syntax test for line_detection_widget.py
"""

try:
    import line_detection_widget
    print("✅ line_detection_widget.py imported successfully")
    print("✅ No syntax errors found")
except SyntaxError as e:
    print(f"❌ Syntax Error: {e}")
    print(f"   Line {e.lineno}: {e.text.strip() if e.text else 'N/A'}")
except Exception as e:
    print(f"❌ Import Error: {e}")
    print(f"   Type: {type(e).__name__}")