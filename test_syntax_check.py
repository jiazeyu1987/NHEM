#!/usr/bin/env python3
"""Test script to check the enhanced realtime endpoint syntax"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    print("Testing imports...")

    # Test EnhancedRealtimeDataResponse import
    from app.models import EnhancedRealtimeDataResponse, LineIntersectionResult
    print("✅ EnhancedRealtimeDataResponse import: OK")
    print("✅ LineIntersectionResult import: OK")

    # Test LineIntersectionDetector import
    from app.core.line_intersection_detector import LineIntersectionDetector
    print("✅ LineIntersectionDetector import: OK")

    # Test settings import
    from app.config import settings
    print("✅ Settings import: OK")

    # Test imports from routes
    import app.api.routes
    print("✅ Routes import: OK")

    print("\n🎉 All imports successful! The enhanced endpoint implementation should work correctly.")

except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()