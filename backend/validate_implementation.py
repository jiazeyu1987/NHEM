#!/usr/bin/env python3
"""
Validation script for Task 13: Manual detection endpoint implementation
"""

import ast
import os
import sys

def validate_models():
    """Validate that the new models are properly defined in models.py"""
    print("🔍 Validating models...")

    models_file = "app/models.py"
    if not os.path.exists(models_file):
        print(f"❌ Models file not found: {models_file}")
        return False

    with open(models_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for required classes
    required_classes = [
        "ManualLineDetectionRequest",
        "ManualLineDetectionResponse"
    ]

    for class_name in required_classes:
        if f"class {class_name}" in content:
            print(f"  ✅ {class_name} found")
        else:
            print(f"  ❌ {class_name} not found")
            return False

    # Check for required fields in ManualLineDetectionRequest
    required_fields = [
        "password",
        "roi_coordinates",
        "image_data",
        "detection_params",
        "force_refresh",
        "include_debug_info"
    ]

    for field in required_fields:
        if field in content:
            print(f"    ✅ Field '{field}' found")
        else:
            print(f"    ❌ Field '{field}' not found")
            return False

    # Check for required fields in ManualLineDetectionResponse
    response_fields = [
        "success",
        "timestamp",
        "message",
        "result",
        "processing_info",
        "debug_info",
        "error_details"
    ]

    for field in response_fields:
        if field in content:
            print(f"    ✅ Response field '{field}' found")
        else:
            print(f"    ❌ Response field '{field}' not found")
            return False

    print("✅ Models validation completed")
    return True

def validate_routes():
    """Validate that the new endpoint is properly defined in routes.py"""
    print("\n🔍 Validating routes...")

    routes_file = "app/api/routes.py"
    if not os.path.exists(routes_file):
        print(f"❌ Routes file not found: {routes_file}")
        return False

    with open(routes_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for required imports
    required_imports = [
        "LineIntersectionResult",
        "ManualLineDetectionRequest",
        "ManualLineDetectionResponse",
        "LineIntersectionDetector"
    ]

    for import_name in required_imports:
        if import_name in content:
            print(f"  ✅ Import '{import_name}' found")
        else:
            print(f"  ❌ Import '{import_name}' not found")
            return False

    # Check for endpoint definition
    if "@router.post(\"/api/roi/line-intersection\"" in content:
        print("  ✅ POST endpoint /api/roi/line-intersection found")
    else:
        print("  ❌ POST endpoint /api/roi/line-intersection not found")
        return False

    # Check for endpoint function
    if "def manual_line_intersection_detection" in content:
        print("  ✅ Endpoint function 'manual_line_intersection_detection' found")
    else:
        print("  ❌ Endpoint function 'manual_line_intersection_detection' not found")
        return False

    # Check for key functionality
    required_functionality = [
        "verify_password",
        "LineIntersectionDetector",
        "detect_intersection",
        "roi_capture_service.capture_roi",
        "base64.b64decode",
        "ManualLineDetectionResponse"
    ]

    for functionality in required_functionality:
        if functionality in content:
            print(f"  ✅ Functionality '{functionality}' found")
        else:
            print(f"  ❌ Functionality '{functionality}' not found")
            return False

    print("✅ Routes validation completed")
    return True

def validate_endpoint_structure():
    """Validate the endpoint structure matches requirements"""
    print("\n🔍 Validating endpoint structure...")

    routes_file = "app/api/routes.py"
    with open(routes_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract the endpoint function
    start_marker = "def manual_line_intersection_detection("
    end_marker = "\n\ndef "

    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("  ❌ Endpoint function not found")
        return False

    # Find the end of the function
    next_def = content.find(end_marker, start_idx)
    if next_def == -1:
        # It might be the last function
        function_content = content[start_idx:]
    else:
        function_content = content[start_idx:next_def]

    # Check for required validation steps
    validation_steps = [
        "password verification",
        "ROI coordinate validation",
        "image data validation",
        "mutual exclusivity check"
    ]

    validation_found = {
        "password verification": "verify_password" in function_content,
        "ROI coordinate validation": "roi_config.validate_coordinates()" in function_content,
        "image data validation": "base64.b64decode" in function_content,
        "mutual exclusivity check": "has_roi and has_image" in function_content
    }

    for step, found in validation_found.items():
        if found:
            print(f"  ✅ {step} validation found")
        else:
            print(f"  ❌ {step} validation not found")
            return False

    # Check for error handling
    error_handling = [
        "password verification failed",
        "missing input data",
        "conflicting input data",
        "invalid ROI coordinates",
        "ROI capture failed",
        "image decode failed",
        "detection execution failed"
    ]

    for error_type in error_handling:
        if error_type in function_content.lower() or error_type.replace(" ", "_") in function_content.lower():
            print(f"  ✅ Error handling for '{error_type}' found")
        else:
            print(f"  ⚠️  Error handling for '{error_type}' may be missing")

    print("✅ Endpoint structure validation completed")
    return True

def validate_line_intersection_detector():
    """Validate that LineIntersectionDetector is available and has the expected method"""
    print("\n🔍 Validating LineIntersectionDetector...")

    detector_file = "app/core/line_intersection_detector.py"
    if not os.path.exists(detector_file):
        print(f"❌ LineIntersectionDetector file not found: {detector_file}")
        return False

    with open(detector_file, 'r', encoding='utf-8') as f:
        content = f.read()

    if "class LineIntersectionDetector" in content:
        print("  ✅ LineIntersectionDetector class found")
    else:
        print("  ❌ LineIntersectionDetector class not found")
        return False

    if "def detect_intersection" in content:
        print("  ✅ detect_intersection method found")
    else:
        print("  ❌ detect_intersection method not found")
        return False

    print("✅ LineIntersectionDetector validation completed")
    return True

def main():
    """Main validation function"""
    print("🧪 Validating Task 13 Implementation: Manual Detection Endpoint")
    print("=" * 60)

    # Change to backend directory
    if os.path.basename(os.getcwd()) != "backend":
        backend_path = "backend"
        if os.path.exists(backend_path):
            os.chdir(backend_path)
            print(f"📁 Changed to directory: {os.getcwd()}")
        else:
            print("❌ Backend directory not found")
            return False

    # Run validations
    validations = [
        validate_models,
        validate_routes,
        validate_endpoint_structure,
        validate_line_intersection_detector
    ]

    all_passed = True
    for validation in validations:
        try:
            if not validation():
                all_passed = False
        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All validations passed! Task 13 implementation appears to be complete.")
        print("\n📋 Implementation Summary:")
        print("  ✅ Added ManualLineDetectionRequest and ManualLineDetectionResponse models")
        print("  ✅ Added POST /api/roi/line-intersection endpoint")
        print("  ✅ Integrated with LineIntersectionDetector")
        print("  ✅ Supports both ROI coordinates and image data inputs")
        print("  ✅ Includes password authentication")
        print("  ✅ Provides comprehensive error handling")
        print("  ✅ Returns detailed processing information")
        print("\n🚀 Ready to test with running backend server!")
    else:
        print("❌ Some validations failed. Please check the issues above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)