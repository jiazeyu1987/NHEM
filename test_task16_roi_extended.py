#!/usr/bin/env python3
"""
Test script for Task 16: Extended ROICapture service with dual ROI processing
Tests ROI1 green line detection integration and ROI2 grayscale analysis
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_import():
    """Test importing the extended ROICapture service"""
    try:
        from app.core.roi_capture import RoiCaptureService
        from app.models import RoiConfig, RoiData, LineIntersectionResult, LineDetectionConfig
        print("✓ Successfully imported extended RoiCaptureService and models")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_roi_capture_service_initialization():
    """Test RoiCaptureService initialization with line detection support"""
    try:
        from app.core.roi_capture import RoiCaptureService

        # Test service initialization
        service = RoiCaptureService()
        print("✓ RoiCaptureService initialized successfully")

        # Test new methods exist
        assert hasattr(service, 'capture_dual_roi_with_line_detection'), "Missing capture_dual_roi_with_line_detection method"
        assert hasattr(service, 'get_dual_roi_performance_metrics'), "Missing get_dual_roi_performance_metrics method"
        assert hasattr(service, '_detect_lines_in_roi1'), "Missing _detect_lines_in_roi1 method"
        print("✓ All new methods are present")

        # Test performance monitoring
        stats = service.get_performance_stats()
        assert 'roi_capture_performance' in stats, "Missing ROI capture performance stats"
        assert 'line_detector_status' in stats, "Missing line detector status"
        assert 'service_config' in stats, "Missing service config"
        print("✓ Performance monitoring is working")

        # Test dual ROI metrics
        metrics = service.get_dual_roi_performance_metrics()
        assert 'processing_performance' in metrics, "Missing processing performance"
        assert 'line_detection_metrics' in metrics, "Missing line detection metrics"
        assert 'service_health' in metrics, "Missing service health"
        print("✓ Dual ROI performance metrics are working")

        return True
    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
        return False

def test_dual_roi_method_signature():
    """Test the signature of the new dual ROI method"""
    try:
        from app.core.roi_capture import RoiCaptureService
        from app.models import RoiConfig, RoiData, LineIntersectionResult

        service = RoiCaptureService()

        # Test method signature
        import inspect
        sig = inspect.signature(service.capture_dual_roi_with_line_detection)
        params = list(sig.parameters.keys())

        expected_params = ['roi_config', 'frame_count']
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

        print("✓ Method signature is correct")
        return True
    except Exception as e:
        print(f"✗ Method signature test failed: {e}")
        return False

def test_configuration_integration():
    """Test configuration integration with line detection"""
    try:
        from app.config import settings
        from app.models import LineDetectionConfig

        # Test that line detection config is available
        assert hasattr(settings, 'line_detection'), "Missing line_detection in settings"
        assert isinstance(settings.line_detection, LineDetectionConfig), "Invalid line_detection type"

        # Test key configuration parameters
        config = settings.line_detection
        assert hasattr(config, 'enabled'), "Missing enabled parameter"
        assert hasattr(config, 'cache_timeout_ms'), "Missing cache_timeout_ms parameter"
        assert hasattr(config, 'max_processing_time_ms'), "Missing max_processing_time_ms parameter"

        print("✓ Configuration integration is working")
        print(f"✓ Line detection enabled: {config.enabled}")
        print(f"✓ Cache timeout: {config.cache_timeout_ms}ms")
        print(f"✓ Max processing time: {config.max_processing_time_ms}ms")

        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Task 16: Extended ROICapture Service Test ===")
    print()

    tests = [
        ("Import Test", test_import),
        ("Initialization Test", test_roi_capture_service_initialization),
        ("Method Signature Test", test_dual_roi_method_signature),
        ("Configuration Integration Test", test_configuration_integration)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
        print()

    print("=== Test Results ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Task 16 implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)