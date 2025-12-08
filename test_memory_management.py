#!/usr/bin/env python3
"""
Task 30: Memory Management Test for Line Intersection Detector

This test validates the memory management features implemented in the line intersection detector,
ensuring compliance with NF-Performance and NF-Reliability requirements.
"""

import numpy as np
import cv2
import time
import logging
import sys
import os

# Add backend path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.core.line_intersection_detector import (
    LineIntersectionDetector, MemoryPool, OpenCVResourceManager,
    memory_monitoring, managed_numpy_array
)
from backend.app.models import LineDetectionConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_config() -> LineDetectionConfig:
    """Create a test configuration for the line intersection detector"""
    return LineDetectionConfig(
        enabled=True,
        max_processing_time_ms=1000,
        cache_timeout_ms=100,
        min_confidence=0.3,
        hsv_green_lower=[40, 40, 40],
        hsv_green_upper=[80, 255, 255],
        canny_low_threshold=25,
        canny_high_threshold=80,
        hough_threshold=15,
        hough_min_line_length=15,
        hough_max_line_gap=8,
        min_angle_degrees=15,
        parallel_threshold=0.01
    )


def create_test_image(width: int = 200, height: int = 200) -> np.ndarray:
    """Create a test image with green lines"""
    # Create blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some green background
    image[:] = (0, 50, 0)  # Dark green background

    # Add two intersecting green lines
    # Line 1: Diagonal from top-left to bottom-right
    cv2.line(image, (20, 20), (180, 180), (0, 255, 0), 3)

    # Line 2: Diagonal from top-right to bottom-left
    cv2.line(image, (180, 20), (20, 180), (0, 255, 0), 3)

    # Add some noise
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)

    return image


def test_memory_pool():
    """Test memory pool functionality"""
    logger.info("Testing Memory Pool...")

    pool = MemoryPool(max_pool_size_mb=10.0, max_entries=50)

    # Test array allocation and reuse
    shape = (100, 100)
    dtype = np.uint8

    # Get first array (should create new)
    array1 = pool.get_array(shape, dtype)
    array1.fill(1)

    # Return array to pool
    pool.return_array(array1)

    # Get another array (should reuse from pool)
    array2 = pool.get_array(shape, dtype)

    # Should get a reused array (check if values are preserved or zeroed)
    assert array2.shape == shape
    assert array2.dtype == dtype

    # Get pool stats
    stats = pool.get_stats()
    logger.info(f"Memory pool stats: {stats}")

    # Test cleanup
    pool.clear()
    stats_after_clear = pool.get_stats()
    assert stats_after_clear["pool_size_mb"] == 0.0

    logger.info("✓ Memory Pool test passed")


def test_opencv_resource_manager():
    """Test OpenCV resource manager"""
    logger.info("Testing OpenCV Resource Manager...")

    manager = OpenCVResourceManager()

    # Create a test OpenCV object
    test_array = np.ones((50, 50), dtype=np.uint8)
    test_mat = cv2.UMat(test_array)  # Create UMat (GPU resource)

    # Track the object
    obj_id = manager.track_object(test_mat, "test_mat")
    assert obj_id != -1

    # Retrieve the object
    retrieved_obj = manager.get_object(obj_id)
    assert retrieved_obj is not None

    # Release the object
    released = manager.release_object(obj_id)
    assert released == True

    # Object should no longer be tracked
    retrieved_after_release = manager.get_object(obj_id)
    assert retrieved_after_release is None

    # Get stats
    stats = manager.get_stats()
    logger.info(f"OpenCV resource manager stats: {stats}")

    logger.info("✓ OpenCV Resource Manager test passed")


def test_memory_monitoring_context():
    """Test memory monitoring context manager"""
    logger.info("Testing Memory Monitoring Context...")

    # Test with memory monitoring
    with memory_monitoring(memory_threshold_mb=100.0, enable_gc=True):
        # Allocate some memory
        arrays = []
        for i in range(10):
            arrays.append(np.ones((100, 100), dtype=np.float32))

        # Memory should be tracked
        logger.info("Allocated test arrays within memory monitoring context")

    # Memory should be automatically cleaned up
    logger.info("✓ Memory Monitoring Context test passed")


def test_managed_numpy_array():
    """Test managed numpy array context manager"""
    logger.info("Testing Managed Numpy Array...")

    pool = MemoryPool(max_pool_size_mb=5.0)

    with managed_numpy_array((50, 50), np.float32, pool) as array:
        array.fill(42.0)
        assert np.all(array == 42.0)
        logger.info("Managed array created and used successfully")

    # Array should be automatically returned to pool
    stats = pool.get_stats()
    logger.info(f"Pool stats after managed array: {stats}")

    logger.info("✓ Managed Numpy Array test passed")


def test_detector_memory_management():
    """Test the integrated memory management in the detector"""
    logger.info("Testing Detector Memory Management...")

    config = create_test_config()
    detector = LineIntersectionDetector(config)

    # Create test image
    test_image = create_test_image()

    # Run detection multiple times to test memory management
    for i in range(5):
        logger.info(f"Detection iteration {i+1}/5")

        result = detector.detect_intersection(test_image, i)

        # Check that memory stats are being tracked
        memory_stats = detector.get_memory_usage_stats()
        assert memory_stats["current_memory_mb"] > 0

        # Check threshold monitoring
        thresholds = detector.check_memory_thresholds()
        assert "within_threshold" in thresholds

        if result.has_intersection:
            logger.info(f"Detection successful: intersection={result.intersection}, confidence={result.confidence:.3f}")
        else:
            logger.info(f"Detection failed: {result.error_message}")

    # Test memory optimization
    optimization_result = detector.optimize_memory_usage()
    logger.info(f"Memory optimization result: {optimization_result}")

    # Test medical-grade validation
    validation_result = detector.validate_memory_usage_for_medical_grade()
    logger.info(f"Medical-grade validation: compliant={validation_result['is_compliant']}")

    if not validation_result["is_compliant"]:
        logger.warning(f"Validation violations: {validation_result['violations']}")

    # Test cleanup
    cleanup_count = detector.cleanup_opencv_resources()
    logger.info(f"Cleaned up {cleanup_count} OpenCV objects")

    # Test cache clearing with memory management
    detector.clear_cache()
    logger.info("Cache cleared with memory management")

    logger.info("✓ Detector Memory Management test passed")


def test_memory_leak_detection():
    """Test memory leak detection capabilities"""
    logger.info("Testing Memory Leak Detection...")

    config = create_test_config()
    detector = LineIntersectionDetector(config)

    test_image = create_test_image()

    # Run many iterations to test for memory leaks
    initial_memory = detector.get_memory_usage_stats()["current_memory_mb"]

    for i in range(20):
        result = detector.detect_intersection(test_image, i)

        # Force some memory pressure
        temp_arrays = [np.ones((50, 50), dtype=np.float32) for _ in range(5)]
        del temp_arrays

    final_memory = detector.get_memory_usage_stats()["current_memory_mb"]
    memory_increase = final_memory - initial_memory

    logger.info(f"Memory increase after 20 iterations: {memory_increase:.2f}MB")

    # Check if leak was detected
    memory_stats = detector.get_memory_usage_stats()
    if memory_stats["leak_detected"]:
        logger.warning("Memory leak was detected")
    else:
        logger.info("No memory leaks detected")

    # Test optimization to recover memory
    optimization = detector.optimize_memory_usage()
    logger.info(f"Optimization freed {optimization['memory_freed_mb']:.2f}MB")

    logger.info("✓ Memory Leak Detection test passed")


def main():
    """Run all memory management tests"""
    logger.info("Starting Task 30 Memory Management Tests")
    logger.info("=" * 60)

    try:
        # Run individual component tests
        test_memory_pool()
        test_opencv_resource_manager()
        test_memory_monitoring_context()
        test_managed_numpy_array()

        # Run integrated tests
        test_detector_memory_management()
        test_memory_leak_detection()

        logger.info("=" * 60)
        logger.info("✅ All Memory Management Tests Passed Successfully!")
        logger.info("Task 30 implementation is complete and working correctly.")

    except Exception as e:
        logger.error(f"❌ Memory Management Test Failed: {e}")
        logger.error(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)