"""
Line Detection Manager Component
This module handles the business logic for line detection and intersection analysis.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time
from datetime import datetime

from ..config import get_widget_config, get_api_config

logger = logging.getLogger(__name__)

class DetectionStatus(Enum):
    """Line detection status states."""
    DISABLED = "disabled"
    ENABLED_NO_DETECTION = "enabled_no_detection"
    DETECTION_SUCCESS = "detection_success"
    DETECTION_ERROR = "detection_error"
    PROCESSING = "processing"

@dataclass
class DetectionResult:
    """Represents a line detection result."""
    success: bool
    lines: List[Dict[str, Any]]
    intersections: List[Dict[str, Any]]
    confidence: float
    timestamp: datetime
    error_message: Optional[str] = None
    processing_time: float = 0.0

@dataclass
class Line:
    """Represents a detected line."""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    confidence: float
    angle: float
    length: float
    line_type: str = "unknown"

@dataclass
class Intersection:
    """Represents a line intersection."""
    point: Tuple[float, float]
    confidence: float
    intersecting_lines: List[str]  # Line identifiers
    intersection_type: str = "cross"
    distance_from_center: float = 0.0

class LineDetectionManager:
    """
    Business logic manager for line detection operations.
    Handles detection state, result processing, and callbacks.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the line detection manager.

        Args:
            config: Configuration dictionary
        """
        self.config = get_widget_config(**(config or {}))
        self.api_config = get_api_config(**(config or {}))

        # Detection state
        self.detection_status: DetectionStatus = DetectionStatus.DISABLED
        self.detection_enabled: bool = False
        self.last_detection_result: Optional[DetectionResult] = None
        self.detection_history: List[DetectionResult] = []

        # Performance metrics
        self.detection_count: int = 0
        self.successful_detections: int = 0
        self.failed_detections: int = 0
        self.total_processing_time: float = 0.0
        self.average_processing_time: float = 0.0

        # Threading and async operations
        self.detection_lock = threading.Lock()
        self.processing_detection: bool = False

        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'status_changed': [],
            'detection_completed': [],
            'error_occurred': [],
            'progress_updated': [],
        }

        # Configuration parameters
        self.max_history_size = self.config.get('max_status_history', 10)
        self.processing_timeout = self.config.get('max_processing_time', 10.0)
        self.error_threshold = self.config.get('error_aggregation_threshold', 3)

        # Error tracking
        self.recent_errors: List[str] = []
        self.error_counts: Dict[str, int] = {}

        logger.info("LineDetectionManager initialized")

    def set_detection_enabled(self, enabled: bool):
        """
        Enable or disable line detection.

        Args:
            enabled: Whether detection should be enabled
        """
        with self.detection_lock:
            if self.detection_enabled != enabled:
                self.detection_enabled = enabled

                if enabled:
                    self.detection_status = DetectionStatus.ENABLED_NO_DETECTION
                    logger.info("Line detection enabled")
                else:
                    self.detection_status = DetectionStatus.DISABLED
                    logger.info("Line detection disabled")

                self._trigger_callbacks('status_changed', self.detection_status)

    def get_detection_status(self) -> DetectionStatus:
        """Get current detection status."""
        return self.detection_status

    def process_detection_result(self, result_data: Dict[str, Any]) -> DetectionResult:
        """
        Process raw detection result data into structured format.

        Args:
            result_data: Raw detection result from API

        Returns:
            Processed DetectionResult
        """
        try:
            start_time = time.time()

            # Extract basic information
            success = result_data.get('success', False)
            lines_data = result_data.get('lines', [])
            intersections_data = result_data.get('intersections', [])
            error_message = result_data.get('error_message')
            confidence = result_data.get('overall_confidence', 0.0)

            # Process lines
            processed_lines = []
            for line_data in lines_data:
                processed_line = self._process_line_data(line_data)
                processed_lines.append(processed_line.__dict__)

            # Process intersections
            processed_intersections = []
            for intersection_data in intersections_data:
                processed_intersection = self._process_intersection_data(intersection_data)
                processed_intersections.append(processed_intersection.__dict__)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create detection result
            result = DetectionResult(
                success=success,
                lines=processed_lines,
                intersections=processed_intersections,
                confidence=confidence,
                timestamp=datetime.now(),
                error_message=error_message,
                processing_time=processing_time
            )

            # Update statistics
            self._update_statistics(result)

            # Store result
            self.last_detection_result = result
            self._add_to_history(result)

            return result

        except Exception as e:
            logger.error(f"Error processing detection result: {e}")
            # Return error result
            return DetectionResult(
                success=False,
                lines=[],
                intersections=[],
                confidence=0.0,
                timestamp=datetime.now(),
                error_message=str(e),
                processing_time=0.0
            )

    def _process_line_data(self, line_data: Dict[str, Any]) -> Line:
        """Process raw line data into Line object."""
        try:
            start_point = line_data.get('start', [0, 0])
            end_point = line_data.get('end', [0, 0])
            confidence = line_data.get('confidence', 1.0)

            # Calculate angle and length
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length = (dx**2 + dy**2)**0.5
            angle = math.degrees(math.atan2(dy, dx))

            line_type = self._classify_line(angle, confidence)

            return Line(
                start_point=(start_point[0], start_point[1]),
                end_point=(end_point[0], end_point[1]),
                confidence=confidence,
                angle=angle,
                length=length,
                line_type=line_type
            )

        except Exception as e:
            logger.error(f"Error processing line data: {e}")
            return Line(
                start_point=(0, 0),
                end_point=(0, 0),
                confidence=0.0,
                angle=0.0,
                length=0.0
            )

    def _process_intersection_data(self, intersection_data: Dict[str, Any]) -> Intersection:
        """Process raw intersection data into Intersection object."""
        try:
            point_data = intersection_data.get('point', [0, 0])
            confidence = intersection_data.get('confidence', 1.0)
            intersecting_lines = intersection_data.get('intersecting_lines', [])
            intersection_type = intersection_data.get('type', 'cross')

            return Intersection(
                point=(point_data[0], point_data[1]),
                confidence=confidence,
                intersecting_lines=intersecting_lines,
                intersection_type=intersection_type
            )

        except Exception as e:
            logger.error(f"Error processing intersection data: {e}")
            return Intersection(
                point=(0, 0),
                confidence=0.0,
                intersecting_lines=[],
                intersection_type='unknown'
            )

    def _classify_line(self, angle: float, confidence: float) -> str:
        """
        Classify line based on angle and confidence.

        Args:
            angle: Line angle in degrees
            confidence: Line confidence (0.0-1.0)

        Returns:
            Line type classification
        """
        if confidence < 0.5:
            return "low_confidence"

        # Normalize angle to 0-180 range
        angle = angle % 180

        # Classify based on angle ranges
        if abs(angle - 0) < 15 or abs(angle - 180) < 15:
            return "horizontal"
        elif abs(angle - 90) < 15:
            return "vertical"
        elif 45 <= angle <= 135:
            return "diagonal"
        else:
            return "irregular"

    def _update_statistics(self, result: DetectionResult):
        """Update detection statistics."""
        self.detection_count += 1
        self.total_processing_time += result.processing_time

        if result.success:
            self.successful_detections += 1
        else:
            self.failed_detections += 1

        # Calculate average processing time
        if self.detection_count > 0:
            self.average_processing_time = self.total_processing_time / self.detection_count

    def _add_to_history(self, result: DetectionResult):
        """Add result to detection history."""
        self.detection_history.append(result)

        # Maintain history size limit
        if len(self.detection_history) > self.max_history_size:
            self.detection_history.pop(0)

    def trigger_manual_detection(self) -> bool:
        """
        Trigger a manual detection operation.

        Returns:
            True if detection was triggered successfully
        """
        with self.detection_lock:
            if self.processing_detection:
                logger.warning("Detection already in progress")
                return False

            if not self.detection_enabled:
                logger.warning("Detection is not enabled")
                return False

            self.processing_detection = True
            self.detection_status = DetectionStatus.PROCESSING

            # Trigger callbacks
            self._trigger_callbacks('status_changed', self.detection_status)
            self._trigger_callbacks('progress_updated', 0.0)

            # Start detection in background thread
            thread = threading.Thread(target=self._execute_manual_detection, daemon=True)
            thread.start()

            return True

    def _execute_manual_detection(self):
        """Execute manual detection in background thread."""
        try:
            logger.info("Starting manual detection")

            # This would typically call the API client
            # For now, simulate the process
            time.sleep(1.0)  # Simulate processing time

            # Trigger progress update
            self._trigger_callbacks('progress_updated', 0.5)

            # Simulate result
            result_data = {
                'success': True,
                'lines': [],
                'intersections': [],
                'overall_confidence': 0.8
            }

            # Process result
            result = self.process_detection_result(result_data)

            # Update status based on result
            if result.success and result.intersections:
                self.detection_status = DetectionStatus.DETECTION_SUCCESS
            elif result.success:
                self.detection_status = DetectionStatus.ENABLED_NO_DETECTION
            else:
                self.detection_status = DetectionStatus.DETECTION_ERROR
                self._track_error(result.error_message or "Unknown error")

            # Trigger callbacks
            self._trigger_callbacks('detection_completed', result)
            self._trigger_callbacks('status_changed', self.detection_status)
            self._trigger_callbacks('progress_updated', 1.0)

            logger.info("Manual detection completed successfully")

        except Exception as e:
            logger.error(f"Error in manual detection: {e}")
            self.detection_status = DetectionStatus.DETECTION_ERROR
            self._track_error(str(e))

            # Create error result
            error_result = DetectionResult(
                success=False,
                lines=[],
                intersections=[],
                confidence=0.0,
                timestamp=datetime.now(),
                error_message=str(e)
            )

            # Trigger error callbacks
            self._trigger_callbacks('error_occurred', e)
            self._trigger_callbacks('detection_completed', error_result)
            self._trigger_callbacks('status_changed', self.detection_status)

        finally:
            with self.detection_lock:
                self.processing_detection = False

    def _track_error(self, error_message: str):
        """Track error occurrences."""
        self.recent_errors.append(error_message)

        # Maintain recent errors list size
        if len(self.recent_errors) > 10:
            self.recent_errors.pop(0)

        # Count error occurrences
        error_key = error_message[:50]  # Use first 50 chars as key
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        logger.warning(f"Detection error tracked: {error_message}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics.

        Returns:
            Dictionary with detection statistics
        """
        success_rate = 0.0
        if self.detection_count > 0:
            success_rate = (self.successful_detections / self.detection_count) * 100

        return {
            'detection_count': self.detection_count,
            'successful_detections': self.successful_detections,
            'failed_detections': self.failed_detections,
            'success_rate': success_rate,
            'average_processing_time': self.average_processing_time,
            'total_processing_time': self.total_processing_time,
            'current_status': self.detection_status.value,
            'detection_enabled': self.detection_enabled,
            'processing_detection': self.processing_detection,
            'recent_errors': self.recent_errors[-5:],  # Last 5 errors
            'error_counts': dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5])  # Top 5 errors
        }

    def get_last_result(self) -> Optional[DetectionResult]:
        """Get the most recent detection result."""
        return self.last_detection_result

    def get_history(self, limit: Optional[int] = None) -> List[DetectionResult]:
        """
        Get detection history.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of detection results
        """
        if limit is None:
            return self.detection_history.copy()
        else:
            return self.detection_history[-limit:]

    def clear_history(self):
        """Clear detection history and reset statistics."""
        self.detection_history = []
        self.detection_count = 0
        self.successful_detections = 0
        self.failed_detections = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        self.recent_errors = []
        self.error_counts = {}
        self.last_detection_result = None

        logger.info("Detection history and statistics cleared")

    def add_callback(self, event_type: str, callback: Callable):
        """
        Add a callback for detection events.

        Args:
            event_type: Type of event ('status_changed', 'detection_completed', etc.)
            callback: Callback function to call
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown callback event type: {event_type}")

    def remove_callback(self, event_type: str, callback: Callable):
        """
        Remove a callback for detection events.

        Args:
            event_type: Type of event
            callback: Callback function to remove
        """
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)

    def _trigger_callbacks(self, event_type: str, *args, **kwargs):
        """Trigger all callbacks for a specific event type."""
        try:
            for callback in self.callbacks.get(event_type, []):
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {event_type} callback: {e}")
        except Exception as e:
            logger.error(f"Error triggering {event_type} callbacks: {e}")

    def reset(self):
        """Reset the detection manager to initial state."""
        with self.detection_lock:
            self.detection_enabled = False
            self.detection_status = DetectionStatus.DISABLED
            self.processing_detection = False
            self.clear_history()

        logger.info("LineDetectionManager reset")

    def validate_detection_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate detection parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check required parameters
            if 'roi_coordinates' in parameters:
                roi = parameters['roi_coordinates']
                if not isinstance(roi, dict):
                    return False, "ROI coordinates must be a dictionary"

                required_keys = ['x1', 'y1', 'x2', 'y2']
                for key in required_keys:
                    if key not in roi:
                        return False, f"Missing ROI coordinate: {key}"

                    if not isinstance(roi[key], (int, float)):
                        return False, f"ROI coordinate {key} must be numeric"

                    if roi[key] < 0:
                        return False, f"ROI coordinate {key} must be non-negative"

                # Validate coordinate order
                if roi['x1'] >= roi['x2'] or roi['y1'] >= roi['y2']:
                    return False, "Invalid ROI coordinate order"

            # Validate threshold parameters
            for threshold_key in ['confidence_threshold', 'line_length_threshold']:
                if threshold_key in parameters:
                    threshold = parameters[threshold_key]
                    if not isinstance(threshold, (int, float)):
                        return False, f"{threshold_key} must be numeric"
                    if not 0.0 <= threshold <= 1.0:
                        return False, f"{threshold_key} must be between 0.0 and 1.0"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def export_results(self, format: str = 'json') -> Dict[str, Any]:
        """
        Export detection results and statistics.

        Args:
            format: Export format ('json', 'csv', 'xml')

        Returns:
            Exported data
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'last_result': self.last_detection_result.__dict__ if self.last_detection_result else None,
            'history_count': len(self.detection_history),
            'configuration': {
                'detection_enabled': self.detection_enabled,
                'max_history_size': self.max_history_size,
                'processing_timeout': self.processing_timeout,
            }
        }

        if format.lower() == 'json':
            # Add full history for JSON export
            export_data['history'] = [result.__dict__ for result in self.detection_history]

        return export_data