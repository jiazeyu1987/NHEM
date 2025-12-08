"""
Data Processor Component
This module handles data processing pipelines for line detection operations.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import base64
import io
from datetime import datetime
from PIL import Image
import json

from ..config import get_widget_config, get_visualization_config

logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Supported data formats."""
    BASE64_IMAGE = "base64_image"
    RAW_IMAGE = "raw_image"
    JSON = "json"
    NUMPY_ARRAY = "numpy_array"
    DETECTION_RESULT = "detection_result"
    ROI_DATA = "roi_data"

class ProcessingStatus(Enum):
    """Data processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class DataPacket:
    """Represents a data packet for processing."""
    data: Any
    format: DataFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    processing_status: ProcessingStatus = ProcessingStatus.PENDING

@dataclass
class ProcessingResult:
    """Represents the result of data processing."""
    success: bool
    processed_data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ROIImageData:
    """Represents ROI image data with metadata."""
    pixels: str  # Base64 encoded image data
    width: int
    height: int
    channels: int
    timestamp: datetime
    roi_coordinates: Optional[Dict[str, int]] = None
    confidence: float = 1.0

class DataProcessor:
    """
    Handles data processing pipelines for line detection operations.
    Provides a clean interface for data transformation and validation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data processor.

        Args:
            config: Configuration dictionary
        """
        self.config = get_widget_config(**(config or {}))
        self.viz_config = get_visualization_config('default', **(config or {}))

        # Processing pipelines
        self.processors: Dict[str, List[Callable]] = {
            'image_data': [],
            'detection_result': [],
            'roi_config': [],
            'coordinate_data': [],
        }

        # Default processors
        self._setup_default_processors()

        # Cache and performance
        self.processing_cache: Dict[str, ProcessingResult] = {}
        self.cache_enabled = self.config.get('cache', {}).get('enable_cache', True)
        self.max_cache_size = self.config.get('cache', {}).get('max_cache_size', 100)

        # Statistics
        self.processing_stats: Dict[str, Any] = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
        }

        logger.info("DataProcessor initialized")

    def _setup_default_processors(self):
        """Setup default data processors."""
        # Image data processors
        self.processors['image_data'].extend([
            self._validate_base64_image,
            self._decode_base64_image,
            self._normalize_image_data,
            self._validate_image_dimensions,
        ])

        # Detection result processors
        self.processors['detection_result'].extend([
            self._validate_detection_structure,
            self._normalize_coordinates,
            self._calculate_confidence_metrics,
            self._filter_invalid_data,
        ])

        # ROI configuration processors
        self.processors['roi_config'].extend([
            self._validate_roi_coordinates,
            self._normalize_roi_format,
            self._calculate_roi_metrics,
        ])

        # Coordinate data processors
        self.processors['coordinate_data'].extend([
            self._validate_coordinate_ranges,
            self._normalize_coordinate_precision,
            self._calculate_distance_metrics,
        ])

    def process_image_data(self, image_data: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process image data from base64 string.

        Args:
            image_data: Base64 encoded image data
            metadata: Optional metadata

        Returns:
            ProcessingResult with processed image data
        """
        try:
            start_time = datetime.now()

            # Create data packet
            packet = DataPacket(
                data=image_data,
                format=DataFormat.BASE64_IMAGE,
                metadata=metadata or {},
                source="image_data"
            )

            # Process through pipeline
            result = self._process_pipeline(packet, 'image_data')

            # Update statistics
            self._update_statistics(result, start_time)

            return result

        except Exception as e:
            logger.error(f"Error processing image data: {e}")
            return ProcessingResult(
                success=False,
                processed_data=None,
                error_message=str(e)
            )

    def process_detection_result(self, detection_data: Dict[str, Any],
                                metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process line detection result data.

        Args:
            detection_data: Raw detection result
            metadata: Optional metadata

        Returns:
            ProcessingResult with processed detection data
        """
        try:
            start_time = datetime.now()

            # Create data packet
            packet = DataPacket(
                data=detection_data,
                format=DataFormat.DETECTION_RESULT,
                metadata=metadata or {},
                source="detection_result"
            )

            # Process through pipeline
            result = self._process_pipeline(packet, 'detection_result')

            # Update statistics
            self._update_statistics(result, start_time)

            return result

        except Exception as e:
            logger.error(f"Error processing detection result: {e}")
            return ProcessingResult(
                success=False,
                processed_data=None,
                error_message=str(e)
            )

    def process_roi_configuration(self, roi_config: Dict[str, Any],
                                 metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process ROI configuration data.

        Args:
            roi_config: ROI configuration
            metadata: Optional metadata

        Returns:
            ProcessingResult with processed ROI data
        """
        try:
            start_time = datetime.now()

            # Create data packet
            packet = DataPacket(
                data=roi_config,
                format=DataFormat.ROI_DATA,
                metadata=metadata or {},
                source="roi_config"
            )

            # Process through pipeline
            result = self._process_pipeline(packet, 'roi_config')

            # Update statistics
            self._update_statistics(result, start_time)

            return result

        except Exception as e:
            logger.error(f"Error processing ROI configuration: {e}")
            return ProcessingResult(
                success=False,
                processed_data=None,
                error_message=str(e)
            )

    def process_coordinate_data(self, coordinate_data: List[Dict[str, Any]],
                               metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process coordinate data for lines and intersections.

        Args:
            coordinate_data: List of coordinate dictionaries
            metadata: Optional metadata

        Returns:
            ProcessingResult with processed coordinate data
        """
        try:
            start_time = datetime.now()

            # Create data packet
            packet = DataPacket(
                data=coordinate_data,
                format=DataFormat.JSON,
                metadata=metadata or {},
                source="coordinate_data"
            )

            # Process through pipeline
            result = self._process_pipeline(packet, 'coordinate_data')

            # Update statistics
            self._update_statistics(result, start_time)

            return result

        except Exception as e:
            logger.error(f"Error processing coordinate data: {e}")
            return ProcessingResult(
                success=False,
                processed_data=None,
                error_message=str(e)
            )

    def _process_pipeline(self, packet: DataPacket, pipeline_type: str) -> ProcessingResult:
        """
        Process a data packet through a specific pipeline.

        Args:
            packet: Data packet to process
            pipeline_type: Type of processing pipeline

        Returns:
            ProcessingResult
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(packet, pipeline_type)
            if self.cache_enabled and cache_key in self.processing_cache:
                cached_result = self.processing_cache[cache_key]
                logger.debug(f"Using cached result for {pipeline_type}")
                return cached_result

            # Update status
            packet.processing_status = ProcessingStatus.PROCESSING

            # Get processors for pipeline type
            processors = self.processors.get(pipeline_type, [])
            current_data = packet.data
            current_metadata = packet.metadata.copy()

            # Process through each processor
            for i, processor in enumerate(processors):
                try:
                    current_data, current_metadata = processor(current_data, current_metadata)
                except Exception as e:
                    logger.error(f"Error in processor {i} for {pipeline_type}: {e}")
                    return ProcessingResult(
                        success=False,
                        processed_data=None,
                        error_message=f"Processor {i} failed: {str(e)}"
                    )

            # Create successful result
            processing_time = (datetime.now() - packet.timestamp).total_seconds()
            result = ProcessingResult(
                success=True,
                processed_data=current_data,
                metadata=current_metadata,
                processing_time=processing_time
            )

            # Cache result
            if self.cache_enabled:
                self._cache_result(cache_key, result)

            packet.processing_status = ProcessingStatus.COMPLETED
            return result

        except Exception as e:
            logger.error(f"Error in processing pipeline for {pipeline_type}: {e}")
            packet.processing_status = ProcessingStatus.ERROR
            return ProcessingResult(
                success=False,
                processed_data=None,
                error_message=str(e)
            )

    # Default processors
    def _validate_base64_image(self, data: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Validate base64 image data."""
        if not isinstance(data, str):
            raise ValueError("Image data must be a string")

        if not data.strip():
            raise ValueError("Image data is empty")

        # Check if it's a data URL
        if data.startswith("data:image/"):
            metadata['is_data_url'] = True
            if "," not in data:
                raise ValueError("Invalid data URL format")
            header, encoded = data.split(",", 1)
            metadata['image_format'] = header.split(":")[1].split(";")[0]
        else:
            metadata['is_data_url'] = False
            metadata['image_format'] = 'unknown'

        return data, metadata

    def _decode_base64_image(self, data: str, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode base64 image data to numpy array."""
        try:
            if metadata.get('is_data_url', False):
                # Extract base64 part from data URL
                if "," in data:
                    _, encoded = data.split(",", 1)
                else:
                    encoded = data
            else:
                encoded = data

            # Decode base64
            image_bytes = base64.b64decode(encoded)

            # Open with PIL and convert to numpy
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')

            image_array = np.array(image)

            # Update metadata
            metadata.update({
                'image_shape': image_array.shape,
                'image_dtype': str(image_array.dtype),
                'image_mode': image.mode,
                'image_size_bytes': len(image_bytes)
            })

            return image_array, metadata

        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")

    def _normalize_image_data(self, data: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize image data for processing."""
        # Convert to float if needed
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Normalize to 0-1 range if not already
        if data.max() > 1.0:
            data = data / 255.0

        # Apply contrast adjustments if enabled
        image_config = self.viz_config.get('image', {})
        if image_config.get('auto_contrast', True):
            low_percentile = image_config.get('contrast_percentile_low', 2)
            high_percentile = image_config.get('contrast_percentile_high', 98)

            if len(data.shape) == 3:
                # Apply per channel
                for i in range(data.shape[2]):
                    channel = data[:, :, i]
                    p_low, p_high = np.percentile(channel, [low_percentile, high_percentile])
                    if p_high > p_low:
                        data[:, :, i] = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
            else:
                # Apply to single channel
                p_low, p_high = np.percentile(data, [low_percentile, high_percentile])
                if p_high > p_low:
                    data = np.clip((data - p_low) / (p_high - p_low), 0, 1)

        metadata['normalized'] = True
        return data, metadata

    def _validate_image_dimensions(self, data: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Validate image dimensions."""
        if len(data.shape) < 2 or len(data.shape) > 3:
            raise ValueError(f"Invalid image dimensions: {data.shape}")

        if len(data.shape) == 3 and data.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Invalid channel count: {data.shape[2]}")

        # Check minimum size
        min_size = 10
        if data.shape[0] < min_size or data.shape[1] < min_size:
            raise ValueError(f"Image too small: {data.shape[:2]}")

        metadata['dimensions_valid'] = True
        return data, metadata

    def _validate_detection_structure(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validate detection result structure."""
        if not isinstance(data, dict):
            raise ValueError("Detection result must be a dictionary")

        # Check required fields
        required_fields = ['success', 'lines', 'intersections']
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing field in detection result: {field}")
                data[field] = [] if field in ['lines', 'intersections'] else False

        # Validate lines and intersections
        if not isinstance(data.get('lines', []), list):
            data['lines'] = []

        if not isinstance(data.get('intersections', []), list):
            data['intersections'] = []

        metadata['structure_valid'] = True
        return data, metadata

    def _normalize_coordinates(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Normalize coordinate data."""
        precision = self.config.get('coordinate_precision', 2)

        # Normalize line coordinates
        for line in data.get('lines', []):
            if 'start' in line and isinstance(line['start'], (list, tuple)):
                line['start'] = [round(float(coord), precision) for coord in line['start']]
            if 'end' in line and isinstance(line['end'], (list, tuple)):
                line['end'] = [round(float(coord), precision) for coord in line['end']]

        # Normalize intersection coordinates
        for intersection in data.get('intersections', []):
            if 'point' in intersection and isinstance(intersection['point'], (list, tuple)):
                intersection['point'] = [round(float(coord), precision) for coord in intersection['point']]

        metadata['coordinates_normalized'] = True
        return data, metadata

    def _calculate_confidence_metrics(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Calculate confidence metrics."""
        lines = data.get('lines', [])
        intersections = data.get('intersections', [])

        # Calculate average confidences
        line_confidences = [line.get('confidence', 0.0) for line in lines if 'confidence' in line]
        intersection_confidences = [int.get('confidence', 0.0) for int in intersections if 'confidence' in int]

        metrics = {
            'line_count': len(lines),
            'intersection_count': len(intersections),
            'average_line_confidence': np.mean(line_confidences) if line_confidences else 0.0,
            'average_intersection_confidence': np.mean(intersection_confidences) if intersection_confidences else 0.0,
            'max_line_confidence': max(line_confidences) if line_confidences else 0.0,
            'max_intersection_confidence': max(intersection_confidences) if intersection_confidences else 0.0,
        }

        # Calculate overall confidence
        if metrics['intersection_count'] > 0:
            metrics['overall_confidence'] = metrics['average_intersection_confidence']
        elif metrics['line_count'] > 0:
            metrics['overall_confidence'] = metrics['average_line_confidence'] * 0.5
        else:
            metrics['overall_confidence'] = 0.0

        data['confidence_metrics'] = metrics
        metadata['confidence_metrics_calculated'] = True
        return data, metadata

    def _filter_invalid_data(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Filter out invalid data entries."""
        # Filter lines with valid coordinates
        valid_lines = []
        for line in data.get('lines', []):
            if self._is_valid_line(line):
                valid_lines.append(line)

        # Filter intersections with valid coordinates
        valid_intersections = []
        for intersection in data.get('intersections', []):
            if self._is_valid_intersection(intersection):
                valid_intersections.append(intersection)

        original_counts = {
            'lines': len(data.get('lines', [])),
            'intersections': len(data.get('intersections', []))
        }

        data['lines'] = valid_lines
        data['intersections'] = valid_intersections

        filtered_counts = {
            'lines': len(valid_lines),
            'intersections': len(valid_intersections)
        }

        metadata.update({
            'original_counts': original_counts,
            'filtered_counts': filtered_counts,
            'data_filtered': True
        })

        return data, metadata

    def _is_valid_line(self, line: Dict[str, Any]) -> bool:
        """Check if a line entry is valid."""
        try:
            start = line.get('start')
            end = line.get('end')

            if not isinstance(start, (list, tuple)) or len(start) != 2:
                return False

            if not isinstance(end, (list, tuple)) or len(end) != 2:
                return False

            # Check coordinate ranges
            for coord in start + end:
                if not isinstance(coord, (int, float)) or coord < 0:
                    return False

            # Check if start and end are different points
            if start[0] == end[0] and start[1] == end[1]:
                return False

            return True

        except Exception:
            return False

    def _is_valid_intersection(self, intersection: Dict[str, Any]) -> bool:
        """Check if an intersection entry is valid."""
        try:
            point = intersection.get('point')

            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return False

            # Check coordinate ranges
            for coord in point:
                if not isinstance(coord, (int, float)) or coord < 0:
                    return False

            return True

        except Exception:
            return False

    def _validate_roi_coordinates(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validate ROI coordinates."""
        required_keys = ['x1', 'y1', 'x2', 'y2']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing ROI coordinate: {key}")

            if not isinstance(data[key], (int, float)):
                raise ValueError(f"ROI coordinate {key} must be numeric")

            if data[key] < 0:
                raise ValueError(f"ROI coordinate {key} must be non-negative")

        # Validate coordinate order
        if data['x1'] >= data['x2'] or data['y1'] >= data['y2']:
            raise ValueError("Invalid ROI coordinate order")

        metadata['roi_coordinates_valid'] = True
        return data, metadata

    def _normalize_roi_format(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Normalize ROI format."""
        precision = self.config.get('coordinate_precision', 2)

        # Round coordinates
        for key in ['x1', 'y1', 'x2', 'y2']:
            data[key] = round(float(data[key]), precision)

        # Calculate derived values
        data['width'] = round(data['x2'] - data['x1'], precision)
        data['height'] = round(data['y2'] - data['y1'], precision)
        data['area'] = round(data['width'] * data['height'], precision)
        data['center_x'] = round((data['x1'] + data['x2']) / 2, precision)
        data['center_y'] = round((data['y1'] + data['y2']) / 2, precision)

        metadata['roi_normalized'] = True
        return data, metadata

    def _calculate_roi_metrics(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Calculate ROI metrics."""
        # Aspect ratio
        data['aspect_ratio'] = round(data['width'] / data['height'], 3) if data['height'] > 0 else 0

        # Diagonal length
        data['diagonal'] = round((data['width']**2 + data['height']**2)**0.5, precision)

        metadata['roi_metrics_calculated'] = True
        return data, metadata

    def _validate_coordinate_ranges(self, data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate coordinate ranges."""
        valid_data = []
        for item in data:
            try:
                # Extract coordinates based on item type
                if 'point' in item:
                    coords = item['point']
                elif 'start' in item and 'end' in item:
                    coords = item['start'] + item['end']
                else:
                    continue  # Skip items without recognizable coordinates

                # Validate coordinates
                if all(isinstance(coord, (int, float)) and coord >= 0 for coord in coords):
                    valid_data.append(item)

            except Exception:
                continue  # Skip invalid items

        metadata['coordinate_ranges_valid'] = True
        metadata['original_count'] = len(data)
        metadata['valid_count'] = len(valid_data)

        return valid_data, metadata

    def _normalize_coordinate_precision(self, data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Normalize coordinate precision."""
        precision = self.config.get('coordinate_precision', 2)

        for item in data:
            if 'point' in item and isinstance(item['point'], (list, tuple)):
                item['point'] = [round(float(coord), precision) for coord in item['point']]

            if 'start' in item and isinstance(item['start'], (list, tuple)):
                item['start'] = [round(float(coord), precision) for coord in item['start']]

            if 'end' in item and isinstance(item['end'], (list, tuple)):
                item['end'] = [round(float(coord), precision) for coord in item['end']]

        metadata['coordinates_precision_normalized'] = True
        return data, metadata

    def _calculate_distance_metrics(self, data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Calculate distance metrics for coordinate data."""
        total_distance = 0.0
        max_distance = 0.0
        distance_count = 0

        for item in data:
            if 'start' in item and 'end' in item:
                start = item['start']
                end = item['end']
                distance = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                item['distance'] = round(distance, 2)

                total_distance += distance
                max_distance = max(max_distance, distance)
                distance_count += 1

        if distance_count > 0:
            metadata['distance_metrics'] = {
                'total_distance': round(total_distance, 2),
                'max_distance': round(max_distance, 2),
                'average_distance': round(total_distance / distance_count, 2),
                'distance_count': distance_count
            }

        metadata['distance_metrics_calculated'] = True
        return data, metadata

    def _generate_cache_key(self, packet: DataPacket, pipeline_type: str) -> str:
        """Generate cache key for data packet."""
        try:
            # Create a simple hash based on data and pipeline type
            data_str = str(packet.data) if isinstance(packet.data, (str, int, float, bool)) else json.dumps(packet.data, sort_keys=True)
            metadata_str = json.dumps(packet.metadata, sort_keys=True)
            cache_input = f"{pipeline_type}:{data_str}:{metadata_str}"
            return str(hash(cache_input))
        except Exception:
            return f"{pipeline_type}:{packet.timestamp.timestamp()}"

    def _cache_result(self, key: str, result: ProcessingResult):
        """Cache processing result."""
        if len(self.processing_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.processing_cache))
            del self.processing_cache[oldest_key]

        self.processing_cache[key] = result

    def _update_statistics(self, result: ProcessingResult, start_time: datetime):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_stats['total_processing_time'] += processing_time

        if result.success:
            self.processing_stats['successful_processed'] += 1
        else:
            self.processing_stats['failed_processed'] += 1

        if self.processing_stats['total_processed'] > 0:
            self.processing_stats['average_processing_time'] = (
                self.processing_stats['total_processing_time'] / self.processing_stats['total_processed']
            )

    def add_processor(self, pipeline_type: str, processor: Callable):
        """
        Add a custom processor to a pipeline.

        Args:
            pipeline_type: Type of pipeline
            processor: Processing function
        """
        if pipeline_type not in self.processors:
            self.processors[pipeline_type] = []

        self.processors[pipeline_type].append(processor)
        logger.info(f"Added processor to {pipeline_type} pipeline")

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()

    def clear_cache(self):
        """Clear processing cache."""
        self.processing_cache.clear()
        logger.info("Processing cache cleared")

    def reset_statistics(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
        }
        logger.info("Processing statistics reset")