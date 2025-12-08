"""
Business Logic Package for Line Detection Widget
This package provides the business logic components for line detection operations.
"""

from .line_detection_manager import (
    LineDetectionManager,
    DetectionStatus,
    DetectionResult,
    Line,
    Intersection,
)
from .api_integration import (
    APIIntegration,
    APIStatus,
    APIRequest,
    APIResponse,
    ROIConfiguration,
)
from .data_processor import (
    DataProcessor,
    DataFormat,
    DataPacket,
    ProcessingResult,
    ROIImageData,
    ProcessingStatus,
)

__all__ = [
    # Line Detection Management
    'LineDetectionManager',
    'DetectionStatus',
    'DetectionResult',
    'Line',
    'Intersection',

    # API Integration
    'APIIntegration',
    'APIStatus',
    'APIRequest',
    'APIResponse',
    'ROIConfiguration',

    # Data Processing
    'DataProcessor',
    'DataFormat',
    'DataPacket',
    'ProcessingResult',
    'ROIImageData',
    'ProcessingStatus',
]