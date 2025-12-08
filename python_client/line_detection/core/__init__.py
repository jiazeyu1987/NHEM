"""
Core Components Package for Line Detection Widget
This package provides the core functionality components for line detection visualization.
"""

from .image_visualizer import ImageVisualizer
from .interaction_handler import InteractionHandler, InteractionMode
from .overlay_manager import (
    OverlayManager,
    OverlayType,
    OverlayElement,
    Point,
    Line,
)
from .coordinate_system import (
    CoordinateTransformer,
    CoordinateSystem,
    Point2D,
    BoundingBox,
)

__all__ = [
    # Image Visualization
    'ImageVisualizer',

    # Interaction Handling
    'InteractionHandler',
    'InteractionMode',

    # Overlay Management
    'OverlayManager',
    'OverlayType',
    'OverlayElement',
    'Point',
    'Line',

    # Coordinate System
    'CoordinateTransformer',
    'CoordinateSystem',
    'Point2D',
    'BoundingBox',
]