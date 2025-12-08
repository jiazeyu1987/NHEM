"""
Utilities Package for Line Detection Widget
This package provides utility functions and helper classes for line detection operations.
"""

from .error_handling import (
    ErrorHandlingSystem,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    ErrorRecord,
)
from .geometry_utils import (
    GeometryUtils,
    Point2D,
    Line2D,
    Circle2D,
)
from .display_utils import (
    DisplayUtils,
    DisplayMode,
    Theme,
    DisplayConfig,
    ViewPort,
)

__all__ = [
    # Error Handling
    'ErrorHandlingSystem',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorContext',
    'ErrorRecord',

    # Geometry Utils
    'GeometryUtils',
    'Point2D',
    'Line2D',
    'Circle2D',

    # Display Utils
    'DisplayUtils',
    'DisplayMode',
    'Theme',
    'DisplayConfig',
    'ViewPort',
]