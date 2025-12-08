"""
Line Detection Package
This package provides a refactored, modular system for line detection operations.

The original monolithic line_detection_widget.py (4900+ lines) has been decomposed
into focused, maintainable components organized by functionality:

## Package Structure

### Configuration (config/)
Centralized configuration management with support for different environments and presets.

- widget_config.py: UI widget configuration and display settings
- visualization_config.py: Visualization themes, colors, and display parameters
- api_config.py: API endpoints, authentication, and communication settings

### Core Components (core/)
Essential visualization and interaction components.

- image_visualizer.py: Image display and processing
- interaction_handler.py: Mouse, keyboard, and user interaction handling
- overlay_manager.py: Visual overlays (lines, points, annotations)
- coordinate_system.py: Coordinate transformations and spatial operations

### Business Logic (business/)
High-level business logic and domain-specific operations.

- line_detection_manager.py: Detection state management and result processing
- api_integration.py: External API communication and data retrieval
- data_processor.py: Data validation, transformation, and pipeline processing

### UI Components (ui/)
User interface components and state management.

- status_display.py: Status display and user feedback
- controls_manager.py: UI controls and user interaction management

### Utilities (utils/)
Helper utilities and supporting functions.

- error_handling.py: Centralized error handling and user-friendly messages
- geometry_utils.py: Geometric calculations and spatial operations
- display_utils.py: Display utilities and visualization helpers

## Key Benefits

1. **Single Responsibility**: Each component has a clear, focused purpose
2. **Loose Coupling**: Components interact through well-defined interfaces
3. **High Cohesion**: Related functionality is grouped together
4. **Testability**: Each component can be tested independently
5. **Maintainability**: Changes are localized and predictable
6. **Extensibility**: New features can be added without modifying existing code

## Usage

```python
from line_detection import (
    ImageVisualizer,
    InteractionHandler,
    LineDetectionManager,
    StatusDisplay,
    ControlsManager,
    get_widget_config,
    get_visualization_config
)

# Create components
config = get_widget_config()
visualizer = ImageVisualizer(figure, axes, config)
interaction = InteractionHandler(figure, canvas, axes, config)
detection_manager = LineDetectionManager(config)
status_display = StatusDisplay(parent_frame, config)
controls = ControlsManager(parent_frame, config)

# Set up component interactions
detection_manager.add_callback('detection_completed', visualizer.update_display)
interaction.add_callback('coordinate_update', status_display.update_coordinates)
```

## Migration from Original

The new modular system maintains backward compatibility while providing a cleaner
architecture. The original LineDetectionWidget functionality is now distributed
across focused components that can be used independently or together as needed.
"""

# Main components
from .core import ImageVisualizer, InteractionHandler, OverlayManager, CoordinateTransformer
from .business import LineDetectionManager, APIIntegration, DataProcessor
from .ui import StatusDisplay, ControlsManager
from .utils import ErrorHandlingSystem, GeometryUtils, DisplayUtils

# Configuration
from .config import (
    get_widget_config,
    get_visualization_config,
    get_api_config,
    get_status_colors,
    get_error_colors,
)

# Types and Enums
from .core import InteractionMode, OverlayType, CoordinateSystem
from .business import DetectionStatus, APIStatus, DataFormat, ProcessingStatus
from .ui import ControlState, ControlType
from .utils import ErrorSeverity, ErrorCategory, DisplayMode, Theme

__version__ = "2.0.0"
__author__ = "Claude Code"

__all__ = [
    # Core Components
    'ImageVisualizer',
    'InteractionHandler',
    'OverlayManager',
    'CoordinateTransformer',

    # Business Logic
    'LineDetectionManager',
    'APIIntegration',
    'DataProcessor',

    # UI Components
    'StatusDisplay',
    'ControlsManager',

    # Utilities
    'ErrorHandlingSystem',
    'GeometryUtils',
    'DisplayUtils',

    # Configuration
    'get_widget_config',
    'get_visualization_config',
    'get_api_config',
    'get_status_colors',
    'get_error_colors',

    # Enums and Types
    'InteractionMode',
    'OverlayType',
    'CoordinateSystem',
    'DetectionStatus',
    'APIStatus',
    'DataFormat',
    'ProcessingStatus',
    'ControlState',
    'ControlType',
    'ErrorSeverity',
    'ErrorCategory',
    'DisplayMode',
    'Theme',
]

# Package metadata
PACKAGE_INFO = {
    'name': 'line_detection',
    'version': '2.0.0',
    'description': 'Modular line detection system with refactored architecture',
    'author': 'Claude Code',
    'created_date': '2025-12-08',
    'refactored_from': 'line_detection_widget.py (4900+ lines)',
    'components': {
        'config': 3,    # 3 configuration modules
        'core': 4,      # 4 core components
        'business': 3,  # 3 business logic modules
        'ui': 2,        # 2 UI components
        'utils': 3,     # 3 utility modules
    },
    'total_modules': 15,
    'lines_of_code_reduced': '4900+ lines distributed across 15 focused modules',
    'benefits': [
        'Single responsibility principle',
        'Loose coupling between components',
        'Improved testability',
        'Enhanced maintainability',
        'Better error handling',
        'Configuration management',
        'Type safety with dataclasses',
        'Comprehensive logging',
    ]
}