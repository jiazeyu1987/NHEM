"""
Configuration Package for Line Detection Widget
This package provides centralized configuration management for all line detection components.
"""

from .widget_config import (
    get_widget_config,
    get_status_colors,
    get_error_colors,
    DEFAULT_WIDGET_CONFIG,
    STATUS_COLORS,
    ERROR_SEVERITY_COLORS,
)

from .visualization_config import (
    get_visualization_config,
    get_color_palette,
    create_custom_colormaps,
    VISUALIZATION_PRESETS,
    COLOR_PALETTES,
)

from .api_config import (
    get_api_config,
    get_endpoint_url,
    validate_api_response,
    APIEnvironment,
    HTTPMethod,
    DEFAULT_ENDPOINTS,
)

__all__ = [
    # Widget Configuration
    'get_widget_config',
    'get_status_colors',
    'get_error_colors',
    'DEFAULT_WIDGET_CONFIG',
    'STATUS_COLORS',
    'ERROR_SEVERITY_COLORS',

    # Visualization Configuration
    'get_visualization_config',
    'get_color_palette',
    'create_custom_colormaps',
    'VISUALIZATION_PRESETS',
    'COLOR_PALETTES',

    # API Configuration
    'get_api_config',
    'get_endpoint_url',
    'validate_api_response',
    'APIEnvironment',
    'HTTPMethod',
    'DEFAULT_ENDPOINTS',
]