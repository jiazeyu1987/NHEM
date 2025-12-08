"""
LineDetectionWidget Configuration Module
This module contains all the widget-level configuration constants and settings.
"""

from typing import Dict, Tuple, Any, List

# Widget Display Configuration
WIDGET_DISPLAY_CONFIG = {
    'default_figsize': (8, 6),
    'default_dpi': 100,
    'save_dpi': 150,
    'max_fig_width': 12.0,
    'max_fig_height': 8.0,
    'width_scale_factor': 0.9,
    'height_scale_factor': 0.8,
}

# UI Colors and Themes
COLOR_SCHEMES = {
    'dark_theme': {
        'background': '#1e1e1e',
        'figure_bg': '#1e1e1e',
        'axes_bg': '#2d2d2d',
        'grid_color': '#444444',
        'text_color': '#ffffff',
    },
    'light_theme': {
        'background': '#ffffff',
        'figure_bg': 'white',
        'axes_bg': 'white',
        'grid_color': '#cccccc',
        'text_color': '#000000',
    }
}

# Status State Colors
STATUS_COLORS = {
    'disabled': '#808080',        # 灰色
    'enabled': '#FFA500',         # 黄色/橙色
    'success': '#00AA00',         # 绿色
    'error': '#FF0000',           # 红色
    'processing': '#FFA500',      # 橙色
}

# Error Severity Colors
ERROR_SEVERITY_COLORS = {
    'info': "#2196F3",           # 蓝色
    'warning': "#FF9800",        # 橙色
    'error': "#F44336",          # 红色
    'critical': "#B71C1C",       # 深红色
}

# Font Configuration
FONT_CONFIG = {
    'default_size': 10,
    'title_size': 12,
    'title_weight': 'bold',
    'annotation_size': 8,
    'legend_size': 8,
}

# Zoom and Pan Configuration
INTERACTION_CONFIG = {
    'zoom_in_factor': 1.1,
    'zoom_out_factor': 0.9,
    'min_zoom_level': 10,
    'max_zoom_level': 1000,
    'pan_step': 0.1,
    'scroll_step': 0.1,
}

# Update Timing Configuration
TIMING_CONFIG = {
    'update_interval': 50,      # milliseconds
    'retry_delay': 1000,        # milliseconds
    'max_retries': 3,
    'api_timeout': 5.0,         # seconds
    'connection_timeout': 2.0,  # seconds
}

# Data Processing Configuration
DATA_CONFIG = {
    'max_intersections_display': 5,
    'max_points_display': 10,
    'intersection_alpha_decay': 0.15,  # Each subsequent intersection reduces alpha by 15%
    'min_coordinate_value': 0,
    'default_line_linewidth': 2.0,
    'default_line_alpha': 0.8,
    'default_point_size': 8,
    'text_offset': 15,          # Pixel offset for annotations
}

# Quality Metrics Configuration
QUALITY_THRESHOLDS = {
    'high_confidence': 0.8,
    'medium_confidence': 0.6,
    'low_confidence': 0.4,
    'min_confidence': 0.2,
}

# Detection Status Configuration
DETECTION_CONFIG = {
    'status_check_interval': 1000,  # milliseconds
    'max_status_history': 10,
    'error_aggregation_threshold': 3,  # Number of same-type errors to aggregate
    'notification_vertical_offset': 160,  # pixels between notifications
    'notification_horizontal_offset': 20,  # pixels from right edge
}

# Network Configuration
NETWORK_CONFIG = {
    'default_host': 'localhost',
    'default_port': 8421,
    'health_check_interval': 5000,  # milliseconds
    'connection_retry_attempts': 3,
    'connection_retry_delay': 2000,  # milliseconds
}

# API Configuration
API_CONFIG = {
    'dual_realtime_endpoint': '/data/dual-realtime',
    'line_intersection_endpoint': '/api/roi/line-intersection',
    'enhanced_endpoint': '/data/dual-realtime/enhanced',
    'default_count': 100,
    'max_count': 1000,
}

# Feature Flags
FEATURES = {
    'dark_mode': True,
    'debug_mode': False,
    'enable_animations': True,
    'enable_sound_notifications': False,
    'auto_save_snapshots': False,
}

# Export and Save Configuration
EXPORT_CONFIG = {
    'default_format': 'png',
    'supported_formats': ['png', 'jpg', 'jpeg', 'pdf', 'svg'],
    'default_filename_prefix': 'line_detection',
    'include_timestamp': True,
}

# Validation Configuration
VALIDATION_CONFIG = {
    'max_title_length': 100,
    'min_title_length': 1,
    'max_description_length': 500,
    'coordinate_precision': 2,
}

# Default Widget Configuration
DEFAULT_WIDGET_CONFIG = {
    **WIDGET_DISPLAY_CONFIG,
    **COLOR_SCHEMES['dark_theme'],
    **STATUS_COLORS,
    **FONT_CONFIG,
    **INTERACTION_CONFIG,
    **TIMING_CONFIG,
    **DATA_CONFIG,
    **QUALITY_THRESHOLDS,
    **DETECTION_CONFIG,
    **NETWORK_CONFIG,
    **API_CONFIG,
    **FEATURES,
    **EXPORT_CONFIG,
    **VALIDATION_CONFIG,
}

def get_widget_config(config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get widget configuration with optional overrides.

    Args:
        config_dict: Optional dictionary with configuration overrides

    Returns:
        Complete widget configuration dictionary
    """
    if config_dict is None:
        config_dict = {}

    merged_config = DEFAULT_WIDGET_CONFIG.copy()
    merged_config.update(config_dict)

    # Handle theme-specific colors
    theme = config_dict.get('theme', 'dark')
    if theme in COLOR_SCHEMES:
        merged_config.update(COLOR_SCHEMES[theme])

    return merged_config

def get_status_colors() -> Dict[str, str]:
    """Get status state colors."""
    return STATUS_COLORS.copy()

def get_error_colors() -> Dict[str, str]:
    """Get error severity colors."""
    return ERROR_SEVERITY_COLORS.copy()