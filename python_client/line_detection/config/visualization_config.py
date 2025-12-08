"""
Visualization Configuration Module
This module contains all visualization-specific configuration constants.
"""

from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Color Palettes
COLOR_PALETTES = {
    'default': {
        'primary': '#00FF00',      # Green (default for lines)
        'secondary': '#FF0000',    # Red (for errors/highlighting)
        'accent': '#FFA500',       # Orange (for warnings)
        'text': '#FFFFFF',         # White text
        'grid': '#444444',         # Dark grid
        'background': '#2d2d2d',   # Dark background
    },
    'medical': {
        'primary': '#00CED1',      # Dark turquoise
        'secondary': '#FF6347',    # Tomato red
        'accent': '#FFD700',       # Gold
        'text': '#F0F0F0',         # Light gray text
        'grid': '#666666',         # Medium gray grid
        'background': '#1a1a1a',   # Very dark background
    },
    'high_contrast': {
        'primary': '#00FF00',      # Bright green
        'secondary': '#FF0000',    # Bright red
        'accent': '#FFFF00',       # Bright yellow
        'text': '#FFFFFF',         # White text
        'grid': '#808080',         # Medium gray grid
        'background': '#000000',   # Black background
    }
}

# Line Visualization Configuration
LINE_VISUALIZATION = {
    'default_linestyle': '-',
    'default_linewidth': 2.0,
    'default_alpha': 0.8,
    'highlight_linewidth': 3.0,
    'highlight_alpha': 1.0,
    'confidence_alpha_high': 1.0,
    'confidence_alpha_medium': 0.8,
    'confidence_alpha_low': 0.6,
    'confidence_alpha_min': 0.4,
}

# Point Visualization Configuration
POINT_VISUALIZATION = {
    'default_marker': 'o',
    'default_size': 8,
    'highlight_size': 12,
    'default_alpha': 0.8,
    'highlight_alpha': 1.0,
    'edge_width': 1.0,
    'highlight_edge_width': 2.0,
}

# Intersection Visualization Configuration
INTERSECTION_VISUALIZATION = {
    'marker': 'x',
    'size': 10,
    'linewidth': 2.0,
    'alpha_base': 1.0,
    'alpha_decay_rate': 0.15,  # Each subsequent intersection reduces alpha by 15%
    'max_display_count': 5,
    'halo_size': 15,
    'halo_alpha': 0.3,
    'connection_linewidth': 1.0,
    'connection_alpha': 0.5,
    'connection_linestyle': '--',
}

# Annotation Configuration
ANNOTATION_CONFIG = {
    'font_family': 'monospace',
    'font_size': 8,
    'font_weight': 'normal',
    'text_color': '#FFFFFF',
    'background_color': 'black',
    'background_alpha': 0.7,
    'border_color': '#FFFFFF',
    'border_width': 1,
    'arrow_style': '->',
    'arrow_size': 8,
    'text_offset': 15,  # pixels
    'max_text_length': 50,
    'precision': 2,  # decimal places for coordinates
}

# Grid and Axes Configuration
GRID_CONFIG = {
    'show_grid': True,
    'grid_alpha': 0.3,
    'grid_linestyle': '-',
    'grid_linewidth': 0.5,
    'major_grid_alpha': 0.5,
    'major_grid_linewidth': 1.0,
    'minor_grid_alpha': 0.2,
    'minor_grid_linewidth': 0.3,
    'show_ticks': True,
    'tick_length': 4,
    'tick_width': 1,
    'label_padding': 5,
}

# Scale Bar Configuration
SCALE_BAR_CONFIG = {
    'show_scale_bar': True,
    'scale_bar_length': 100,  # pixels
    'scale_bar_height': 2,    # pixels
    'scale_bar_color': '#FFFFFF',
    'scale_bar_alpha': 0.8,
    'scale_bar_position': 'lower-left',
    'scale_bar_offset': (10, 10),  # pixels from corner
    'scale_bar_label_size': 8,
    'scale_bar_label_color': '#FFFFFF',
}

# ROI Visualization Configuration
ROI_VISUALIZATION = {
    'border_linewidth': 2.0,
    'border_alpha': 0.8,
    'border_style': '-',
    'corner_marker_size': 6,
    'corner_marker_style': 's',
    'fill_alpha': 0.1,
    'selection_border_color': '#FFD700',
    'hover_border_color': '#00FF00',
    'active_border_color': '#FF0000',
    'resize_handle_size': 8,
    'resize_handle_color': '#FFA500',
}

# Image Processing Configuration
IMAGE_PROCESSING = {
    'interpolation': 'bilinear',
    'colormap': 'gray',
    'vmin': None,
    'vmax': None,
    'aspect': 'equal',
    'auto_contrast': True,
    'contrast_percentile_low': 2,
    'contrast_percentile_high': 98,
    'sharpen_kernel': None,
    'noise_reduction': False,
}

# Animation Configuration
ANIMATION_CONFIG = {
    'enable_animations': True,
    'fade_in_duration': 200,   # milliseconds
    'fade_out_duration': 150,  # milliseconds
    'pulse_duration': 1000,    # milliseconds
    'pulse_alpha_min': 0.3,
    'pulse_alpha_max': 1.0,
    'transition_duration': 300,  # milliseconds
    'frame_rate': 30,  # FPS for animations
}

# Performance Optimization Configuration
PERFORMANCE_CONFIG = {
    'max_points_rendered': 1000,
    'max_lines_rendered': 100,
    'max_intersections_rendered': 50,
    'culling_distance': 1000,  # pixels
    'level_of_detail': True,
    'simplify_lines': True,
    'simplify_tolerance': 1.0,  # pixels
    'use_blit': True,  # matplotlib blitting for performance
    'redraw_throttle': 16,  # milliseconds (60 FPS)
}

# Export Configuration
EXPORT_VISUALIZATION = {
    'default_dpi': 150,
    'high_dpi': 300,
    'export_format': 'png',
    'transparent_background': False,
    'bbox_inches': 'tight',
    'pad_inches': 0.1,
    'face_color': '#1e1e1e',
    'edge_color': 'none',
    'include_annotations': True,
    'include_scale_bar': True,
    'include_grid': True,
}

# Accessibility Configuration
ACCESSIBILITY_CONFIG = {
    'high_contrast_mode': False,
    'large_font_mode': False,
    'color_blind_friendly': False,
    'daltionism_mode': None,  # 'protanopia', 'deuteranopia', 'tritanopia'
    'min_line_width': 2.0,
    'min_font_size': 10,
    'contrast_ratio_threshold': 4.5,  # WCAG AA standard
}

# Custom Colormaps
def create_custom_colormaps():
    """Create custom colormaps for medical visualization."""

    # Medical-grade colormap for signal intensity
    medical_colors = ['#000033', '#000055', '#0000ff', '#00ffff', '#ffff00', '#ff0000', '#ffffff']
    medical_cmap = ListedColormap(medical_colors, name='medical')

    # High contrast colormap
    high_contrast_colors = ['#000000', '#404040', '#808080', '#c0c0c0', '#ffffff']
    high_contrast_cmap = ListedColormap(high_contrast_colors, name='high_contrast')

    # Register colormaps
    plt.register_cmap(cmap=medical_cmap)
    plt.register_cmap(cmap=high_contrast_cmap)

    return {
        'medical': medical_cmap,
        'high_contrast': high_contrast_cmap
    }

# Visualization Settings Presets
VISUALIZATION_PRESETS = {
    'default': {
        'colors': COLOR_PALETTES['default'],
        'lines': LINE_VISUALIZATION,
        'points': POINT_VISUALIZATION,
        'intersections': INTERSECTION_VISUALIZATION,
        'annotations': ANNOTATION_CONFIG,
        'grid': GRID_CONFIG,
        'performance': PERFORMANCE_CONFIG,
    },
    'medical_grade': {
        'colors': COLOR_PALETTES['medical'],
        'lines': {**LINE_VISUALIZATION, 'default_linewidth': 2.5},
        'points': {**POINT_VISUALIZATION, 'default_size': 10},
        'intersections': {**INTERSECTION_VISUALIZATION, 'size': 12},
        'annotations': {**ANNOTATION_CONFIG, 'font_size': 10},
        'grid': {**GRID_CONFIG, 'grid_alpha': 0.4},
        'image': IMAGE_PROCESSING,
    },
    'high_performance': {
        'colors': COLOR_PALETTES['high_contrast'],
        'lines': LINE_VISUALIZATION,
        'points': {**POINT_VISUALIZATION, 'default_size': 6},
        'intersections': {**INTERSECTION_VISUALIZATION, 'size': 8},
        'performance': {**PERFORMANCE_CONFIG, 'max_points_rendered': 500, 'use_blit': True},
        'animations': {**ANIMATION_CONFIG, 'enable_animations': False},
    },
    'accessibility': {
        'colors': COLOR_PALETTES['high_contrast'],
        'lines': {**LINE_VISUALIZATION, 'default_linewidth': 3.0},
        'points': {**POINT_VISUALIZATION, 'default_size': 12},
        'intersections': {**INTERSECTION_VISUALIZATION, 'size': 15, 'linewidth': 3.0},
        'annotations': {**ANNOTATION_CONFIG, 'font_size': 12, 'font_weight': 'bold'},
        'accessibility': ACCESSIBILITY_CONFIG,
    }
}

def get_visualization_config(preset: str = 'default', **overrides) -> Dict[str, Any]:
    """
    Get visualization configuration with preset and overrides.

    Args:
        preset: Name of the preset configuration
        **overrides: Configuration overrides

    Returns:
        Complete visualization configuration dictionary
    """
    if preset not in VISUALIZATION_PRESETS:
        preset = 'default'

    config = VISUALIZATION_PRESETS[preset].copy()

    # Apply overrides
    for key, value in overrides.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value

    return config

def get_color_palette(palette_name: str = 'default') -> Dict[str, str]:
    """Get a specific color palette."""
    return COLOR_PALETTES.get(palette_name, COLOR_PALETTES['default']).copy()