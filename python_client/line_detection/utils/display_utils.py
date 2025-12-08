"""
Display Utilities
This module provides display and visualization utilities for line detection operations.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)

class DisplayMode(Enum):
    """Display modes for visualization."""
    NORMAL = "normal"
    ZOOMED = "zoomed"
    FULLSCREEN = "fullscreen"
    COMPACT = "compact"

class Theme(Enum):
    """Display themes."""
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"
    MEDICAL = "medical"

@dataclass
class DisplayConfig:
    """Configuration for display settings."""
    mode: DisplayMode = DisplayMode.NORMAL
    theme: Theme = Theme.DARK
    show_grid: bool = True
    show_legend: bool = True
    show_scale_bar: bool = True
    show_crosshair: bool = False
    antialiasing: bool = True
    interactive: bool = True
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 100
    background_color: str = '#1e1e1e'
    grid_color: str = '#444444'
    text_color: str = '#ffffff'

@dataclass
class ViewPort:
    """Viewport for zooming and panning operations."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    zoom_level: float = 1.0

    def contains(self, x: float, y: float) -> bool:
        """Check if point is within viewport."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def get_center(self) -> Tuple[float, float]:
        """Get viewport center."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def get_size(self) -> Tuple[float, float]:
        """Get viewport size."""
        return (self.x_max - self.x_min, self.y_max - self.y_min)

class DisplayUtils:
    """Utility class for display and visualization operations."""

    # Theme configurations
    THEMES = {
        Theme.LIGHT: {
            'background': '#ffffff',
            'grid': '#cccccc',
            'text': '#000000',
            'axes': '#333333',
            'primary': '#2196F3',
            'secondary': '#FF9800',
            'success': '#4CAF50',
            'error': '#F44336',
            'warning': '#FFC107',
        },
        Theme.DARK: {
            'background': '#1e1e1e',
            'grid': '#444444',
            'text': '#ffffff',
            'axes': '#888888',
            'primary': '#2196F3',
            'secondary': '#FF9800',
            'success': '#4CAF50',
            'error': '#F44336',
            'warning': '#FFC107',
        },
        Theme.HIGH_CONTRAST: {
            'background': '#000000',
            'grid': '#ffffff',
            'text': '#ffffff',
            'axes': '#ffffff',
            'primary': '#ffff00',
            'secondary': '#00ffff',
            'success': '#00ff00',
            'error': '#ff0000',
            'warning': '#ff8800',
        },
        Theme.MEDICAL: {
            'background': '#0a0a0a',
            'grid': '#2a2a2a',
            'text': '#00ff00',
            'axes': '#00aa00',
            'primary': '#00ffff',
            'secondary': '#ff00ff',
            'success': '#00ff00',
            'error': '#ff0000',
            'warning': '#ffff00',
        }
    }

    @staticmethod
    def get_theme_colors(theme: Theme) -> Dict[str, str]:
        """
        Get color palette for a theme.

        Args:
            theme: Theme to get colors for

        Returns:
            Dictionary of color names to hex values
        """
        return DisplayUtils.THEMES.get(theme, DisplayUtils.THEMES[Theme.DARK]).copy()

    @staticmethod
    def apply_theme_to_figure(fig: Figure, theme: Theme, config: Optional[DisplayConfig] = None):
        """
        Apply theme to matplotlib figure.

        Args:
            fig: Matplotlib figure
            theme: Theme to apply
            config: Additional display configuration
        """
        colors = DisplayUtils.get_theme_colors(theme)

        if config is None:
            config = DisplayConfig(theme=theme)
        else:
            config.background_color = colors['background']

        # Apply to figure
        fig.patch.set_facecolor(colors['background'])

        # Apply to all axes
        for ax in fig.get_axes():
            ax.set_facecolor(colors['background'])
            ax.spines['bottom'].set_color(colors['axes'])
            ax.spines['top'].set_color(colors['axes'])
            ax.spines['left'].set_color(colors['axes'])
            ax.spines['right'].set_color(colors['axes'])
            ax.xaxis.label.set_color(colors['text'])
            ax.yaxis.label.set_color(colors['text'])
            ax.tick_params(colors=colors['text'])

            # Grid styling
            if config.show_grid:
                ax.grid(True, color=colors['grid'], alpha=0.3, linestyle='-', linewidth=0.5)

    @staticmethod
    def create_figure(config: DisplayConfig) -> Figure:
        """
        Create a matplotlib figure with specified configuration.

        Args:
            config: Display configuration

        Returns:
            Configured matplotlib figure
        """
        fig = Figure(
            figsize=config.figure_size,
            dpi=config.dpi,
            facecolor=config.background_color
        )

        DisplayUtils.apply_theme_to_figure(fig, config.theme, config)

        return fig

    @staticmethod
    def create_canvas(fig: Figure, parent_widget: tk.Widget) -> FigureCanvasTkAgg:
        """
        Create a tkinter canvas for matplotlib figure.

        Args:
            fig: Matplotlib figure
            parent_widget: Parent tkinter widget

        Returns:
            Figure canvas
        """
        canvas = FigureCanvasTkAgg(fig, parent_widget)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        return canvas

    @staticmethod
    def add_scale_bar(ax: plt.Axes, length_pixels: float = 100,
                     position: str = 'lower-left',
                     color: str = '#ffffff',
                     label: str = None) -> patches.Rectangle:
        """
        Add a scale bar to the axes.

        Args:
            ax: Matplotlib axes
            length_pixels: Length of scale bar in pixels
            position: Position ('lower-left', 'lower-right', 'upper-left', 'upper-right')
            color: Scale bar color
            label: Optional label for scale bar

        Returns:
            Scale bar rectangle patch
        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Calculate position
        if position == 'lower-left':
            x, y = xlim[0] + 10, ylim[0] + 10
        elif position == 'lower-right':
            x, y = xlim[1] - length_pixels - 10, ylim[0] + 10
        elif position == 'upper-left':
            x, y = xlim[0] + 10, ylim[1] - 10
        elif position == 'upper-right':
            x, y = xlim[1] - length_pixels - 10, ylim[1] - 10
        else:
            x, y = xlim[0] + 10, ylim[0] + 10  # Default to lower-left

        # Create scale bar
        scale_bar = patches.Rectangle(
            (x, y), length_pixels, 2,
            linewidth=1, edgecolor=color, facecolor=color
        )
        ax.add_patch(scale_bar)

        # Add label if provided
        if label:
            ax.text(x + length_pixels/2, y + 5, label,
                   color=color, fontsize=8, ha='center', va='bottom')

        return scale_bar

    @staticmethod
    def add_crosshair(ax: plt.Axes, x: float, y: float,
                     color: str = '#ffff00',
                     linestyle: str = '--',
                     alpha: float = 0.7,
                     linewidth: float = 1.0) -> Tuple[plt.Line2D, plt.Line2D]:
        """
        Add crosshair to the axes.

        Args:
            ax: Matplotlib axes
            x: X coordinate
            y: Y coordinate
            color: Crosshair color
            linestyle: Line style
            alpha: Transparency
            linewidth: Line width

        Returns:
            Tuple of (horizontal_line, vertical_line)
        """
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        h_line = ax.axhline(y=y, color=color, linestyle=linestyle,
                           alpha=alpha, linewidth=linewidth)
        v_line = ax.axvline(x=x, color=color, linestyle=linestyle,
                           alpha=alpha, linewidth=linewidth)

        return h_line, v_line

    @staticmethod
    def zoom_to_region(ax: plt.Axes, x_min: float, y_min: float,
                      x_max: float, y_max: float,
                      padding: float = 0.05) -> ViewPort:
        """
        Zoom axes to specified region.

        Args:
            ax: Matplotlib axes
            x_min: Minimum X coordinate
            y_min: Minimum Y coordinate
            x_max: Maximum X coordinate
            y_max: Maximum Y coordinate
            padding: Padding around region (0.0-1.0)

        Returns:
            Viewport representing the zoomed region
        """
        # Calculate padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * padding
        y_padding = y_range * padding

        # Apply padding
        x_min_padded = x_min - x_padding
        x_max_padded = x_max + x_padding
        y_min_padded = y_min - y_padding
        y_max_padded = y_max + y_padding

        # Set axis limits
        ax.set_xlim(x_min_padded, x_max_padded)
        ax.set_ylim(y_max_padded, y_min_padded)  # Inverted for image coordinates

        # Calculate zoom level
        original_xlim = ax.get_xlim()
        original_ylim = ax.get_ylim()
        zoom_x = (original_xlim[1] - original_xlim[0]) / (x_max_padded - x_min_padded)
        zoom_y = (original_ylim[0] - original_ylim[1]) / (y_max_padded - y_min_padded)
        zoom_level = (zoom_x + zoom_y) / 2

        return ViewPort(x_min_padded, y_min_padded, x_max_padded, y_max_padded, zoom_level)

    @staticmethod
    def reset_zoom(ax: plt.Axes) -> ViewPort:
        """
        Reset axes zoom to show all data.

        Args:
            ax: Matplotlib axes

        Returns:
            Viewport representing the reset view
        """
        ax.relim()
        ax.autoscale()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        return ViewPort(xlim[0], ylim[1], xlim[1], ylim[0])  # Inverted Y for image coordinates

    @staticmethod
    def create_legend(ax: plt.Axes, items: List[Tuple[str, str]],
                     location: str = 'upper right',
                     fontsize: int = 8,
                     framealpha: float = 0.8) -> None:
        """
        Create custom legend for axes.

        Args:
            ax: Matplotlib axes
            items: List of (label, color) tuples
            location: Legend location
            fontsize: Font size
            framealpha: Background transparency
        """
        from matplotlib.lines import Line2D

        legend_elements = []
        for label, color in items:
            legend_elements.append(
                Line2D([0], [0], color=color, lw=2, label=label)
            )

        ax.legend(
            handles=legend_elements,
            loc=location,
            fontsize=fontsize,
            framealpha=framealpha,
            facecolor='#1e1e1e',
            edgecolor='#666666'
        )

    @staticmethod
    def add_annotation(ax: plt.Axes, x: float, y: float, text: str,
                     style: str = 'default',
                     **kwargs) -> plt.Annotation:
        """
        Add annotation to axes.

        Args:
            ax: Matplotlib axes
            x: X coordinate
            y: Y coordinate
            text: Annotation text
            style: Annotation style ('default', 'boxed', 'arrow')
            **kwargs: Additional annotation parameters

        Returns:
            Annotation object
        """
        default_params = {
            'fontsize': 8,
            'color': '#ffffff',
            'ha': 'center',
            'va': 'bottom'
        }

        if style == 'boxed':
            default_params.update({
                'bbox': dict(boxstyle='round,pad=0.3',
                           facecolor='black', alpha=0.7),
                'xytext': (5, 5),
                'textcoords': 'offset points'
            })
        elif style == 'arrow':
            default_params.update({
                'arrowprops': dict(arrowstyle='->', color='#ffffff'),
                'xytext': (10, 10),
                'textcoords': 'offset points'
            })

        default_params.update(kwargs)

        return ax.annotate(text, (x, y), **default_params)

    @staticmethod
    def format_coordinates(x: float, y: float, precision: int = 2) -> str:
        """
        Format coordinates for display.

        Args:
            x: X coordinate
            y: Y coordinate
            precision: Decimal precision

        Returns:
            Formatted coordinate string
        """
        return f"({x:.{precision}f}, {y:.{precision}f})"

    @staticmethod
    def calculate_optimal_figure_size(image_shape: Tuple[int, int],
                                    max_width: int = 800,
                                    max_height: int = 600,
                                    dpi: int = 100) -> Tuple[int, int]:
        """
        Calculate optimal figure size for image display.

        Args:
            image_shape: Image shape (height, width)
            max_width: Maximum display width in pixels
            max_height: Maximum display height in pixels
            dpi: Display resolution

        Returns:
            Figure size in inches (width, height)
        """
        img_height, img_width = image_shape[:2]

        # Calculate aspect ratio
        aspect_ratio = img_width / img_height

        # Calculate size in pixels
        if aspect_ratio > (max_width / max_height):
            # Width-limited
            display_width = max_width
            display_height = max_width / aspect_ratio
        else:
            # Height-limited
            display_height = max_height
            display_width = max_height * aspect_ratio

        # Convert to inches
        width_inches = display_width / dpi
        height_inches = display_height / dpi

        return (width_inches, height_inches)

    @staticmethod
    def optimize_display_performance(ax: plt.Axes,
                                   max_points: int = 1000,
                                   enable_blit: bool = True) -> None:
        """
        Optimize matplotlib axes for better performance.

        Args:
            ax: Matplotlib axes
            max_points: Maximum points to render for line plots
            enable_blit: Enable blitting for faster updates
        """
        # Set aggressive auto-scaling
        ax.autoscale(True, tight=True)

        # Enable blitting if requested
        if enable_blit:
            ax.set_animated(True)

        # Optimize rendering
        for line in ax.get_lines():
            if len(line.get_data()[0]) > max_points:
                # Downsample data
                x, y = line.get_data()
                step = len(x) // max_points
                line.set_data(x[::step], y[::step])

        # Set grid to minimal
        ax.grid(True, alpha=0.3, linewidth=0.5)

    @staticmethod
    def create_colorbar(ax: plt.Axes, mappable, label: str = None,
                       orientation: str = 'vertical',
                       shrink: float = 0.8) -> plt.colorbar.Colorbar:
        """
        Create colorbar for axes.

        Args:
            ax: Matplotlib axes
            mappable: Image or contour to create colorbar for
            label: Colorbar label
            orientation: Colorbar orientation
            shrink: Size factor

        Returns:
            Colorbar object
        """
        cbar = plt.colorbar(mappable, ax=ax, orientation=orientation, shrink=shrink)

        if label:
            cbar.set_label(label, color='#ffffff')

        # Style colorbar
        cbar.ax.yaxis.set_tick_params(color='#ffffff')
        cbar.ax.xaxis.set_tick_params(color='#ffffff')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ffffff')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='#ffffff')

        return cbar

    @staticmethod
    def export_figure(fig: Figure, filename: str,
                     dpi: int = 300,
                     format: str = 'png',
                     transparent: bool = False,
                     bbox_inches: str = 'tight') -> bool:
        """
        Export figure to file.

        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Export resolution
            format: File format
            transparent: Transparent background
            bbox_inches: Bounding box setting

        Returns:
            True if export was successful
        """
        try:
            fig.savefig(
                filename,
                dpi=dpi,
                format=format,
                transparent=transparent,
                bbox_inches=bbox_inches,
                facecolor=fig.get_facecolor()
            )
            logger.info(f"Figure exported to: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting figure: {e}")
            return False

    @staticmethod
    def create_measurement_overlay(ax: plt.Axes,
                                 points: List[Tuple[float, float]],
                                 connect_points: bool = True,
                                 show_coordinates: bool = True,
                                 color: str = '#00ff00',
                                 fontsize: int = 8) -> List[plt.Artist]:
        """
        Create measurement overlay with points and optional connections.

        Args:
            ax: Matplotlib axes
            points: List of (x, y) coordinate tuples
            connect_points: Whether to connect points with lines
            show_coordinates: Whether to show coordinate labels
            color: Overlay color
            fontsize: Font size for labels

        Returns:
            List of created matplotlib artists
        """
        artists = []

        if not points:
            return artists

        # Plot points
        x_coords, y_coords = zip(*points)
        scatter = ax.scatter(x_coords, y_coords, c=color, s=50, alpha=0.8, zorder=10)
        artists.append(scatter)

        # Connect points
        if connect_points and len(points) > 1:
            line = ax.plot(x_coords, y_coords, color=color, alpha=0.6, linewidth=1, zorder=9)[0]
            artists.append(line)

        # Add coordinate labels
        if show_coordinates:
            for i, (x, y) in enumerate(points):
                coord_text = DisplayUtils.format_coordinates(x, y, 1)
                text = ax.annotate(
                    f"P{i+1}: {coord_text}",
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=fontsize,
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
                    zorder=11
                )
                artists.append(text)

        return artists

    @staticmethod
    def add_roi_overlay(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float,
                       color: str = '#ff0000',
                       linewidth: float = 2.0,
                       alpha: float = 0.8,
                       label: str = None) -> patches.Rectangle:
        """
        Add ROI (Region of Interest) overlay to axes.

        Args:
            ax: Matplotlib axes
            x1, y1: Top-left coordinates
            x2, y2: Bottom-right coordinates
            color: ROI color
            linewidth: Border line width
            alpha: Transparency
            label: Optional label

        Returns:
            Rectangle patch
        """
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        x = min(x1, x2)
        y = min(y1, y2)

        roi_rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            alpha=alpha,
            label=label,
            zorder=15
        )
        ax.add_patch(roi_rect)

        return roi_rect

    @staticmethod
    def set_interactive_mode(fig: Figure, enabled: bool = True):
        """
        Enable or disable interactive mode for figure.

        Args:
            fig: Matplotlib figure
            enabled: Whether to enable interactive mode
        """
        if enabled:
            fig.canvas.mpl_connect('key_press_event', DisplayUtils._on_key_press)
            fig.canvas.mpl_connect('scroll_event', DisplayUtils._on_scroll)
            fig.canvas.mpl_connect('button_press_event', DisplayUtils._on_mouse_press)
        else:
            # Disconnect all interactive callbacks
            for cid in fig.canvas.callbacks.callbacks.get('key_press_event', []):
                fig.canvas.mpl_disconnect(cid)
            for cid in fig.canvas.callbacks.callbacks.get('scroll_event', []):
                fig.canvas.mpl_disconnect(cid)
            for cid in fig.canvas.callbacks.callbacks.get('button_press_event', []):
                fig.canvas.mpl_disconnect(cid)

    @staticmethod
    def _on_key_press(event):
        """Handle keyboard events for interactive mode."""
        if event.key == 'r':
            # Reset zoom on current axes
            if event.inaxes:
                DisplayUtils.reset_zoom(event.inaxes)
                event.canvas.draw()
        elif event.key == 'g':
            # Toggle grid
            if event.inaxes:
                event.inaxes.grid(not event.inaxes.grid)
                event.canvas.draw()
        elif event.key == 'f':
            # Toggle fullscreen (simplified)
            pass

    @staticmethod
    def _on_scroll(event):
        """Handle scroll events for zooming."""
        if event.inaxes:
            # Implement zoom on scroll
            ax = event.inaxes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Calculate zoom factor
            scale_factor = 1.1 if event.button == 'up' else 0.9

            # Get mouse position
            x_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
            y_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2

            # Calculate new limits
            x_range = (xlim[1] - xlim[0]) * scale_factor
            y_range = (ylim[1] - ylim[0]) * scale_factor

            new_xlim = [x_center - x_range / 2, x_center + x_range / 2]
            new_ylim = [y_center - y_range / 2, y_center + y_range / 2]

            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            event.canvas.draw()

    @staticmethod
    def _on_mouse_press(event):
        """Handle mouse press events."""
        if event.button == 3:  # Right click
            # Show context menu or reset view
            if event.inaxes:
                DisplayUtils.reset_zoom(event.inaxes)
                event.canvas.draw()