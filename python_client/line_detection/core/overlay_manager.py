"""
Overlay Manager Component
This module handles all overlay elements including lines, points, annotations, and visual indicators.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from ..config import get_visualization_config

logger = logging.getLogger(__name__)

class OverlayType(Enum):
    """Types of overlay elements."""
    LINE = "line"
    POINT = "point"
    INTERSECTION = "intersection"
    TEXT = "text"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"
    ARROW = "arrow"
    CONFIDENCE_INDICATOR = "confidence_indicator"
    SCALE_BAR = "scale_bar"
    CROSSHAIR = "crosshair"

@dataclass
class OverlayElement:
    """Represents a single overlay element."""
    element_type: OverlayType
    matplotlib_object: Any
    data: Dict[str, Any]
    visible: bool = True
    z_order: int = 5
    group: Optional[str] = None

@dataclass
class Point:
    """Represents a point with optional metadata."""
    x: float
    y: float
    confidence: float = 1.0
    label: Optional[str] = None
    color: Optional[str] = None
    size: float = 8.0

@dataclass
class Line:
    """Represents a line segment with optional metadata."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    confidence: float = 1.0
    label: Optional[str] = None
    color: Optional[str] = None
    linewidth: float = 2.0
    alpha: float = 0.8
    linestyle: str = '-'

class OverlayManager:
    """
    Manages all overlay elements on the matplotlib axes.
    Provides a clean interface for adding, removing, and updating visual overlays.
    """

    def __init__(self, axes, config: Dict[str, Any] = None):
        """
        Initialize the overlay manager.

        Args:
            axes: Matplotlib axes instance
            config: Configuration dictionary
        """
        self.ax = axes
        self.config = get_visualization_config('default', **(config or {}))

        # Overlay storage
        self.overlays: Dict[str, List[OverlayElement]] = {
            'lines': [],
            'points': [],
            'intersections': [],
            'texts': [],
            'shapes': [],
            'confidence_indicators': [],
            'scale_bar': [],
            'crosshair': [],
            'measurements': [],
            'annotations': [],
        }

        # Visualization configuration
        self.point_config = self.config.get('points', {})
        self.line_config = self.config.get('lines', {})
        self.intersection_config = self.config.get('intersections', {})
        self.annotation_config = self.config.get('annotations', {})

        # Performance settings
        self.performance_config = self.config.get('performance', {})
        self.max_overlays = self.performance_config.get('max_lines_rendered', 1000)

        # State tracking
        self.overlay_counter: int = 0
        self.group_visibility: Dict[str, bool] = {}

    def add_point(self, point: Union[Point, Tuple[float, float]], **kwargs) -> str:
        """
        Add a point overlay.

        Args:
            point: Point object or coordinate tuple
            **kwargs: Override parameters

        Returns:
            Unique overlay ID
        """
        try:
            # Handle different input types
            if isinstance(point, tuple):
                point_obj = Point(x=point[0], y=point[1], **kwargs)
            else:
                point_obj = point

            # Apply defaults from config
            color = kwargs.get('color', point_obj.color or self.point_config.get('default_marker_color', 'lime'))
            size = kwargs.get('size', point_obj.size or self.point_config.get('default_size', 8))
            marker = kwargs.get('marker', self.point_config.get('default_marker', 'o'))
            alpha = kwargs.get('alpha', self.point_config.get('default_alpha', 0.8))

            # Create scatter plot for point
            scatter = self.ax.scatter(point_obj.x, point_obj.y,
                                    c=color, s=size**2,
                                    marker=marker, alpha=alpha,
                                    edgecolors='white', linewidth=1,
                                    zorder=self.point_config.get('z_order', 5))

            # Create overlay element
            overlay_id = f"point_{self.overlay_counter}"
            overlay_element = OverlayElement(
                element_type=OverlayType.POINT,
                matplotlib_object=scatter,
                data={
                    'point': point_obj,
                    'kwargs': kwargs
                },
                z_order=self.point_config.get('z_order', 5)
            )

            self.overlays['points'].append(overlay_element)
            self.overlay_counter += 1

            # Add annotation if provided
            if point_obj.label:
                self.add_text_annotation(point_obj.x, point_obj.y, point_obj.label,
                                       color=color, offset=(5, 5))

            return overlay_id

        except Exception as e:
            logger.error(f"Error adding point: {e}")
            return None

    def add_line(self, line: Union[Line, Dict[str, Any]], **kwargs) -> str:
        """
        Add a line overlay.

        Args:
            line: Line object or dictionary with line data
            **kwargs: Override parameters

        Returns:
            Unique overlay ID
        """
        try:
            # Handle different input types
            if isinstance(line, dict):
                start = line.get('start', [0, 0])
                end = line.get('end', [0, 0])
                line_obj = Line(start=start, end=end, **line)
            else:
                line_obj = line

            # Apply defaults from config
            color = kwargs.get('color', line_obj.color or self.line_config.get('default_color', 'lime'))
            linewidth = kwargs.get('linewidth', line_obj.linewidth or self.line_config.get('default_linewidth', 2.0))
            alpha = kwargs.get('alpha', line_obj.alpha or self.line_config.get('default_alpha', 0.8))
            linestyle = kwargs.get('linestyle', line_obj.linestyle or self.line_config.get('default_linestyle', '-'))

            # Create line plot
            line_plot, = self.ax.plot([line_obj.start[0], line_obj.end[0]],
                                     [line_obj.start[1], line_obj.end[1]],
                                     color=color, linewidth=linewidth,
                                     alpha=alpha, linestyle=linestyle,
                                     zorder=self.line_config.get('z_order', 4))

            # Create overlay element
            overlay_id = f"line_{self.overlay_counter}"
            overlay_element = OverlayElement(
                element_type=OverlayType.LINE,
                matplotlib_object=line_plot,
                data={
                    'line': line_obj,
                    'kwargs': kwargs
                },
                z_order=self.line_config.get('z_order', 4)
            )

            self.overlays['lines'].append(overlay_element)
            self.overlay_counter += 1

            # Add label if provided
            if line_obj.label:
                mid_x = (line_obj.start[0] + line_obj.end[0]) / 2
                mid_y = (line_obj.start[1] + line_obj.end[1]) / 2
                self.add_text_annotation(mid_x, mid_y, line_obj.label,
                                       color=color, offset=(10, -10))

            return overlay_id

        except Exception as e:
            logger.error(f"Error adding line: {e}")
            return None

    def add_intersection(self, point: Union[Point, Tuple[float, float]], confidence: float = 1.0, **kwargs) -> str:
        """
        Add an intersection point with special styling.

        Args:
            point: Intersection point
            confidence: Confidence value (0.0-1.0)
            **kwargs: Override parameters

        Returns:
            Unique overlay ID
        """
        try:
            # Handle different input types
            if isinstance(point, tuple):
                point_obj = Point(x=point[0], y=point[1], confidence=confidence, **kwargs)
            else:
                point_obj = point
                if confidence != 1.0:
                    point_obj.confidence = confidence

            # Apply intersection-specific styling
            config = self.intersection_config
            marker = kwargs.get('marker', config.get('marker', 'x'))
            size = kwargs.get('size', config.get('size', 10))
            linewidth = kwargs.get('linewidth', config.get('linewidth', 2.0))

            # Color based on confidence
            color = self._get_confidence_color(point_obj.confidence)

            # Create intersection marker
            scatter = self.ax.scatter(point_obj.x, point_obj.y,
                                    c=color, s=size**2,
                                    marker=marker, linewidth=linewidth,
                                    edgecolors='white', alpha=1.0,
                                    zorder=config.get('z_order', 6))

            # Add confidence indicator
            self._add_confidence_indicator(point_obj)

            # Create overlay element
            overlay_id = f"intersection_{self.overlay_counter}"
            overlay_element = OverlayElement(
                element_type=OverlayType.INTERSECTION,
                matplotlib_object=scatter,
                data={
                    'point': point_obj,
                    'confidence': confidence,
                    'kwargs': kwargs
                },
                z_order=config.get('z_order', 6)
            )

            self.overlays['intersections'].append(overlay_element)
            self.overlay_counter += 1

            # Add coordinate annotation
            coord_text = f"({int(point_obj.x)}, {int(point_obj.y)})"
            self.add_text_annotation(point_obj.x, point_obj.y, coord_text,
                                   color=color, offset=(5, 5), fontsize=8)

            return overlay_id

        except Exception as e:
            logger.error(f"Error adding intersection: {e}")
            return None

    def add_text_annotation(self, x: float, y: float, text: str, **kwargs) -> str:
        """
        Add a text annotation.

        Args:
            x: X coordinate
            y: Y coordinate
            text: Text content
            **kwargs: Additional text parameters

        Returns:
            Unique overlay ID
        """
        try:
            # Apply annotation configuration
            config = self.annotation_config
            fontsize = kwargs.get('fontsize', config.get('font_size', 8))
            color = kwargs.get('color', config.get('text_color', 'white'))
            offset = kwargs.get('offset', config.get('text_offset', 15))

            # Create annotation
            annotation = self.ax.annotate(text, (x, y),
                                       xytext=offset,
                                       textcoords='offset points',
                                       fontsize=fontsize, color=color,
                                       bbox=dict(boxstyle='round,pad=0.3',
                                               facecolor='black', alpha=0.7),
                                       zorder=config.get('z_order', 7))

            # Create overlay element
            overlay_id = f"text_{self.overlay_counter}"
            overlay_element = OverlayElement(
                element_type=OverlayType.TEXT,
                matplotlib_object=annotation,
                data={
                    'x': x, 'y': y,
                    'text': text,
                    'kwargs': kwargs
                },
                z_order=config.get('z_order', 7)
            )

            self.overlays['texts'].append(overlay_element)
            self.overlay_counter += 1

            return overlay_id

        except Exception as e:
            logger.error(f"Error adding text annotation: {e}")
            return None

    def add_rectangle(self, x: float, y: float, width: float, height: float, **kwargs) -> str:
        """
        Add a rectangle overlay.

        Args:
            x: Left coordinate
            y: Top coordinate
            width: Rectangle width
            height: Rectangle height
            **kwargs: Rectangle parameters

        Returns:
            Unique overlay ID
        """
        try:
            # Create rectangle patch
            rectangle = Rectangle((x, y), width, height, **kwargs)

            self.ax.add_patch(rectangle)

            # Create overlay element
            overlay_id = f"rectangle_{self.overlay_counter}"
            overlay_element = OverlayElement(
                element_type=OverlayType.RECTANGLE,
                matplotlib_object=rectangle,
                data={
                    'x': x, 'y': y,
                    'width': width, 'height': height,
                    'kwargs': kwargs
                }
            )

            self.overlays['shapes'].append(overlay_element)
            self.overlay_counter += 1

            return overlay_id

        except Exception as e:
            logger.error(f"Error adding rectangle: {e}")
            return None

    def add_scale_bar(self, length_pixels: float, position: str = 'lower-left', **kwargs) -> str:
        """
        Add a scale bar to the visualization.

        Args:
            length_pixels: Length of scale bar in pixels
            position: Position ('lower-left', 'lower-right', 'upper-left', 'upper-right')
            **kwargs: Scale bar parameters

        Returns:
            Unique overlay ID
        """
        try:
            # Get axis limits for positioning
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

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

            # Create scale bar line
            line, = self.ax.plot([x, x + length_pixels], [y, y],
                                color='white', linewidth=2, zorder=8)

            # Create scale bar text
            text = f"{length_pixels:.0f} px"
            text_obj = self.ax.text(x + length_pixels/2, y + 5, text,
                                   fontsize=8, color='white',
                                   ha='center', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='black', alpha=0.7),
                                   zorder=9)

            # Group scale bar elements
            scale_bar_group = [line, text_obj]

            # Create overlay element (using the line as the main object)
            overlay_id = f"scale_bar_{self.overlay_counter}"
            overlay_element = OverlayElement(
                element_type=OverlayType.SCALE_BAR,
                matplotlib_object=scale_bar_group,
                data={
                    'length': length_pixels,
                    'position': position,
                    'kwargs': kwargs
                },
                z_order=8
            )

            self.overlays['scale_bar'] = [overlay_element]  # Only one scale bar at a time
            self.overlay_counter += 1

            return overlay_id

        except Exception as e:
            logger.error(f"Error adding scale bar: {e}")
            return None

    def _add_confidence_indicator(self, point: Point):
        """Add a visual confidence indicator for a point."""
        try:
            config = self.intersection_config
            halo_size = config.get('halo_size', 15)
            halo_alpha = config.get('halo_alpha', 0.3)

            # Create confidence halo
            color = self._get_confidence_color(point.confidence)
            halo = Circle((point.x, point.y), halo_size,
                        color=color, alpha=halo_alpha * point.confidence,
                        zorder=config.get('z_order', 5))

            self.ax.add_patch(halo)

            # Store as confidence indicator
            overlay_element = OverlayElement(
                element_type=OverlayType.CONFIDENCE_INDICATOR,
                matplotlib_object=halo,
                data={'point': point, 'confidence': point.confidence},
                z_order=config.get('z_order', 5)
            )

            self.overlays['confidence_indicators'].append(overlay_element)

        except Exception as e:
            logger.error(f"Error adding confidence indicator: {e}")

    def _get_confidence_color(self, confidence: float) -> str:
        """
        Get color based on confidence value.

        Args:
            confidence: Confidence value (0.0-1.0)

        Returns:
            Color string
        """
        # Use quality thresholds from config
        thresholds = self.config.get('quality', {})
        high_conf = thresholds.get('high_confidence', 0.8)
        medium_conf = thresholds.get('medium_confidence', 0.6)
        low_conf = thresholds.get('low_confidence', 0.4)

        if confidence >= high_conf:
            return '#00FF00'  # Green
        elif confidence >= medium_conf:
            return '#FFD700'  # Gold
        elif confidence >= low_conf:
            return '#FFA500'  # Orange
        else:
            return '#FF0000'  # Red

    def remove_overlay(self, overlay_id: str) -> bool:
        """
        Remove a specific overlay by ID.

        Args:
            overlay_id: ID of overlay to remove

        Returns:
            True if overlay was removed, False if not found
        """
        try:
            # Search for overlay in all categories
            for category, overlays in self.overlays.items():
                for i, overlay in enumerate(overlays):
                    if overlay_id in str(overlay.matplotlib_object):
                        # Remove matplotlib object
                        if isinstance(overlay.matplotlib_object, list):
                            for obj in overlay.matplotlib_object:
                                obj.remove()
                        else:
                            overlay.matplotlib_object.remove()

                        # Remove from storage
                        overlays.pop(i)
                        return True

            return False

        except Exception as e:
            logger.error(f"Error removing overlay: {e}")
            return False

    def clear_overlays(self, category: Optional[str] = None):
        """
        Clear overlays from a specific category or all categories.

        Args:
            category: Category to clear, or None for all categories
        """
        try:
            categories = [category] if category else list(self.overlays.keys())

            for cat in categories:
                if cat in self.overlays:
                    for overlay in self.overlays[cat]:
                        try:
                            if isinstance(overlay.matplotlib_object, list):
                                for obj in overlay.matplotlib_object:
                                    obj.remove()
                            else:
                                overlay.matplotlib_object.remove()
                        except:
                            pass  # Object may already be removed

                    self.overlays[cat] = []

        except Exception as e:
            logger.error(f"Error clearing overlays: {e}")

    def set_visibility(self, category: str, visible: bool):
        """
        Set visibility for overlays in a category.

        Args:
            category: Overlay category
            visible: Visibility status
        """
        try:
            if category in self.overlays:
                for overlay in self.overlays[category]:
                    if isinstance(overlay.matplotlib_object, list):
                        for obj in overlay.matplotlib_object:
                            obj.set_visible(visible)
                    else:
                        overlay.matplotlib_object.set_visible(visible)

                self.group_visibility[category] = visible

        except Exception as e:
            logger.error(f"Error setting overlay visibility: {e}")

    def update_overlays(self):
        """Redraw the canvas to show overlay changes."""
        try:
            if hasattr(self.ax.figure, 'canvas'):
                self.ax.figure.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error updating overlays: {e}")

    def get_overlay_info(self) -> Dict[str, int]:
        """
        Get information about current overlays.

        Returns:
            Dictionary with overlay counts by category
        """
        return {
            category: len(overlays)
            for category, overlays in self.overlays.items()
        }

    def get_overlays_by_type(self, overlay_type: OverlayType) -> List[OverlayElement]:
        """
        Get all overlays of a specific type.

        Args:
            overlay_type: Type of overlay to retrieve

        Returns:
            List of overlay elements
        """
        result = []
        for overlays in self.overlays.values():
            result.extend([o for o in overlays if o.element_type == overlay_type])
        return result

    def cleanup_overlays(self):
        """Clean up all overlays and reset state."""
        try:
            # Clear all overlays
            self.clear_overlays()

            # Reset state
            self.overlay_counter = 0
            self.group_visibility = {}

            logger.info("Overlay cleanup completed")

        except Exception as e:
            logger.error(f"Error during overlay cleanup: {e}")

    def export_overlays(self) -> Dict[str, Any]:
        """
        Export overlay data for saving or sharing.

        Returns:
            Dictionary containing overlay data
        """
        export_data = {
            'version': '1.0',
            'categories': {}
        }

        for category, overlays in self.overlays.items():
            export_data['categories'][category] = []
            for overlay in overlays:
                overlay_data = {
                    'type': overlay.element_type.value,
                    'data': overlay.data,
                    'visible': overlay.visible,
                    'z_order': overlay.z_order,
                    'group': overlay.group
                }
                export_data['categories'][category].append(overlay_data)

        return export_data

    def import_overlays(self, overlay_data: Dict[str, Any]) -> bool:
        """
        Import overlay data from exported format.

        Args:
            overlay_data: Overlay data from export_overlays()

        Returns:
            True if import was successful
        """
        try:
            # Clear existing overlays
            self.cleanup_overlays()

            # Import overlays from data
            for category, overlays in overlay_data.get('categories', {}).items():
                for overlay_info in overlays:
                    # Recreate overlay based on type and data
                    # This would need to be implemented based on specific requirements
                    pass

            return True

        except Exception as e:
            logger.error(f"Error importing overlays: {e}")
            return False