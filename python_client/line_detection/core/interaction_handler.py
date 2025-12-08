"""
Interaction Handler Component
This module handles all user interaction events including mouse, keyboard, and touch events.
"""

import logging
from typing import Optional, Tuple, Callable, Dict, Any, List
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..config import get_visualization_config

logger = logging.getLogger(__name__)

class InteractionMode(Enum):
    """Interaction modes for the widget."""
    NAVIGATION = "navigation"      # Pan and zoom
    MEASUREMENT = "measurement"    # Distance/area measurement
    ANNOTATION = "annotation"      # Add annotations
    SELECTION = "selection"       # Select regions/objects
    DRAWING = "drawing"           # Draw lines/shapes

class MouseEvent:
    """Encapsulates mouse event data."""
    def __init__(self, event):
        self.x = event.x
        self.y = event.y
        self.xdata = event.xdata
        self.ydata = event.ydata
        self.button = event.button
        self.key = event.key
        self.step = getattr(event, 'step', 0)
        self.inaxes = event.inaxes
        self.guiEvent = getattr(event, 'guiEvent', None)

class KeyboardEvent:
    """Encapsulates keyboard event data."""
    def __init__(self, event):
        self.key = event.key
        self.x = event.x
        self.y = event.y
        self.xdata = event.xdata
        self.ydata = event.ydata
        self.inaxes = event.inaxes

class InteractionHandler:
    """
    Handles all user interaction events for the LineDetectionWidget.
    Provides a clean interface for mouse, keyboard, and custom interactions.
    """

    def __init__(self, figure: Figure, canvas: FigureCanvasTkAgg, axes, config: Dict[str, Any] = None):
        """
        Initialize the interaction handler.

        Args:
            figure: Matplotlib figure instance
            canvas: Figure canvas instance
            axes: Matplotlib axes instance
            config: Configuration dictionary
        """
        self.figure = figure
        self.canvas = canvas
        self.ax = axes
        self.config = get_visualization_config('default', **(config or {}))

        # Interaction state
        self.mode = InteractionMode.NAVIGATION
        self.mouse_pressed: bool = False
        self.last_mouse_pos: Optional[Tuple[float, float]] = None
        self.drag_start_pos: Optional[Tuple[float, float]] = None
        self.keyboard_modifiers: Dict[str, bool] = {}

        # Zoom and pan state
        self.zoom_level: float = 1.0
        self.pan_start: Optional[Tuple[float, float]] = None
        self.original_xlim: Optional[Tuple[float, float]] = None
        self.original_ylim: Optional[Tuple[float, float]] = None

        # Crosshair state
        self.crosshair_enabled: bool = False
        self.crosshair_line_h = None
        self.crosshair_line_v = None

        # Measurement state
        self.measurement_points: List[Tuple[float, float]] = []
        self.measurement_lines: List = []
        self.measurement_text = None

        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'mouse_move': [],
            'mouse_press': [],
            'mouse_release': [],
            'mouse_scroll': [],
            'key_press': [],
            'key_release': [],
            'coordinate_update': [],
            'zoom_change': [],
            'pan_change': [],
            'measurement_complete': [],
            'annotation_added': [],
        }

        # Interaction configuration
        self.interaction_config = self.config.get('interaction', {})
        self.zoom_in_factor = self.interaction_config.get('zoom_in_factor', 1.1)
        self.zoom_out_factor = self.interaction_config.get('zoom_out_factor', 0.9)
        self.min_zoom_level = self.interaction_config.get('min_zoom_level', 10)
        self.max_zoom_level = self.interaction_config.get('max_zoom_level', 1000)

        # Setup event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup matplotlib event handlers."""
        try:
            # Mouse events
            self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
            self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
            self.canvas.mpl_connect('scroll_event', self._on_mouse_scroll)

            # Keyboard events
            self.canvas.mpl_connect('key_press_event', self._on_key_press)
            self.canvas.mpl_connect('key_release_event', self._on_key_release)

            # Window events
            self.canvas.mpl_connect('close_event', self._on_close)

            logger.debug("Event handlers setup completed")

        except Exception as e:
            logger.error(f"Error setting up event handlers: {e}")

    def set_mode(self, mode: InteractionMode):
        """
        Set the interaction mode.

        Args:
            mode: New interaction mode
        """
        if self.mode != mode:
            self.mode = mode
            self._cleanup_mode_specific_state()
            logger.info(f"Interaction mode changed to: {mode.value}")

    def _cleanup_mode_specific_state(self):
        """Clean up mode-specific state when changing modes."""
        if self.mode != InteractionMode.MEASUREMENT:
            self._clear_measurements()

        if self.mode != InteractionMode.DRAWING:
            # Clear any temporary drawing elements
            pass

    def get_mouse_coordinates(self, event) -> Optional[Tuple[float, float]]:
        """
        Get mouse coordinates in data space.

        Args:
            event: Matplotlib mouse event

        Returns:
            Tuple of (x, y) coordinates or None if outside axes
        """
        try:
            if event.inaxes != self.ax:
                return None

            x, y = event.xdata, event.ydata

            if x is not None and y is not None:
                return (float(x), float(y))

            return None

        except Exception as e:
            logger.error(f"Error getting mouse coordinates: {e}")
            return None

    def _on_mouse_move(self, event):
        """Handle mouse move events."""
        try:
            coords = self.get_mouse_coordinates(event)

            # Update coordinate display
            if coords:
                self._trigger_callbacks('coordinate_update', coords)
                self._update_crosshair(coords)

                # Handle dragging
                if self.mouse_pressed and self.drag_start_pos:
                    self._handle_drag(coords)

                # Handle mode-specific behavior
                if self.mode == InteractionMode.NAVIGATION:
                    self._handle_navigation_drag(coords)
                elif self.mode == InteractionMode.MEASUREMENT:
                    self._handle_measurement_preview(coords)
                elif self.mode == InteractionMode.DRAWING:
                    self._handle_drawing_preview(coords)

            # Trigger callbacks
            mouse_event = MouseEvent(event)
            self._trigger_callbacks('mouse_move', mouse_event, coords)

        except Exception as e:
            logger.error(f"Error handling mouse move: {e}")

    def _on_mouse_press(self, event):
        """Handle mouse press events."""
        try:
            if event.inaxes == self.ax:
                self.mouse_pressed = True
                coords = self.get_mouse_coordinates(event)

                if coords:
                    self.last_mouse_pos = coords
                    self.drag_start_pos = coords

                    # Handle mode-specific behavior
                    if self.mode == InteractionMode.NAVIGATION:
                        self._start_pan(coords)
                    elif self.mode == InteractionMode.MEASUREMENT:
                        self._add_measurement_point(coords)
                    elif self.mode == InteractionMode.ANNOTATION:
                        self._add_annotation(coords)
                    elif self.mode == InteractionMode.SELECTION:
                        self._start_selection(coords)
                    elif self.mode == InteractionMode.DRAWING:
                        self._start_drawing(coords)

                # Trigger callbacks
                mouse_event = MouseEvent(event)
                self._trigger_callbacks('mouse_press', mouse_event, coords)

        except Exception as e:
            logger.error(f"Error handling mouse press: {e}")

    def _on_mouse_release(self, event):
        """Handle mouse release events."""
        try:
            self.mouse_pressed = False

            # Handle mode-specific behavior
            if self.mode == InteractionMode.NAVIGATION:
                self._end_pan()
            elif self.mode == InteractionMode.SELECTION:
                self._end_selection()
            elif self.mode == InteractionMode.DRAWING:
                self._end_drawing()

            # Reset state
            self.last_mouse_pos = None
            self.drag_start_pos = None

            # Trigger callbacks
            mouse_event = MouseEvent(event)
            coords = self.get_mouse_coordinates(event)
            self._trigger_callbacks('mouse_release', mouse_event, coords)

        except Exception as e:
            logger.error(f"Error handling mouse release: {e}")

    def _on_mouse_scroll(self, event):
        """Handle mouse scroll events for zooming."""
        try:
            if event.inaxes == self.ax:
                coords = self.get_mouse_coordinates(event)
                if coords:
                    self._zoom_at_position(coords, event.button == 'up')

                # Trigger callbacks
                mouse_event = MouseEvent(event)
                self._trigger_callbacks('mouse_scroll', mouse_event, coords)

        except Exception as e:
            logger.error(f"Error handling mouse scroll: {e}")

    def _on_key_press(self, event):
        """Handle key press events."""
        try:
            key = event.key
            self.keyboard_modifiers[key] = True

            # Handle mode-specific keyboard shortcuts
            if key == 'r':
                self.reset_view()
            elif key == 'c':
                self.clear_measurements()
            elif key == 'h':
                self.toggle_crosshair()
            elif key == 'm':
                self.set_mode(InteractionMode.MEASUREMENT)
            elif key == 'n':
                self.set_mode(InteractionMode.NAVIGATION)
            elif key == 'a':
                self.set_mode(InteractionMode.ANNOTATION)

            # Trigger callbacks
            keyboard_event = KeyboardEvent(event)
            coords = self.get_mouse_coordinates(event)
            self._trigger_callbacks('key_press', keyboard_event, coords)

        except Exception as e:
            logger.error(f"Error handling key press: {e}")

    def _on_key_release(self, event):
        """Handle key release events."""
        try:
            key = event.key
            self.keyboard_modifiers[key] = False

            # Trigger callbacks
            keyboard_event = KeyboardEvent(event)
            coords = self.get_mouse_coordinates(event)
            self._trigger_callbacks('key_release', keyboard_event, coords)

        except Exception as e:
            logger.error(f"Error handling key release: {e}")

    def _on_close(self, event):
        """Handle window close event."""
        try:
            self._cleanup()
        except Exception as e:
            logger.error(f"Error handling close event: {e}")

    def _start_pan(self, coords: Tuple[float, float]):
        """Start panning operation."""
        self.pan_start = coords
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

    def _handle_navigation_drag(self, coords: Tuple[float, float]):
        """Handle dragging for navigation (panning)."""
        if self.pan_start and self.original_xlim and self.original_ylim:
            dx = coords[0] - self.pan_start[0]
            dy = coords[1] - self.pan_start[1]

            new_xlim = (self.original_xlim[0] - dx, self.original_xlim[1] - dx)
            new_ylim = (self.original_ylim[0] - dy, self.original_ylim[1] - dy)

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self._trigger_callbacks('pan_change', coords)

            # Trigger redraw
            self.canvas.draw_idle()

    def _end_pan(self):
        """End panning operation."""
        self.pan_start = None
        self.original_xlim = None
        self.original_ylim = None

    def _zoom_at_position(self, coords: Tuple[float, float], zoom_in: bool):
        """
        Zoom at a specific position.

        Args:
            coords: Coordinates to zoom at
            zoom_in: True to zoom in, False to zoom out
        """
        try:
            # Get current limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            # Calculate zoom factor
            scale_factor = self.zoom_in_factor if zoom_in else self.zoom_out_factor

            # Calculate new ranges
            x_range = (xlim[1] - xlim[0]) * scale_factor / 2
            y_range = (ylim[1] - ylim[0]) * scale_factor / 2

            # Calculate new limits centered on mouse position
            new_xlim = [coords[0] - x_range, coords[0] + x_range]
            new_ylim = [coords[1] - y_range, coords[1] + y_range]

            # Apply zoom limits
            current_zoom = 100 / (xlim[1] - xlim[0])
            new_zoom = current_zoom * (1 / scale_factor)

            if self.min_zoom_level <= new_zoom <= self.max_zoom_level:
                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)
                self.zoom_level = new_zoom

                # Trigger redraw and callbacks
                self.canvas.draw_idle()
                self._trigger_callbacks('zoom_change', self.zoom_level)

        except Exception as e:
            logger.error(f"Error zooming at position: {e}")

    def _add_measurement_point(self, coords: Tuple[float, float]):
        """Add a measurement point."""
        self.measurement_points.append(coords)

        # Draw measurement point
        point, = self.ax.plot(coords[0], coords[1], 'ro', markersize=8)
        self.measurement_lines.append(point)

        # Draw line to previous point
        if len(self.measurement_points) > 1:
            prev_point = self.measurement_points[-2]
            line, = self.ax.plot([prev_point[0], coords[0]], [prev_point[1], coords[1]], 'r-', linewidth=2)
            self.measurement_lines.append(line)

        # Calculate and display distance
        if len(self.measurement_points) > 1:
            self._calculate_measurement()

        self.canvas.draw_idle()

    def _calculate_measurement(self):
        """Calculate and display measurement results."""
        try:
            if len(self.measurement_points) < 2:
                return

            total_distance = 0
            for i in range(1, len(self.measurement_points)):
                p1 = self.measurement_points[i-1]
                p2 = self.measurement_points[i]
                distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                total_distance += distance

            # Display measurement text
            if self.measurement_text:
                self.measurement_text.remove()

            last_point = self.measurement_points[-1]
            text = f"{total_distance:.1f} px"
            self.measurement_text = self.ax.text(last_point[0] + 10, last_point[1] + 10, text,
                                                fontsize=10, color='white',
                                                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

            self.canvas.draw_idle()

            # Trigger callback
            self._trigger_callbacks('measurement_complete', total_distance)

        except Exception as e:
            logger.error(f"Error calculating measurement: {e}")

    def _handle_measurement_preview(self, coords: Tuple[float, float]):
        """Handle measurement preview while moving mouse."""
        # Implementation for measurement preview
        pass

    def _clear_measurements(self):
        """Clear all measurement points and lines."""
        try:
            for line in self.measurement_lines:
                try:
                    line.remove()
                except:
                    pass

            if self.measurement_text:
                try:
                    self.measurement_text.remove()
                except:
                    pass

            self.measurement_points = []
            self.measurement_lines = []
            self.measurement_text = None

            self.canvas.draw_idle()

        except Exception as e:
            logger.error(f"Error clearing measurements: {e}")

    def _update_crosshair(self, coords: Tuple[float, float]):
        """Update crosshair display."""
        if not self.crosshair_enabled:
            return

        try:
            # Remove old crosshair
            if self.crosshair_line_h:
                self.crosshair_line_h.remove()
            if self.crosshair_line_v:
                self.crosshair_line_v.remove()

            # Get current limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            # Draw new crosshair
            self.crosshair_line_h = self.ax.axhline(y=coords[1], color='yellow', linestyle='--', alpha=0.5)
            self.crosshair_line_v = self.ax.axvline(x=coords[0], color='yellow', linestyle='--', alpha=0.5)

            self.canvas.draw_idle()

        except Exception as e:
            logger.error(f"Error updating crosshair: {e}")

    def toggle_crosshair(self):
        """Toggle crosshair display."""
        self.crosshair_enabled = not self.crosshair_enabled

        if not self.crosshair_enabled:
            # Remove crosshair
            if self.crosshair_line_h:
                self.crosshair_line_h.remove()
                self.crosshair_line_h = None
            if self.crosshair_line_v:
                self.crosshair_line_v.remove()
                self.crosshair_line_v = None
            self.canvas.draw_idle()

    def reset_view(self):
        """Reset view to show entire data."""
        try:
            self.ax.relim()
            self.ax.autoscale()
            self.zoom_level = 1.0
            self.canvas.draw_idle()
            self._trigger_callbacks('zoom_change', self.zoom_level)
        except Exception as e:
            logger.error(f"Error resetting view: {e}")

    def add_callback(self, event_type: str, callback: Callable):
        """
        Add a callback for a specific event type.

        Args:
            event_type: Type of event ('mouse_move', 'key_press', etc.)
            callback: Callback function to call
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    def remove_callback(self, event_type: str, callback: Callable):
        """
        Remove a callback for a specific event type.

        Args:
            event_type: Type of event
            callback: Callback function to remove
        """
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)

    def _trigger_callbacks(self, event_type: str, *args, **kwargs):
        """Trigger all callbacks for a specific event type."""
        try:
            for callback in self.callbacks.get(event_type, []):
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")
        except Exception as e:
            logger.error(f"Error triggering callbacks for {event_type}: {e}")

    def get_interaction_state(self) -> Dict[str, Any]:
        """
        Get current interaction state.

        Returns:
            Dictionary containing interaction state information
        """
        return {
            'mode': self.mode.value,
            'mouse_pressed': self.mouse_pressed,
            'last_mouse_pos': self.last_mouse_pos,
            'zoom_level': self.zoom_level,
            'crosshair_enabled': self.crosshair_enabled,
            'measurement_points': len(self.measurement_points),
            'keyboard_modifiers': self.keyboard_modifiers.copy(),
        }

    def _handle_drag(self, coords: Tuple[float, float]):
        """Handle generic dragging behavior."""
        # Override in subclasses or extend for custom drag behavior
        pass

    def _add_annotation(self, coords: Tuple[float, float]):
        """Add annotation at coordinates."""
        # Implementation for adding annotations
        pass

    def _start_selection(self, coords: Tuple[float, float]):
        """Start selection at coordinates."""
        # Implementation for selection
        pass

    def _end_selection(self):
        """End selection operation."""
        # Implementation for ending selection
        pass

    def _start_drawing(self, coords: Tuple[float, float]):
        """Start drawing at coordinates."""
        # Implementation for drawing
        pass

    def _handle_drawing_preview(self, coords: Tuple[float, float]):
        """Handle drawing preview."""
        # Implementation for drawing preview
        pass

    def _end_drawing(self):
        """End drawing operation."""
        # Implementation for ending drawing
        pass

    def _cleanup(self):
        """Clean up resources."""
        try:
            self._clear_measurements()
            if self.crosshair_line_h:
                self.crosshair_line_h.remove()
            if self.crosshair_line_v:
                self.crosshair_line_v.remove()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")