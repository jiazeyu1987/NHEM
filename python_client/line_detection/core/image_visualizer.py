"""
Image Visualizer Component
This module handles all image display and visualization functionality for the LineDetectionWidget.
"""

import logging
import base64
import io
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image

from ..config import get_visualization_config, get_widget_config

logger = logging.getLogger(__name__)

class ImageVisualizer:
    """
    Core image visualization component for ROI1 image display.
    Handles image loading, processing, and rendering.
    """

    def __init__(self, figure: Figure, axes: Axes, config: Dict[str, Any] = None):
        """
        Initialize the image visualizer.

        Args:
            figure: Matplotlib figure instance
            axes: Matplotlib axes instance
            config: Configuration dictionary
        """
        self.figure = figure
        self.ax = axes
        self.config = get_visualization_config('default', **(config or {}))
        self.widget_config = get_widget_config(**(config or {}))

        # State tracking
        self.current_image: Optional[np.ndarray] = None
        self.image_shape: Optional[Tuple[int, ...]] = None
        self.image_displayed: bool = False
        self.last_base64_data: Optional[str] = None

        # Display parameters
        self.extent: Optional[Tuple[float, float, float, float]] = None
        self.colormap = self.config.get('image', {}).get('colormap', 'gray')
        self.interpolation = self.config.get('image', {}).get('interpolation', 'bilinear')
        self.auto_contrast = self.config.get('image', {}).get('auto_contrast', True)

        # Initialize display
        self._setup_display()

    def _setup_display(self):
        """Set up initial display parameters."""
        # Configure axes appearance
        self.ax.set_facecolor(self.config.get('colors', {}).get('background', '#2d2d2d'))

        # Grid configuration
        grid_config = self.config.get('grid', {})
        if grid_config.get('show_grid', True):
            self.ax.grid(True,
                        alpha=grid_config.get('grid_alpha', 0.3),
                        linestyle=grid_config.get('grid_linestyle', '-'),
                        linewidth=grid_config.get('grid_linewidth', 0.5))

    def update_image(self, roi_data: str) -> bool:
        """
        Update the displayed image with new ROI data.

        Args:
            roi_data: Base64-encoded image data (with or without data URL prefix)

        Returns:
            True if image was successfully updated, False otherwise
        """
        try:
            if not roi_data or not isinstance(roi_data, str):
                logger.warning("Invalid ROI data received")
                return False

            # Parse base64 image data
            image_bytes = self._parse_base64_data(roi_data)
            if image_bytes is None:
                return False

            # Load and process image
            image = self._load_image(image_bytes)
            if image is None:
                return False

            # Convert to numpy array and store
            self.current_image = np.array(image)
            self.image_shape = self.current_image.shape

            # Store original data for offline mode
            self.last_base64_data = roi_data

            # Display the image
            success = self._display_image()
            if success:
                self.image_displayed = True
                logger.info(f"Image updated: {self.image_shape}")

            return success

        except Exception as e:
            logger.error(f"Error updating image: {e}")
            self.image_displayed = False
            return False

    def _parse_base64_data(self, roi_data: str) -> Optional[bytes]:
        """
        Parse base64 image data with or without data URL prefix.

        Args:
            roi_data: Base64 image data string

        Returns:
            Decoded image bytes or None if parsing failed
        """
        try:
            if roi_data.startswith("data:image/"):
                # Extract base64 part from data URL
                if "," in roi_data:
                    header, encoded = roi_data.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                else:
                    logger.error("Invalid data URL format")
                    return None
            else:
                # Direct base64 data
                image_bytes = base64.b64decode(roi_data)

            return image_bytes

        except Exception as e:
            logger.error(f"Error parsing base64 data: {e}")
            return None

    def _load_image(self, image_bytes: bytes) -> Optional[Image.Image]:
        """
        Load image from bytes using PIL.

        Args:
            image_bytes: Raw image bytes

        Returns:
            PIL Image instance or None if loading failed
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')

            return image

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def _display_image(self) -> bool:
        """
        Display the current image on the axes.

        Returns:
            True if display was successful, False otherwise
        """
        try:
            if self.current_image is None:
                logger.error("No image to display")
                return False

            # Clear previous content
            self.ax.clear()
            self._setup_display()

            # Set display extent
            height, width = self.image_shape[:2]
            self.extent = (0, width, height, 0)

            # Apply contrast adjustments if enabled
            display_image = self._apply_contrast_adjustments(self.current_image)

            # Display image based on color channels
            if len(self.image_shape) == 3:
                # Color image
                self.ax.imshow(display_image,
                             extent=self.extent,
                             interpolation=self.interpolation)
            else:
                # Grayscale image
                self.ax.imshow(display_image,
                             cmap=self.colormap,
                             extent=self.extent,
                             interpolation=self.interpolation)

            # Update axes limits
            self.ax.set_xlim(0, width)
            self.ax.set_ylim(height, 0)

            # Set labels
            self.ax.set_xlabel("X (pixels)", color=self.config.get('colors', {}).get('text', '#FFFFFF'))
            self.ax.set_ylabel("Y (pixels)", color=self.config.get('colors', {}).get('text', '#FFFFFF'))

            return True

        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            return False

    def _apply_contrast_adjustments(self, image: np.ndarray) -> np.ndarray:
        """
        Apply contrast adjustments to the image if enabled.

        Args:
            image: Input image array

        Returns:
            Adjusted image array
        """
        if not self.auto_contrast:
            return image

        try:
            # Get contrast percentiles from config
            contrast_config = self.config.get('image', {})
            low_percentile = contrast_config.get('contrast_percentile_low', 2)
            high_percentile = contrast_config.get('contrast_percentile_high', 98)

            # Calculate percentiles
            if len(image.shape) == 3:
                # Color image - apply contrast per channel
                adjusted_image = np.zeros_like(image, dtype=np.float32)
                for i in range(image.shape[2]):
                    channel = image[:, :, i].astype(np.float32)
                    p_low, p_high = np.percentile(channel, [low_percentile, high_percentile])

                    if p_high > p_low:  # Avoid division by zero
                        channel = np.clip((channel - p_low) / (p_high - p_low), 0, 1)

                    adjusted_image[:, :, i] = channel

                return (adjusted_image * 255).astype(np.uint8)
            else:
                # Grayscale image
                image_float = image.astype(np.float32)
                p_low, p_high = np.percentile(image_float, [low_percentile, high_percentile])

                if p_high > p_low:  # Avoid division by zero
                    image_float = np.clip((image_float - p_low) / (p_high - p_low), 0, 1)

                return (image_float * 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Error applying contrast adjustments: {e}")
            return image

    def clear_display(self):
        """Clear the current image display."""
        try:
            self.ax.clear()
            self._setup_display()
            self.image_displayed = False
            self.current_image = None
            self.image_shape = None
            self.extent = None
        except Exception as e:
            logger.error(f"Error clearing display: {e}")

    def refresh_display(self) -> bool:
        """
        Refresh the current image display.

        Returns:
            True if refresh was successful, False otherwise
        """
        if not self.image_displayed or self.current_image is None:
            return False

        return self._display_image()

    def set_colormap(self, colormap: str):
        """
        Set the colormap for grayscale images.

        Args:
            colormap: Matplotlib colormap name
        """
        self.colormap = colormap
        if self.image_displayed and len(self.image_shape) == 2:
            self.refresh_display()

    def set_interpolation(self, interpolation: str):
        """
        Set the interpolation method for image display.

        Args:
            interpolation: Interpolation method name
        """
        self.interpolation = interpolation
        if self.image_displayed:
            self.refresh_display()

    def get_image_info(self) -> Dict[str, Any]:
        """
        Get information about the current image.

        Returns:
            Dictionary containing image information
        """
        info = {
            'image_displayed': self.image_displayed,
            'image_shape': self.image_shape,
            'extent': self.extent,
        }

        if self.image_shape:
            info.update({
                'width': self.image_shape[1] if len(self.image_shape) >= 2 else 0,
                'height': self.image_shape[0] if len(self.image_shape) >= 1 else 0,
                'channels': self.image_shape[2] if len(self.image_shape) >= 3 else 1,
            })

        return info

    def pixel_to_data_coords(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to data coordinates.

        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate

        Returns:
            Tuple of (data_x, data_y)
        """
        if self.extent is None:
            return float(pixel_x), float(pixel_y)

        x_min, x_max, y_max, y_min = self.extent

        # Calculate data coordinates
        data_x = x_min + (pixel_x / (self.ax.get_window_extent().width)) * (x_max - x_min)
        data_y = y_min + ((self.ax.get_window_extent().height - pixel_y) /
                         self.ax.get_window_extent().height) * (y_max - y_min)

        return data_x, data_y

    def data_to_pixel_coords(self, data_x: float, data_y: float) -> Tuple[int, int]:
        """
        Convert data coordinates to pixel coordinates.

        Args:
            data_x: X data coordinate
            data_y: Y data coordinate

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        if self.extent is None:
            return int(data_x), int(data_y)

        x_min, x_max, y_max, y_min = self.extent

        # Calculate pixel coordinates
        pixel_x = int(((data_x - x_min) / (x_max - x_min)) * self.ax.get_window_extent().width)
        pixel_y = int(((y_max - data_y) / (y_max - y_min)) * self.ax.get_window_extent().height)

        return pixel_x, pixel_y

    def zoom_to_region(self, x_min: float, y_min: float, x_max: float, y_max: float):
        """
        Zoom to a specific region of the image.

        Args:
            x_min: Minimum X coordinate
            y_min: Minimum Y coordinate
            x_max: Maximum X coordinate
            y_max: Maximum Y coordinate
        """
        try:
            # Validate coordinates
            if x_min >= x_max or y_min >= y_max:
                logger.warning("Invalid zoom region coordinates")
                return

            # Apply zoom
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_max, y_min)  # Note: Y axis is inverted

        except Exception as e:
            logger.error(f"Error zooming to region: {e}")

    def reset_zoom(self):
        """Reset zoom to show the entire image."""
        if self.extent is not None:
            self.ax.set_xlim(self.extent[0], self.extent[1])
            self.ax.set_ylim(self.extent[2], self.extent[3])

    def get_displayed_image(self) -> Optional[np.ndarray]:
        """
        Get the currently displayed image array.

        Returns:
            Current image array or None if no image is displayed
        """
        return self.current_image.copy() if self.current_image is not None else None

    def save_current_view(self, filename: str, dpi: int = None) -> bool:
        """
        Save the current view to a file.

        Args:
            filename: Output filename
            dpi: Resolution for saving (uses config default if None)

        Returns:
            True if save was successful, False otherwise
        """
        try:
            if dpi is None:
                dpi = self.widget_config.get('save_dpi', 150)

            self.figure.savefig(filename,
                              dpi=dpi,
                              bbox_inches='tight',
                              facecolor=self.config.get('colors', {}).get('background', '#1e1e1e'))

            logger.info(f"View saved to: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving view: {e}")
            return False