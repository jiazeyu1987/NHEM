"""
Coordinate System Component
This module handles coordinate transformations and spatial operations.
"""

import logging
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from ..config import get_visualization_config

logger = logging.getLogger(__name__)

class CoordinateSystem(Enum):
    """Supported coordinate systems."""
    PIXEL = "pixel"           # Image pixel coordinates
    DATA = "data"            # Matplotlib data coordinates
    DISPLAY = "display"      # Screen/display coordinates
    WORLD = "world"          # Real-world coordinates (e.g., mm, cm)
    NORMALIZED = "normalized" # Normalized coordinates (0.0-1.0)

@dataclass
class Point2D:
    """Represents a 2D point in different coordinate systems."""
    x: float
    y: float
    coordinate_system: CoordinateSystem = CoordinateSystem.PIXEL

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple format."""
        return (self.x, self.y)

@dataclass
class BoundingBox:
    """Represents a rectangular bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    coordinate_system: CoordinateSystem = CoordinateSystem.PIXEL

    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.y_max - self.y_min

    @property
    def center(self) -> Point2D:
        """Get center point of bounding box."""
        return Point2D(
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            self.coordinate_system
        )

    def contains(self, point: Point2D) -> bool:
        """Check if point is inside bounding box."""
        return (self.x_min <= point.x <= self.x_max and
                self.y_min <= point.y <= self.y_max)

class CoordinateTransformer:
    """
    Handles coordinate transformations between different coordinate systems.
    """

    def __init__(self, axes=None, config: Dict[str, Any] = None):
        """
        Initialize the coordinate transformer.

        Args:
            axes: Matplotlib axes instance
            config: Configuration dictionary
        """
        self.ax = axes
        self.config = get_visualization_config('default', **(config or {}))

        # Image properties
        self.image_shape: Optional[Tuple[int, ...]] = None
        self.pixel_extent: Optional[Tuple[float, float, float, float]] = None  # (x0, x1, y0, y1)

        # Real-world coordinate calibration
        self.world_calibration: Optional[Dict[str, Any]] = None
        self.world_units: str = "pixels"

        # Transformation cache
        self._transform_cache: Dict[str, Any] = {}

    def set_image_properties(self, image_shape: Tuple[int, ...], extent: Optional[Tuple[float, float, float, float]] = None):
        """
        Set image properties for coordinate transformations.

        Args:
            image_shape: Shape of the image (height, width, channels)
            extent: Image extent in data coordinates (x0, x1, y0, y1)
        """
        self.image_shape = image_shape

        if extent:
            self.pixel_extent = extent
        elif image_shape and len(image_shape) >= 2:
            # Default extent: pixel coordinates
            self.pixel_extent = (0, image_shape[1], image_shape[0], 0)

        # Clear cache when image properties change
        self._transform_cache.clear()

    def set_world_calibration(self, pixels_per_unit: float, units: str = "mm"):
        """
        Set calibration for real-world coordinate transformations.

        Args:
            pixels_per_unit: Number of pixels per real-world unit
            units: Unit name (mm, cm, etc.)
        """
        self.world_calibration = {
            'pixels_per_unit': pixels_per_unit,
            'units': units,
            'scale_factor': 1.0 / pixels_per_unit
        }
        self.world_units = units

    def pixel_to_data(self, pixel_point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Convert pixel coordinates to data coordinates.

        Args:
            pixel_point: Point in pixel coordinates

        Returns:
            Point in data coordinates
        """
        try:
            if isinstance(pixel_point, tuple):
                pixel_point = Point2D(pixel_point[0], pixel_point[1], CoordinateSystem.PIXEL)

            if not self.pixel_extent:
                # No extent information, return same coordinates
                return Point2D(pixel_point.x, pixel_point.y, CoordinateSystem.DATA)

            # Convert pixel to data coordinates
            x0, x1, y1, y0 = self.pixel_extent  # Note: extent is (x0, x1, y0, y1)

            data_x = x0 + (pixel_point.x / (self.image_shape[1] if self.image_shape else 1)) * (x1 - x0)
            data_y = y0 + (pixel_point.y / (self.image_shape[0] if self.image_shape else 1)) * (y1 - y0)

            return Point2D(data_x, data_y, CoordinateSystem.DATA)

        except Exception as e:
            logger.error(f"Error converting pixel to data coordinates: {e}")
            return pixel_point

    def data_to_pixel(self, data_point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Convert data coordinates to pixel coordinates.

        Args:
            data_point: Point in data coordinates

        Returns:
            Point in pixel coordinates
        """
        try:
            if isinstance(data_point, tuple):
                data_point = Point2D(data_point[0], data_point[1], CoordinateSystem.DATA)

            if not self.pixel_extent:
                # No extent information, return same coordinates
                return Point2D(data_point.x, data_point.y, CoordinateSystem.PIXEL)

            # Convert data to pixel coordinates
            x0, x1, y1, y0 = self.pixel_extent

            pixel_x = ((data_point.x - x0) / (x1 - x0)) * (self.image_shape[1] if self.image_shape else 1)
            pixel_y = ((data_point.y - y0) / (y1 - y0)) * (self.image_shape[0] if self.image_shape else 1)

            return Point2D(pixel_x, pixel_y, CoordinateSystem.PIXEL)

        except Exception as e:
            logger.error(f"Error converting data to pixel coordinates: {e}")
            return data_point

    def pixel_to_world(self, pixel_point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Convert pixel coordinates to real-world coordinates.

        Args:
            pixel_point: Point in pixel coordinates

        Returns:
            Point in world coordinates
        """
        try:
            if isinstance(pixel_point, tuple):
                pixel_point = Point2D(pixel_point[0], pixel_point[1], CoordinateSystem.PIXEL)

            if not self.world_calibration:
                # No calibration, return pixel coordinates
                return Point2D(pixel_point.x, pixel_point.y, CoordinateSystem.WORLD)

            scale_factor = self.world_calibration['scale_factor']
            world_x = pixel_point.x * scale_factor
            world_y = pixel_point.y * scale_factor

            return Point2D(world_x, world_y, CoordinateSystem.WORLD)

        except Exception as e:
            logger.error(f"Error converting pixel to world coordinates: {e}")
            return pixel_point

    def world_to_pixel(self, world_point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Convert real-world coordinates to pixel coordinates.

        Args:
            world_point: Point in world coordinates

        Returns:
            Point in pixel coordinates
        """
        try:
            if isinstance(world_point, tuple):
                world_point = Point2D(world_point[0], world_point[1], CoordinateSystem.WORLD)

            if not self.world_calibration:
                # No calibration, return world coordinates as pixels
                return Point2D(world_point.x, world_point.y, CoordinateSystem.PIXEL)

            pixels_per_unit = self.world_calibration['pixels_per_unit']
            pixel_x = world_point.x * pixels_per_unit
            pixel_y = world_point.y * pixels_per_unit

            return Point2D(pixel_x, pixel_y, CoordinateSystem.PIXEL)

        except Exception as e:
            logger.error(f"Error converting world to pixel coordinates: {e}")
            return world_point

    def data_to_world(self, data_point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Convert data coordinates to world coordinates.

        Args:
            data_point: Point in data coordinates

        Returns:
            Point in world coordinates
        """
        # Convert data to pixel, then pixel to world
        pixel_point = self.data_to_pixel(data_point)
        return self.pixel_to_world(pixel_point)

    def world_to_data(self, world_point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Convert world coordinates to data coordinates.

        Args:
            world_point: Point in world coordinates

        Returns:
            Point in data coordinates
        """
        # Convert world to pixel, then pixel to data
        pixel_point = self.world_to_pixel(world_point)
        return self.pixel_to_data(pixel_point)

    def display_to_data(self, display_point: Union[Point2D, Tuple[float, float]]) -> Optional[Point2D]:
        """
        Convert display coordinates to data coordinates.

        Args:
            display_point: Point in display coordinates

        Returns:
            Point in data coordinates or None if conversion fails
        """
        try:
            if not self.ax:
                return None

            if isinstance(display_point, tuple):
                display_point = Point2D(display_point[0], display_point[1], CoordinateSystem.DISPLAY)

            # Use matplotlib's transformation system
            display_coords = self.ax.transData.inverted().transform([display_point.x, display_point.y])
            return Point2D(display_coords[0], display_coords[1], CoordinateSystem.DATA)

        except Exception as e:
            logger.error(f"Error converting display to data coordinates: {e}")
            return None

    def data_to_display(self, data_point: Union[Point2D, Tuple[float, float]]) -> Optional[Point2D]:
        """
        Convert data coordinates to display coordinates.

        Args:
            data_point: Point in data coordinates

        Returns:
            Point in display coordinates or None if conversion fails
        """
        try:
            if not self.ax:
                return None

            if isinstance(data_point, tuple):
                data_point = Point2D(data_point[0], data_point[1], CoordinateSystem.DATA)

            # Use matplotlib's transformation system
            display_coords = self.ax.transData.transform([data_point.x, data_point.y])
            return Point2D(display_coords[0], display_coords[1], CoordinateSystem.DISPLAY)

        except Exception as e:
            logger.error(f"Error converting data to display coordinates: {e}")
            return None

    def normalize_coordinates(self, point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Normalize coordinates to 0.0-1.0 range.

        Args:
            point: Point to normalize

        Returns:
            Normalized point
        """
        try:
            if isinstance(point, tuple):
                point = Point2D(point[0], point[1])

            # Convert to data coordinates first
            if point.coordinate_system != CoordinateSystem.DATA:
                if point.coordinate_system == CoordinateSystem.PIXEL:
                    point = self.pixel_to_data(point)
                elif point.coordinate_system == CoordinateSystem.WORLD:
                    point = self.world_to_data(point)

            # Get data limits
            if self.ax:
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
            elif self.pixel_extent:
                xlim = (self.pixel_extent[0], self.pixel_extent[1])
                ylim = (self.pixel_extent[3], self.pixel_extent[2])  # y0, y1
            else:
                return point  # Cannot normalize without limits

            # Normalize to 0.0-1.0
            norm_x = (point.x - xlim[0]) / (xlim[1] - xlim[0])
            norm_y = (point.y - ylim[0]) / (ylim[1] - ylim[0])

            # Clamp to 0.0-1.0 range
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            return Point2D(norm_x, norm_y, CoordinateSystem.NORMALIZED)

        except Exception as e:
            logger.error(f"Error normalizing coordinates: {e}")
            return point

    def denormalize_coordinates(self, norm_point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Convert normalized coordinates back to data coordinates.

        Args:
            norm_point: Normalized point (0.0-1.0 range)

        Returns:
            Point in data coordinates
        """
        try:
            if isinstance(norm_point, tuple):
                norm_point = Point2D(norm_point[0], norm_point[1], CoordinateSystem.NORMALIZED)

            # Get data limits
            if self.ax:
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
            elif self.pixel_extent:
                xlim = (self.pixel_extent[0], self.pixel_extent[1])
                ylim = (self.pixel_extent[3], self.pixel_extent[2])
            else:
                return norm_point  # Cannot denormalize without limits

            # Convert back to data coordinates
            data_x = xlim[0] + norm_point.x * (xlim[1] - xlim[0])
            data_y = ylim[0] + norm_point.y * (ylim[1] - ylim[0])

            return Point2D(data_x, data_y, CoordinateSystem.DATA)

        except Exception as e:
            logger.error(f"Error denormalizing coordinates: {e}")
            return norm_point

    def transform_bounding_box(self, bbox: BoundingBox, target_system: CoordinateSystem) -> BoundingBox:
        """
        Transform a bounding box to a different coordinate system.

        Args:
            bbox: Input bounding box
            target_system: Target coordinate system

        Returns:
            Transformed bounding box
        """
        try:
            # Transform corners
            corners = [
                Point2D(bbox.x_min, bbox.y_min, bbox.coordinate_system),
                Point2D(bbox.x_max, bbox.y_max, bbox.coordinate_system)
            ]

            transformed_corners = []
            for corner in corners:
                if target_system == CoordinateSystem.PIXEL:
                    if bbox.coordinate_system == CoordinateSystem.DATA:
                        transformed = self.data_to_pixel(corner)
                    elif bbox.coordinate_system == CoordinateSystem.WORLD:
                        transformed = self.world_to_pixel(corner)
                    else:
                        transformed = corner
                elif target_system == CoordinateSystem.DATA:
                    if bbox.coordinate_system == CoordinateSystem.PIXEL:
                        transformed = self.pixel_to_data(corner)
                    elif bbox.coordinate_system == CoordinateSystem.WORLD:
                        transformed = self.world_to_data(corner)
                    else:
                        transformed = corner
                elif target_system == CoordinateSystem.WORLD:
                    if bbox.coordinate_system == CoordinateSystem.PIXEL:
                        transformed = self.pixel_to_world(corner)
                    elif bbox.coordinate_system == CoordinateSystem.DATA:
                        transformed = self.data_to_world(corner)
                    else:
                        transformed = corner
                else:
                    transformed = corner

                transformed_corners.append(transformed)

            # Create new bounding box
            return BoundingBox(
                x_min=transformed_corners[0].x,
                y_min=transformed_corners[0].y,
                x_max=transformed_corners[1].x,
                y_max=transformed_corners[1].y,
                coordinate_system=target_system
            )

        except Exception as e:
            logger.error(f"Error transforming bounding box: {e}")
            return bbox

    def calculate_distance(self, point1: Union[Point2D, Tuple[float, float]],
                          point2: Union[Point2D, Tuple[float, float]],
                          coordinate_system: CoordinateSystem = CoordinateSystem.PIXEL) -> float:
        """
        Calculate distance between two points.

        Args:
            point1: First point
            point2: Second point
            coordinate_system: Coordinate system for distance calculation

        Returns:
            Distance between points
        """
        try:
            # Convert to Point2D objects
            if isinstance(point1, tuple):
                point1 = Point2D(point1[0], point1[1], coordinate_system)
            if isinstance(point2, tuple):
                point2 = Point2D(point2[0], point2[1], coordinate_system)

            # Transform to target coordinate system
            if point1.coordinate_system != coordinate_system:
                if coordinate_system == CoordinateSystem.PIXEL:
                    point1 = self.data_to_pixel(point1) if point1.coordinate_system == CoordinateSystem.DATA else point1
                elif coordinate_system == CoordinateSystem.DATA:
                    point1 = self.pixel_to_data(point1) if point1.coordinate_system == CoordinateSystem.PIXEL else point1

            if point2.coordinate_system != coordinate_system:
                if coordinate_system == CoordinateSystem.PIXEL:
                    point2 = self.data_to_pixel(point2) if point2.coordinate_system == CoordinateSystem.DATA else point2
                elif coordinate_system == CoordinateSystem.DATA:
                    point2 = self.pixel_to_data(point2) if point2.coordinate_system == CoordinateSystem.PIXEL else point2

            # Calculate Euclidean distance
            dx = point2.x - point1.x
            dy = point2.y - point1.y
            distance = (dx**2 + dy**2)**0.5

            # Convert to world units if available
            if coordinate_system == CoordinateSystem.PIXEL and self.world_calibration:
                distance = distance * self.world_calibration['scale_factor']

            return distance

        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0

    def calculate_angle(self, point1: Union[Point2D, Tuple[float, float]],
                       point2: Union[Point2D, Tuple[float, float]],
                       point3: Union[Point2D, Tuple[float, float]]) -> float:
        """
        Calculate angle between three points (p1-p2-p3).

        Args:
            point1: First point
            point2: Vertex point
            point3: Third point

        Returns:
            Angle in degrees
        """
        try:
            # Convert to Point2D objects and ensure same coordinate system
            p1 = self._to_common_system(point1, CoordinateSystem.PIXEL)
            p2 = self._to_common_system(point2, CoordinateSystem.PIXEL)
            p3 = self._to_common_system(point3, CoordinateSystem.PIXEL)

            # Calculate vectors
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])

            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)

            return angle_deg

        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return 0.0

    def _to_common_system(self, point: Union[Point2D, Tuple[float, float]], target_system: CoordinateSystem) -> Point2D:
        """Convert point to target coordinate system."""
        if isinstance(point, tuple):
            point = Point2D(point[0], point[1])

        if point.coordinate_system == target_system:
            return point

        if target_system == CoordinateSystem.PIXEL:
            if point.coordinate_system == CoordinateSystem.DATA:
                return self.data_to_pixel(point)
            elif point.coordinate_system == CoordinateSystem.WORLD:
                return self.world_to_pixel(point)
        elif target_system == CoordinateSystem.DATA:
            if point.coordinate_system == CoordinateSystem.PIXEL:
                return self.pixel_to_data(point)
            elif point.coordinate_system == CoordinateSystem.WORLD:
                return self.world_to_data(point)

        return point

    def get_transformation_info(self) -> Dict[str, Any]:
        """
        Get information about current transformation setup.

        Returns:
            Dictionary with transformation information
        """
        return {
            'image_shape': self.image_shape,
            'pixel_extent': self.pixel_extent,
            'world_calibration': self.world_calibration,
            'world_units': self.world_units,
            'coordinate_systems_supported': [cs.value for cs in CoordinateSystem],
        }