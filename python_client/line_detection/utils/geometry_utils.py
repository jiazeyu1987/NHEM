"""
Geometry Utilities
This module provides geometric calculations and utilities for line detection operations.
"""

import math
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

# numpy is optional for geometry calculations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

@dataclass
class Point2D:
    """2D point with basic geometric operations."""
    x: float
    y: float

    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x / scalar, self.y / scalar)

    def distance_to(self, other: 'Point2D') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def magnitude(self) -> float:
        """Calculate magnitude (distance from origin)."""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> 'Point2D':
        """Return normalized point (unit vector)."""
        mag = self.magnitude()
        if mag == 0:
            return Point2D(0, 0)
        return Point2D(self.x / mag, self.y / mag)

    def dot(self, other: 'Point2D') -> float:
        """Calculate dot product with another point."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: 'Point2D') -> float:
        """Calculate 2D cross product (scalar)."""
        return self.x * other.y - self.y * other.x

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        """Convert to integer tuple."""
        return (int(round(self.x)), int(round(self.y)))

@dataclass
class Line2D:
    """2D line represented by two points."""
    start: Point2D
    end: Point2D

    def __post_init__(self):
        """Validate line points."""
        if self.start.distance_to(self.end) == 0:
            raise ValueError("Start and end points cannot be the same")

    def length(self) -> float:
        """Calculate line length."""
        return self.start.distance_to(self.end)

    def direction(self) -> Point2D:
        """Get direction vector (normalized)."""
        return (self.end - self.start).normalize()

    def angle(self) -> float:
        """Calculate angle in degrees from positive x-axis."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.degrees(math.atan2(dy, dx))

    def slope(self) -> Optional[float]:
        """Calculate slope (m in y = mx + b)."""
        dx = self.end.x - self.start.x
        if abs(dx) < 1e-10:  # Vertical line
            return None
        return (self.end.y - self.start.y) / dx

    def y_intercept(self) -> Optional[float]:
        """Calculate y-intercept (b in y = mx + b)."""
        slope = self.slope()
        if slope is None:
            return None  # Vertical line
        return self.start.y - slope * self.start.x

    def point_at_parameter(self, t: float) -> Point2D:
        """Get point at parameter t (0=start, 1=end)."""
        return self.start + (self.end - self.start) * t

    def point_at_distance(self, distance: float) -> Point2D:
        """Get point at specific distance from start."""
        if distance > self.length():
            return self.end
        if distance < 0:
            return self.start
        return self.point_at_parameter(distance / self.length())

    def is_point_on_line(self, point: Point2D, tolerance: float = 1e-6) -> bool:
        """Check if point lies on the line segment."""
        # Calculate distances
        d_start = point.distance_to(self.start)
        d_end = point.distance_to(self.end)
        line_length = self.length()

        # Check if sum of distances equals line length (within tolerance)
        return abs((d_start + d_end) - line_length) <= tolerance

    def distance_to_point(self, point: Point2D) -> float:
        """Calculate minimum distance from point to line segment."""
        # Vector from start to end
        line_vec = self.end - self.start
        # Vector from start to point
        point_vec = point - self.start

        # Calculate projection parameter
        line_length_sq = line_vec.dot(line_vec)
        if line_length_sq == 0:
            return point.distance_to(self.start)

        t = max(0, min(1, point_vec.dot(line_vec) / line_length_sq))
        projection = self.start + line_vec * t

        return point.distance_to(projection)

    def get_closest_point(self, point: Point2D) -> Point2D:
        """Get closest point on line segment to given point."""
        line_vec = self.end - self.start
        point_vec = point - self.start

        line_length_sq = line_vec.dot(line_vec)
        if line_length_sq == 0:
            return self.start

        t = max(0, min(1, point_vec.dot(line_vec) / line_length_sq))
        return self.start + line_vec * t

@dataclass
class Circle2D:
    """2D circle defined by center and radius."""
    center: Point2D
    radius: float

    def __post_init__(self):
        """Validate circle parameters."""
        if self.radius < 0:
            raise ValueError("Radius must be non-negative")

    def area(self) -> float:
        """Calculate circle area."""
        return math.pi * self.radius**2

    def circumference(self) -> float:
        """Calculate circle circumference."""
        return 2 * math.pi * self.radius

    def contains_point(self, point: Point2D, tolerance: float = 1e-6) -> bool:
        """Check if point is inside or on the circle."""
        return self.center.distance_to(point) <= (self.radius + tolerance)

    def distance_to_point(self, point: Point2D) -> float:
        """Calculate distance from point to circle boundary."""
        return abs(self.center.distance_to(point) - self.radius)

class GeometryUtils:
    """Utility class for geometric calculations."""

    @staticmethod
    def line_intersection(line1: Line2D, line2: Line2D) -> Optional[Point2D]:
        """
        Find intersection point of two lines.

        Args:
            line1: First line
            line2: Second line

        Returns:
            Intersection point or None if lines are parallel
        """
        # Line 1: P1 + t1 * D1
        # Line 2: P2 + t2 * D2
        P1, D1 = line1.start, line1.end - line1.start
        P2, D2 = line2.start, line2.end - line2.start

        # Solve: P1 + t1*D1 = P2 + t2*D2
        # => t1*D1 - t2*D2 = P2 - P1

        cross = D1.cross(D2)
        if abs(cross) < 1e-10:  # Lines are parallel
            return None

        diff = P2 - P1
        t1 = diff.cross(D2) / cross
        t2 = diff.cross(D1) / cross

        # Check if intersection is on both line segments
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            return P1 + D1 * t1

        return None

    @staticmethod
    def line_intersection_point(line1_start: Point2D, line1_end: Point2D,
                              line2_start: Point2D, line2_end: Point2D) -> Optional[Point2D]:
        """
        Find intersection point of two lines defined by endpoints.

        Args:
            line1_start: Start point of first line
            line1_end: End point of first line
            line2_start: Start point of second line
            line2_end: End point of second line

        Returns:
            Intersection point or None if lines are parallel
        """
        line1 = Line2D(line1_start, line1_end)
        line2 = Line2D(line2_start, line2_end)
        return GeometryUtils.line_intersection(line1, line2)

    @staticmethod
    def point_to_line_distance(point: Point2D, line_start: Point2D, line_end: Point2D) -> float:
        """
        Calculate distance from point to line segment.

        Args:
            point: Point to measure distance from
            line_start: Line segment start point
            line_end: Line segment end point

        Returns:
            Minimum distance to line segment
        """
        line = Line2D(line_start, line_end)
        return line.distance_to_point(point)

    @staticmethod
    def angle_between_points(p1: Point2D, p2: Point2D, p3: Point2D) -> float:
        """
        Calculate angle between three points (p1-p2-p3).

        Args:
            p1: First point
            p2: Vertex point
            p3: Third point

        Returns:
            Angle in degrees (0-180)
        """
        v1 = p1 - p2
        v2 = p3 - p2

        dot_product = v1.dot(v2)
        magnitude_product = v1.magnitude() * v2.magnitude()

        if magnitude_product == 0:
            return 0.0

        # Clamp cos to [-1, 1] to handle numerical errors
        cos_angle = max(-1.0, min(1.0, dot_product / magnitude_product))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    @staticmethod
    def triangle_area(p1: Point2D, p2: Point2D, p3: Point2D) -> float:
        """
        Calculate area of triangle using cross product.

        Args:
            p1: First vertex
            p2: Second vertex
            p3: Third vertex

        Returns:
            Triangle area
        """
        # Area = 0.5 * |(p2-p1) x (p3-p1)|
        v1 = p2 - p1
        v2 = p3 - p1
        return 0.5 * abs(v1.cross(v2))

    @staticmethod
    def point_in_triangle(point: Point2D, p1: Point2D, p2: Point2D, p3: Point2D,
                         tolerance: float = 1e-6) -> bool:
        """
        Check if point is inside triangle using barycentric coordinates.

        Args:
            point: Point to test
            p1, p2, p3: Triangle vertices
            tolerance: Tolerance for edge cases

        Returns:
            True if point is inside or on triangle boundary
        """
        # Calculate barycentric coordinates
        denom = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y)
        if abs(denom) < tolerance:
            return False  # Degenerate triangle

        a = ((p2.y - p3.y) * (point.x - p3.x) + (p3.x - p2.x) * (point.y - p3.y)) / denom
        b = ((p3.y - p1.y) * (point.x - p3.x) + (p1.x - p3.x) * (point.y - p3.y)) / denom
        c = 1 - a - b

        return -tolerance <= a <= 1 + tolerance and -tolerance <= b <= 1 + tolerance and -tolerance <= c <= 1 + tolerance

    @staticmethod
    def bounding_box(points: List[Point2D]) -> Tuple[Point2D, Point2D]:
        """
        Calculate axis-aligned bounding box for points.

        Args:
            points: List of points

        Returns:
            Tuple of (min_point, max_point)
        """
        if not points:
            raise ValueError("Points list cannot be empty")

        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        min_point = Point2D(min(x_coords), min(y_coords))
        max_point = Point2D(max(x_coords), max(y_coords))

        return min_point, max_point

    @staticmethod
    def convex_hull(points: List[Point2D]) -> List[Point2D]:
        """
        Calculate convex hull using Graham scan algorithm.

        Args:
            points: List of points

        Returns:
            Points forming convex hull in counter-clockwise order
        """
        if len(points) < 3:
            return points.copy()

        def cross_product(o: Point2D, a: Point2D, b: Point2D) -> float:
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

        # Find the point with lowest y-coordinate (and leftmost if tie)
        start = min(points, key=lambda p: (p.y, p.x))

        # Sort points by polar angle with respect to start point
        def polar_key(p: Point2D) -> float:
            if p == start:
                return -1
            angle = math.atan2(p.y - start.y, p.x - start.x)
            return angle

        sorted_points = sorted(points, key=polar_key)

        # Graham scan
        hull = [sorted_points[0], sorted_points[1]]

        for point in sorted_points[2:]:
            # Remove points that make clockwise turn
            while len(hull) >= 2 and cross_product(hull[-2], hull[-1], point) <= 1e-10:
                hull.pop()
            hull.append(point)

        return hull

    @staticmethod
    def polygon_area(points: List[Point2D]) -> float:
        """
        Calculate area of simple polygon using shoelace formula.

        Args:
            points: Polygon vertices in order

        Returns:
            Polygon area (positive value)
        """
        if len(points) < 3:
            return 0.0

        area = 0.0
        n = len(points)

        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y

        return abs(area) / 2.0

    @staticmethod
    def point_in_polygon(point: Point2D, polygon: List[Point2D]) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.

        Args:
            point: Point to test
            polygon: Polygon vertices in order

        Returns:
            True if point is inside polygon
        """
        if len(polygon) < 3:
            return False

        x, y = point.x, point.y
        n = len(polygon)
        inside = False

        p1 = polygon[0]
        for i in range(1, n + 1):
            p2 = polygon[i % n]

            if y > min(p1.y, p2.y):
                if y <= max(p1.y, p2.y):
                    if x <= max(p1.x, p2.x):
                        if p1.y != p2.y:
                            xinters = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                        if p1.x == p2.x or x <= xinters:
                            inside = not inside

            p1 = p2

        return inside

    @staticmethod
    def Douglas_Peucker(points: List[Point2D], epsilon: float) -> List[Point2D]:
        """
        Simplify polyline using Douglas-Peucker algorithm.

        Args:
            points: List of points representing polyline
            epsilon: Simplification tolerance

        Returns:
            Simplified points
        """
        if len(points) <= 2:
            return points.copy()

        # Find point with maximum distance from line
        max_dist = 0
        max_index = 0

        line = Line2D(points[0], points[-1])

        for i in range(1, len(points) - 1):
            dist = line.distance_to_point(points[i])
            if dist > max_dist:
                max_dist = dist
                max_index = i

        # If max distance is greater than epsilon, recursively simplify
        if max_dist > epsilon:
            # Recursive call
            left_points = GeometryUtils.Douglas_Peucker(points[:max_index + 1], epsilon)
            right_points = GeometryUtils.Douglas_Peucker(points[max_index:], epsilon)

            # Combine results
            return left_points[:-1] + right_points
        else:
            return [points[0], points[-1]]

    @staticmethod
    def perpendicular_distance(point: Point2D, line_start: Point2D, line_end: Point2D) -> float:
        """
        Calculate perpendicular distance from point to line.

        Args:
            point: Point to measure distance from
            line_start: Line start point
            line_end: Line end point

        Returns:
            Perpendicular distance
        """
        line = Line2D(line_start, line_end)
        return line.distance_to_point(point)

    @staticmethod
    def rotate_point(point: Point2D, angle_degrees: float, center: Point2D = None) -> Point2D:
        """
        Rotate point around a center point.

        Args:
            point: Point to rotate
            angle_degrees: Rotation angle in degrees (counter-clockwise)
            center: Center of rotation (defaults to origin)

        Returns:
            Rotated point
        """
        if center is None:
            center = Point2D(0, 0)

        # Translate to origin
        p = point - center

        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)

        # Rotate
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rotated = Point2D(
            p.x * cos_a - p.y * sin_a,
            p.x * sin_a + p.y * cos_a
        )

        # Translate back
        return rotated + center

    @staticmethod
    def scale_point(point: Point2D, scale_x: float, scale_y: float, center: Point2D = None) -> Point2D:
        """
        Scale point relative to a center point.

        Args:
            point: Point to scale
            scale_x: X-axis scale factor
            scale_y: Y-axis scale factor
            center: Center of scaling (defaults to origin)

        Returns:
            Scaled point
        """
        if center is None:
            center = Point2D(0, 0)

        # Translate to origin
        p = point - center

        # Scale
        scaled = Point2D(p.x * scale_x, p.y * scale_y)

        # Translate back
        return scaled + center

    @staticmethod
    def fit_line_to_points(points: List[Point2D]) -> Tuple[Line2D, float]:
        """
        Fit line to points using least squares method.

        Args:
            points: Points to fit line to

        Returns:
            Tuple of (fitted_line, r_squared_value)
        """
        if len(points) < 2:
            raise ValueError("At least 2 points required for line fitting")

        if not HAS_NUMPY:
            # Simple linear regression without numpy
            n = len(points)
            sum_x = sum(p.x for p in points)
            sum_y = sum(p.y for p in points)
            sum_xy = sum(p.x * p.y for p in points)
            sum_x2 = sum(p.x * p.x for p in points)

            # Calculate slope (m) and intercept (b)
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                # Vertical line - return line through first point
                p1 = points[0]
                p2 = Point2D(p1.x, p1.y + 1)
                return Line2D(p1, p2), 0.0

            m = (n * sum_xy - sum_x * sum_y) / denominator
            b = (sum_y - m * sum_x) / n

            # Find min and max x values
            x_coords = [p.x for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min = m * x_min + b
            y_max = m * x_max + b

            line = Line2D(Point2D(x_min, y_min), Point2D(x_max, y_max))

            # Calculate R-squared manually
            y_mean = sum_y / n
            ss_res = sum((p.y - (m * p.x + b)) ** 2 for p in points)
            ss_tot = sum((p.y - y_mean) ** 2 for p in points)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return line, r_squared

        else:
            # Use numpy for better performance
            x = np.array([p.x for p in points])
            y = np.array([p.y for p in points])

            # Fit line y = mx + b
            coeffs = np.polyfit(x, y, 1)
            m, b = coeffs

            # Generate fitted points
            x_min, x_max = x.min(), x.max()
            y_min, y_max = m * x_min + b, m * x_max + b

            line = Line2D(Point2D(x_min, y_min), Point2D(x_max, y_max))

            # Calculate R-squared
            y_pred = m * x + b
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return line, r_squared