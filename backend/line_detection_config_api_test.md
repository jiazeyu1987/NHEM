# Line Detection Configuration API Test Documentation

This document describes the three new API endpoints added for line detection configuration management as part of Task 15.

## API Endpoints Added

### 1. GET `/api/roi/line-intersection/config`

Retrieves the current line detection configuration.

**Parameters:**
- `password` (query, required): Admin password

**Response Format:**
```json
{
    "timestamp": "2025-12-08T12:00:00.000Z",
    "success": true,
    "data": {
        "enabled": false,
        "hsv_green_lower": [40, 50, 50],
        "hsv_green_upper": [80, 255, 255],
        "canny_low_threshold": 25,
        "canny_high_threshold": 80,
        "hough_threshold": 50,
        "hough_min_line_length": 15,
        "hough_max_line_gap": 8,
        "min_confidence": 0.4,
        "roi_processing_mode": "roi1_only",
        "cache_timeout_ms": 100,
        "max_processing_time_ms": 300,
        "min_angle_degrees": 10.0,
        "max_angle_degrees": 80.0,
        "parallel_threshold": 0.01
    },
    "message": "Line detection configuration retrieved successfully"
}
```

**Example Request:**
```bash
curl -X GET "http://localhost:8421/api/roi/line-intersection/config?password=31415"
```

### 2. POST `/api/roi/line-intersection/config`

Updates line detection configuration parameters with partial update support.

**Parameters:**
- `password` (form, required): Admin password
- `enabled` (form, optional): Boolean to enable/disable line detection
- `hsv_green_lower_0`, `hsv_green_lower_1`, `hsv_green_lower_2` (form, optional): HSV lower bounds (H,S,V)
- `hsv_green_upper_0`, `hsv_green_upper_1`, `hsv_green_upper_2` (form, optional): HSV upper bounds (H,S,V)
- `canny_low_threshold` (form, optional, range 0-255): Canny edge detection low threshold
- `canny_high_threshold` (form, optional, range 0-255): Canny edge detection high threshold
- `hough_threshold` (form, optional, min 1): Hough line transform vote threshold
- `hough_min_line_length` (form, optional, min 1): Minimum line length in pixels
- `hough_max_line_gap` (form, optional, min 0): Maximum line gap in pixels
- `min_confidence` (form, optional, range 0.0-1.0): Minimum confidence threshold
- `roi_processing_mode` (form, optional): ROI processing mode (currently only "roi1_only")
- `cache_timeout_ms` (form, optional, min 0): Result cache timeout in milliseconds
- `max_processing_time_ms` (form, optional, min 50): Maximum processing time limit in milliseconds
- `min_angle_degrees` (form, optional, range 0.0-90.0): Minimum angle for filtering horizontal lines
- `max_angle_degrees` (form, optional, range 0.0-90.0): Maximum angle for filtering vertical lines
- `parallel_threshold` (form, optional, range 0.0001-1.0): Parallel line detection threshold

**Response Format:**
```json
{
    "timestamp": "2025-12-08T12:00:00.000Z",
    "success": true,
    "data": {
        "enabled": true,
        "min_confidence": 0.6
    },
    "message": "Line detection configuration updated: enabled=True, min_confidence=0.6"
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8421/api/roi/line-intersection/config" \
  -F "password=31415" \
  -F "enabled=true" \
  -F "min_confidence=0.6"
```

### 3. POST `/api/roi/line-intersection/config/reset`

Resets line detection configuration to default values.

**Parameters:**
- `password` (form, required): Admin password

**Response Format:**
```json
{
    "timestamp": "2025-12-08T12:00:00.000Z",
    "success": true,
    "data": {
        "enabled": false,
        "hsv_green_lower": [40, 50, 50],
        "hsv_green_upper": [80, 255, 255],
        "canny_low_threshold": 25,
        "canny_high_threshold": 80,
        "hough_threshold": 50,
        "hough_min_line_length": 15,
        "hough_max_line_gap": 8,
        "min_confidence": 0.4,
        "roi_processing_mode": "roi1_only",
        "cache_timeout_ms": 100,
        "max_processing_time_ms": 300,
        "min_angle_degrees": 10.0,
        "max_angle_degrees": 80.0,
        "parallel_threshold": 0.01
    },
    "message": "Line detection configuration reset to defaults successfully"
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8421/api/roi/line-intersection/config/reset" \
  -F "password=31415"
```

## Key Implementation Features

### Configuration Validation
- HSV ranges are validated (H: 0-179, S/V: 0-255)
- Canny thresholds validated with low < high requirement
- Hough parameters validated with min_line_length > max_line_gap
- Angle ranges validated (0-90 degrees) with min < max
- All parameters have appropriate range constraints

### Persistence
- Configuration updates are saved to JSON file (`app/fem_config.json`)
- Runtime settings are updated immediately for real-time effect
- Thread-safe configuration updates using ConfigManager

### Partial Updates
- Only provided parameters are updated
- Supports single or multiple parameter updates in one request
- Maintains existing values for unspecified parameters

### Error Handling
- Comprehensive validation with detailed error messages
- NHEM-standard error response format
- Proper HTTP status codes (400 for validation errors, 500 for system errors)

### Authentication
- All endpoints require password authentication
- Uses existing `verify_password()` function
- Consistent with NHEM security patterns

## Integration with Existing System

### ConfigManager Integration
- Uses existing `ConfigManager` for JSON file operations
- Follows established patterns for configuration sections
- Maintains consistency with other configuration endpoints

### Runtime Updates
- Updates `settings.line_detection` object for immediate effect
- Ensures changes take effect without server restart
- Maintains backward compatibility

### NHEM Response Format
- Follows established response format with timestamp, success, data, message
- Uses existing `ErrorResponse` and `ErrorDetails` models
- Consistent logging patterns with structured logging

This implementation successfully fulfills Task 15 requirements for configuration management endpoints with runtime updates and persistence.