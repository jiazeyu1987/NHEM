# Enhanced Realtime Endpoint Documentation

## Overview

The `/data/realtime/enhanced` endpoint has been implemented to provide enhanced dual ROI realtime data with optional ROI1 green line intersection detection.

## Endpoint Details

### URL
```
GET /data/realtime/enhanced
```

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `count` | integer | 100 | 1-1000 | Number of data points to return |
| `include_line_intersection` | boolean | false | true/false | Include ROI1 line intersection detection results |

### Response Model

The endpoint returns an `EnhancedRealtimeDataResponse` object with the following structure:

```python
{
    "type": "enhanced_realtime_data",
    "timestamp": "datetime",
    "frame_count": "integer",
    "series": "List[TimeSeriesPoint]",
    "dual_roi_data": {
        "roi1_data": "RoiData",
        "roi2_data": "RoiData",
        "roi1_config": "RoiConfig",
        "roi2_config": "RoiConfig"
    },
    "peak_signal": "Optional[int]",
    "enhanced_peak": "Optional[EnhancedPeakSignal]",
    "baseline": "float",
    "line_intersection": "Optional[LineIntersectionResult]"  # Only included when requested
}
```

### Line Intersection Result Structure

When `include_line_intersection=true`, the response includes a `line_intersection` field:

```python
{
    "has_intersection": "boolean",
    "intersection": "Optional[Tuple[float, float]]",  # (x, y) coordinates
    "confidence": "float",  # 0.0 to 1.0
    "detected_lines": "List[Tuple[Tuple[int, int, int, int], float]]",
    "processing_time_ms": "float",
    "error_message": "Optional[str]",
    "edge_quality": "float",
    "temporal_stability": "float",
    "frame_count": "integer"
}
```

## Usage Examples

### Basic Usage (Without Line Intersection)
```bash
curl "http://localhost:8421/data/realtime/enhanced?count=50"
```

### With Line Intersection Detection
```bash
curl "http://localhost:8421/data/realtime/enhanced?count=50&include_line_intersection=true"
```

### Python Client Example
```python
import requests

# Basic enhanced data
response = requests.get("http://localhost:8421/data/realtime/enhanced", params={
    "count": 100
})

# Enhanced data with line intersection
response = requests.get("http://localhost:8421/data/realtime/enhanced", params={
    "count": 100,
    "include_line_intersection": True
})

data = response.json()
```

## Behavior

### When `include_line_intersection=false` (default)
- Returns enhanced dual ROI realtime data without line intersection detection
- Faster response time as no image processing is required
- Compatible with existing dual realtime data consumers

### When `include_line_intersection=true`
- Performs ROI1 green line intersection detection
- Requires line detection to be enabled in configuration
- Requires ROI to be configured
- Includes detailed line detection results in the response
- Processing time increases due to image analysis

### Error Handling
The endpoint gracefully handles various error conditions:

1. **Line detection disabled**: Returns result with error message
2. **ROI not configured**: Returns result with error message
3. **Invalid ROI data**: Returns result with error message
4. **Processing timeout**: Returns timeout result
5. **General errors**: Returns error result with detailed message

## Performance Considerations

- **Basic usage**: Similar performance to `/data/dual-realtime` endpoint
- **With line detection**: Additional 50-300ms processing time for image analysis
- **Memory usage**: Minimal additional memory overhead
- **Threading**: Thread-safe implementation compatible with concurrent requests

## Configuration

Line intersection detection behavior can be configured through:

1. **Configuration file**: `backend/app/fem_config.json` section `line_detection`
2. **Runtime API**: `/config` endpoint for dynamic updates
3. **Control endpoints**: `/api/roi/line-intersection/enable` and `/api/roi/line-intersection/disable`

## Integration Notes

### Backward Compatibility
- Existing clients using `/data/dual-realtime` are unaffected
- New endpoint provides superset of dual realtime data functionality
- Gradual migration path available for existing applications

### ROI Processing
- Only ROI1 is processed for line intersection detection (as per requirement 3.3)
- ROI2 continues to provide grayscale analysis as before
- Maintains separation between ROI1 and ROI2 processing pipelines

### Caching
- No caching for line intersection results to ensure real-time accuracy
- ROI capture service may use its own caching mechanism
- Processing time metrics included for performance monitoring

## Implementation Details

### Core Logic Flow
1. Retrieve base dual ROI realtime data
2. Convert to EnhancedRealtimeDataResponse format
3. If line intersection requested:
   - Validate configuration and ROI setup
   - Decode ROI1 image from base64
   - Execute LineIntersectionDetector
   - Include results in response
4. Return enhanced response

### Dependencies
- `LineIntersectionDetector`: Core detection algorithm
- `EnhancedRealtimeDataResponse`: Response model
- `DualRealtimeDataResponse`: Base data source
- Existing ROI capture and data store services

This implementation fulfills Task 14 requirements by providing an enhanced realtime data endpoint with optional line intersection detection while maintaining backward compatibility and proper error handling.