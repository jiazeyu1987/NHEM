# Dual ROI System Documentation

## Overview

The NHEM Dual ROI system provides advanced region-of-interest processing with two separate ROI areas working in coordination:

- **ROI1 (Large Region)**: Main capture area for display and user visualization
- **ROI2 (Center Region)**: 50x50 pixel center region extracted from ROI1 for precise signal analysis

## System Architecture

### Dual ROI Data Flow

```
Screen Capture → ROI1 (1100x500) → Image Processing → ROI2 Extract (50x50) → Peak Detection
                                     ↓
                              Display to User    ←    Gray Value Analysis
```

### Key Components

#### 1. ROI Configuration (`RoiConfig`)
```python
class RoiConfig(BaseModel):
    x1: int  # ROI1 left coordinate
    y1: int  # ROI1 top coordinate
    x2: int  # ROI1 right coordinate
    y2: int  # ROI1 bottom coordinate

    @property
    def center_x(self) -> int  # ROI1 center X coordinate
    @property
    def center_y(self) -> int  # ROI1 center Y coordinate
```

#### 2. ROI Data Structure (`RoiData`)
```python
class RoiData(BaseModel):
    width: int           # ROI width
    height: int          # ROI height
    pixels: str          # Base64 encoded PNG image
    gray_value: float    # Average gray value (0-255)
    timestamp: datetime  # Capture timestamp
```

#### 3. ROI Capture Service (`roi_capture.py`)
Core service responsible for:
- **Dual ROI Extraction**: Captures ROI1 and extracts ROI2 from ROI1 center
- **Coordinate Transformation**: Maps screen coordinates to image coordinates
- **Image Processing**: Resizes, encodes, and analyzes ROI data
- **Error Handling**: Fallback mechanisms for capture failures

## Technical Implementation

### ROI2 Extraction Algorithm

1. **Screen Coordinate Calculation**:
   ```python
   # ROI2 center in ROI1 coordinates
   roi2_center_x = (roi2_x1 + roi2_x2) // 2 - roi1_config.x1
   roi2_center_y = (roi2_y1 + roi2_y2) // 2 - roi1_config.y1
   ```

2. **Image Space Transformation**:
   ```python
   # Scale from screen coordinates to resized image coordinates
   scale_x = roi1_image.width / roi1_screen_width
   scale_y = roi1_image.height / roi1_screen_height

   rel_center_x = int(roi2_center_x * scale_x)
   rel_center_y = int(roi2_center_y * scale_y)
   ```

3. **ROI2 Region Definition**:
   ```python
   # Adaptive sizing based on image dimensions
   roi2_image_size = min(50, min(roi1_image.width, roi1_image.height) // 4)
   rel_x1 = max(0, rel_center_x - roi2_image_size // 2)
   rel_y1 = max(0, rel_center_y - roi2_image_size // 2)
   rel_x2 = min(roi1_image.width, rel_x1 + roi2_image_size)
   rel_y2 = min(roi1_image.height, rel_y1 + roi2_image_size)
   ```

### Gray Value Calculation

The system uses histogram-based gray value calculation for accuracy:

```python
def calculate_gray_value(image):
    gray_image = image.convert('L')  # Convert to grayscale
    histogram = gray_image.histogram()
    total_sum = sum(i * count for i, count in enumerate(histogram))
    total_pixels = image.width * image.height
    return float(total_sum / total_pixels)
```

### Error Handling and Fallbacks

1. **ROI2 Extraction Failure**:
   - Falls back to ROI1 gray value for signal processing
   - Logs appropriate warnings for debugging
   - Maintains system operation even with ROI2 issues

2. **Coordinate Bounds Errors**:
   - Uses safe fallback coordinates (0,0) to (50,50)
   - Validates coordinates before processing
   - Prevents system crashes from invalid coordinates

3. **Image Processing Errors**:
   - Graceful degradation to simulated data
   - Comprehensive error logging
   - Automatic retry mechanisms

## Performance Characteristics

### Frame Rates

- **Main Processing**: 45 FPS (22.22ms intervals)
- **ROI Capture**: 4 FPS (250ms intervals) - separate from main processing
- **WebSocket Broadcasting**: 60 FPS (16.67ms intervals)
- **Frontend Updates**: 20 FPS (50ms intervals)

### Memory Usage

- **ROI1 Images**: 200x150 pixels, PNG encoded (Base64)
- **ROI2 Images**: 50x50 pixels extracted from ROI1 center
- **Circular Buffers**: 100-10000 frames configurable
- **Peak Detection**: Enhanced algorithm with configurable parameters

### Optimization Features

- **Intelligent Caching**: Reduces redundant screen captures
- **Coordinate Transformation**: Efficient screen-to-image mapping
- **Adaptive ROI Sizing**: Automatically adjusts to image dimensions
- **Memory Management**: Circular buffers prevent memory leaks

## Configuration

### ROI1 Configuration
```json
{
  "roi": {
    "x1": 480,
    "y1": 80,
    "x2": 1580,
    "y2": 580,
    "frame_rate": 4
  }
}
```

### ROI2 Parameters
- **Size**: 50x50 pixels (adaptive based on ROI1 size)
- **Location**: Center region of ROI1
- **Extraction Method**: Cropped from ROI1 image (not direct screen capture)

### Peak Detection Configuration
```json
{
  "peak_detection": {
    "threshold": 104.0,
    "margin_frames": 5,
    "difference_threshold": 1.1,
    "min_region_length": 5
  }
}
```

## API Integration

### Dual ROI Endpoints

#### Get Dual ROI Data
```http
GET /api/roi/dual-frame
```

Returns both ROI1 and ROI2 data:
```json
{
  "type": "dual_roi_frame",
  "timestamp": "2025-12-07T21:00:00Z",
  "roi1_data": {
    "width": 200,
    "height": 150,
    "pixels": "data:image/png;base64,iVBORw0KGgo...",
    "gray_value": 156.78
  },
  "roi2_data": {
    "width": 50,
    "height": 50,
    "pixels": "data:image/png;base64,iVBORw0KGgo...",
    "gray_value": 142.35
  }
}
```

#### Configure ROI
```http
POST /api/roi/config
Content-Type: application/json

{
  "x1": 480,
  "y1": 80,
  "x2": 1580,
  "y2": 580,
  "password": "31415"
}
```

## Frontend Integration

### ROI Display
- **ROI1**: Large image display for user visualization
- **ROI2**: Small center region display with gray value analysis
- **Real-time Updates**: 20 FPS refresh rate
- **Status Indicators**: Color-coded ROI health status

### Error States
- **Green**: ROI2 data normal
- **Orange**: ROI2 gray value is 0 or warning state
- **Red**: ROI2 extraction error

## Python Client Integration

### GUI Components
- **ROI1 Display**: Main capture region visualization
- **ROI2 Display**: Center region analysis
- **Gray Value Monitoring**: Real-time gray value display
- **Status Indicators**: Error state visualization

### Configuration
```python
# ROI Configuration
client.set_roi_config(
    x1=480, y1=80, x2=1580, y2=580,
    password="31415"
)

# Get Dual ROI Data
dual_frame = client.get_dual_roi_frame()
roi1_gray = dual_frame.roi1_data.gray_value
roi2_gray = dual_frame.roi2_data.gray_value
```

## Usage Examples

### Basic Dual ROI Setup
```python
from nhem.client import NHEMClient

client = NHEMClient("http://localhost:8421")

# Configure ROI1 region (ROI2 automatically extracted from center)
client.set_roi_config(
    x1=480, y1=80,     # ROI1 top-left
    x2=1580, y2=580,   # ROI1 bottom-right
    password="31415"
)

# Start detection
client.start_detection(password="31415")

# Monitor dual ROI data
while True:
    dual_frame = client.get_dual_roi_frame()
    print(f"ROI1: {dual_frame.roi1_data.gray_value:.1f}")
    print(f"ROI2: {dual_frame.roi2_data.gray_value:.1f}")
    time.sleep(0.05)  # 20 FPS
```

### Error Handling
```python
try:
    dual_frame = client.get_dual_roi_frame()

    if dual_frame.roi2_data.gray_value == 0:
        print("Warning: ROI2 gray value is 0")
        print(f"Using ROI1 fallback: {dual_frame.roi1_data.gray_value:.1f}")

    # Process ROI data...

except Exception as e:
    print(f"ROI capture error: {e}")
    # Implement fallback logic
```

## Troubleshooting

### Common Issues

1. **ROI2 Shows Black Images**
   - **Cause**: ROI1 center region is actually black on screen
   - **Solution**: Adjust ROI1 coordinates to capture region with content

2. **ROI2 Gray Value is 0**
   - **Cause**: ROI2 extraction failed or ROI2 region is black
   - **Check**: ROI1 configuration and screen content

3. **Coordinate Bounds Errors**
   - **Cause**: ROI coordinates exceed screen dimensions
   - **Solution**: Validate ROI configuration within screen bounds

### Debug Logging

Enable debug logging for detailed ROI processing information:

```python
import logging
logging.getLogger("nhem.roi_capture").setLevel(logging.DEBUG)
```

### Performance Optimization

1. **Reduce ROI Capture Rate**: Lower frame rate for ROI capture
2. **Optimize ROI Size**: Smaller ROI1 for faster processing
3. **Adjust Buffer Sizes**: Configure appropriate circular buffer sizes
4. **Monitor Memory Usage**: Check for memory leaks in long-running sessions

## Security Considerations

### Authentication
- **Control Commands**: Password-protected (default: 31415)
- **Data Access**: Public endpoints for ROI data
- **Configuration**: Password required for ROI configuration changes

### Data Validation
- **Input Sanitization**: ROI coordinate validation
- **Bounds Checking**: Prevent invalid screen coordinates
- **Error Handling**: Graceful handling of malformed requests

## Future Enhancements

### Planned Features
1. **Multiple ROI2 Regions**: Support for multiple analysis regions
2. **Adaptive ROI Sizing**: Dynamic ROI2 size based on content
3. **Machine Learning Integration**: Advanced pattern recognition
4. **Performance Monitoring**: ROI processing metrics dashboard

### Extension Points
- **Custom ROI Algorithms**: Plugin system for custom extraction methods
- **Alternative Image Formats**: Support for JPEG, WebP formats
- **Real-time Filters**: Image preprocessing and enhancement
- **Export Functionality**: ROI data export and analysis tools

This comprehensive dual ROI system provides precise signal analysis capabilities while maintaining excellent performance and reliability for real-time HEM monitoring applications.