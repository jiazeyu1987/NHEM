# Bug Report

## Bug Summary
Need to implement dual ROI (Region of Interest) functionality with two independent ROI capture areas. The new ROI should read from configuration table, display to the right of the existing ROI, while the original ROI should be modified to capture a fixed 50x50 region from the new ROI's center without reading from configuration.

## Bug Details

### Expected Behavior
System should support two simultaneous ROI capture areas:
1. **New ROI**: Reads coordinates from configuration table, displays on the right side
2. **Original ROI**: Captures fixed 50x50 area from center of new ROI, displays in original position
3. Both ROIs should capture at the same frequency
4. All existing analysis and logic should continue to work with the original ROI

### Actual Behavior
Currently only supports single ROI capture from configuration table. Need to extend to dual ROI system with different capture logic for each ROI.

### Steps to Reproduce
1. Current system reads ROI configuration from `fem_config.json`
2. Single ROI capture processes at configured frame rate
3. All analysis performed on this single ROI data
4. No support for second ROI or different capture logic

### Environment
- **Version**: NHEM (New HEM Monitor) current version
- **Platform**: Windows-based development environment
- **Configuration**: Current single ROI configuration in `backend/app/fem_config.json`

## Impact Assessment

### Severity
- [ ] Critical - System unusable
- [x] High - Major functionality broken
- [ ] Medium - Feature impaired but workaround exists
- [ ] Low - Minor issue or cosmetic

### Affected Users
Users requiring dual region monitoring for enhanced HEM detection capabilities

### Affected Features
- ROI capture service
- Frontend display (needs to show two ROI areas)
- Configuration management
- Data processing pipeline

## Additional Context

### Current Configuration Structure
```json
"roi_capture": {
  "frame_rate": 4,
  "update_interval": 0.5,
  "default_config": {
    "x1": 1480,
    "y1": 480,
    "x2": 1580,
    "y2": 580
  }
}
```

### Error Messages
No current errors - this is a feature enhancement/bug for missing functionality

### Screenshots/Media
Current single ROI display needs to be extended to dual ROI display

### Related Issues
This extends the existing ROI capture system without breaking current functionality

## Initial Analysis

### Suspected Root Cause
System architecture designed for single ROI capture only. Need to extend data structures, processing logic, and UI to support dual ROI streams.

### Affected Components
- **Backend**: `app/core/roi_capture.py` - ROI capture logic
- **Backend**: `app/core/data_store.py` - Data storage for multiple ROI streams
- **Backend**: `app/api/routes.py` - API endpoints for ROI data
- **Frontend**: `front/index.html` - Canvas rendering for dual ROI display
- **Configuration**: `backend/app/fem_config.json` - Configuration structure for dual ROI
- **Python Client**: `python_client/` - Client display updates