# Dual ROI Implementation Validation

## Task 1.1: Create dual-ROI data models

### Implementation Summary
Successfully added dual-ROI data models to `backend/app/models.py` with the following components:

### 1. DualRoiMode Enum
```python
class DualRoiMode(str, Enum):
    """双ROI模式枚举"""
    SINGLE = "single"  # 传统单ROI模式
    DUAL = "dual"      # 双ROI模式
```

### 2. DualRoiConfig Class
```python
class DualRoiConfig(BaseModel):
    """双ROI配置模型"""
    mode: DualRoiMode = Field(DualRoiMode.SINGLE, description="ROI模式: single(单ROI)或dual(双ROI)")
    large_roi: RoiConfig = Field(
        default_factory=lambda: RoiConfig(x1=0, y1=0, x2=300, y2=200),
        description="大ROI配置(可配置，默认300x200)"
    )
    small_roi_size: int = Field(50, ge=10, le=200, description="小ROI尺寸(固定50x50)")
```

**Features:**
- `mode` field supports both single and dual ROI modes (requirement 1.1)
- `large_roi` configurable with default 300x200 size
- `small_roi_size` fixed 50x50 as specified (requirement 1.2)
- Automatic small ROI extraction from large ROI center
- Comprehensive validation methods for ROI bounds and minimum sizes

### 3. RoiFrame Class (Added)
```python
class RoiFrame(BaseModel):
    """ROI帧数据模型"""
    roi_data: RoiData
    timestamp: datetime
    frame_count: int
```

### 4. DualRoiFrame Class
```python
class DualRoiFrame(BaseModel):
    """双ROI帧数据模型"""
    large_roi: RoiData
    small_roi: RoiData
    timestamp: datetime
    frame_count: int
    mode: DualRoiMode = Field(DualRoiMode.DUAL, description="当前帧的ROI模式")
```

**Features:**
- Contains both large and small ROI data
- Timestamp and frame count tracking
- Mode-aware primary ROI selection
- Backward compatibility methods

### Key Methods Implemented

#### DualRoiConfig.small_roi Property
- Automatically extracts 50x50 small ROI from center of large ROI
- Handles boundary conditions with max(0, ...) to prevent negative coordinates

#### DualRoiConfig.validate_roi_bounds()
- Validates large ROI coordinates within screen bounds (default 1920x1080)
- Ensures large ROI can contain the small ROI
- Validates small ROI size constraints (10-200 pixels)
- Returns boolean validation result

#### DualRoiConfig.to_legacy_config()
- Backward compatibility method for existing single-ROI system
- Returns appropriate ROI based on mode

#### DualRoiFrame.primary_roi Property
- Returns appropriate ROI data based on current mode
- Large ROI for single mode, small ROI for dual mode

#### DualRoiFrame.to_legacy_frame()
- Converts dual-ROI frame to legacy single-ROI frame
- Maintains backward compatibility

### Requirements Satisfied

✅ **Requirement 1.1**: Support both dual-ROI mode and legacy single-ROI mode
- Implemented through `DualRoiMode` enum and mode-aware behavior
- Default mode is SINGLE for backward compatibility

✅ **Requirement 1.2**: Store configuration for both large ROI (configurable) and small ROI (fixed 50x50)
- Large ROI configurable with default 300x200
- Small ROI size fixed at 50x50
- Automatic center extraction of small ROI from large ROI

### Code Quality Features
- Type-safe Pydantic models with proper validation
- Comprehensive Field descriptions for documentation
- Property methods for computed values
- Backward compatibility methods
- Validation methods with reasonable default parameters
- Follows existing code patterns and conventions
- Proper Chinese comments matching project style

### Integration Points
The implementation leverages existing components:
- Extends existing `RoiConfig` class
- Uses existing `RoiData` for ROI frame data
- Follows same Pydantic BaseModel patterns
- Maintains compatibility with existing API structures

The implementation is ready for integration and provides a solid foundation for the dual-ROI system.