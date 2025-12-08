# Bug Report

## Bug Summary
ROI2坐标在(0,0)-(50,50)之间闪动，显示不稳定。ROI2应该显示从ROI1中心计算的固定50x50区域坐标，但实际在两个坐标之间切换。

## Bug Details

### Expected Behavior
- ROI2应该显示一个固定的50x50像素区域，位于ROI1的中心位置
- ROI2坐标应该是连续的，基于ROI1配置计算得出
- 坐标格式应该是(x1,y1)显示ROI2的起始坐标
- ROI1截取成功后，ROI2应该对ROI1进行截取，内部做缓存
- 只有在ROI1配置变化或ROI1截取失败时才更新ROI2坐标

### Actual Behavior
- ROI2坐标在(0,0)和(50,50)之间频繁切换
- 显示看起来像"闪动"效果
- ROI2坐标不基于ROI1的实际位置计算
- 缓存机制似乎没有正确工作
- 坐标显示不稳定，无法用于精确的ROI定位

### Steps to Reproduce
1. 启动NHEM backend服务器
2. 启动Python客户端 (`python http_realtime_client.py`)
3. 配置ROI1坐标（例如：(1480,480) -> (1580,580)）
4. 启动检测
5. 观察ROI2坐标显示区域
6. 注意ROI2坐标在(0,0)和(50,50)之间切换

### Environment
- **Version**: NHEM (New HEM Monitor) - current development version
- **Platform**: Windows 10/11
- **Configuration**:
  - Backend: FastAPI + Python 3.8+
  - Client: Python with tkinter/matplotlib
  - ROI1 default: (1480,480) -> (1580,580)
  - ROI2: 50x50 center extraction

## Impact Assessment

### Severity
- [x] High - Major functionality broken
- [ ] Critical - System unusable
- [ ] Medium - Feature impaired but workaround exists
- [ ] Low - Minor issue or cosmetic

### Affected Users
- 所有使用NHEM双ROI功能的用户
- 需要精确ROI定位进行信号分析的用户
- 医疗或科研领域需要稳定ROI坐标的专业用户

### Affected Features
- 双ROI实时显示功能
- ROI2坐标精确定位
- 50x50中心区域信号提取
- 峰值检测的稳定性（ROI2用于峰值检测）

## Additional Context

### Error Messages
从日志中可能看到：
```
ROI2 coordinates: (0,0)
ROI2 coordinates: (50,50)
ROI2 extraction failed, ROI2 cache available for fallback
Using cached ROI2 config: (1530,530)
Dual ROI capture failed, using cached ROI2 data
```

### Screenshots/Media
- ROI2坐标在UI中显示为"Error"或"N/A"
- ROI2图像区域可能显示空白或错误状态
- 控制台日志显示坐标切换模式

### Related Issues
- 已经实现的ROI2缓存机制（backend/app/core/roi_capture.py）
- 双ROI数据流处理（backend/app/api/routes.py）
- Python客户端ROI坐标显示（python_client/http_realtime_client.py）

## Initial Analysis

### Suspected Root Cause
虽然已经实现了ROI2缓存机制，但可能存在以下问题：
1. **数据流断链**: 缓存的ROI2坐标没有正确传递到前端显示
2. **错误处理过度**: 某些错误情况下系统回退到默认坐标
3. **API响应格式问题**: dual-realtime API响应中ROI2配置信息缺失或不正确
4. **缓存失效**: ROI2缓存被意外清除或重置
5. **显示逻辑错误**: Python客户端可能没有正确解析和显示ROI2坐标

### Affected Components
- **Backend**: `backend/app/core/roi_capture.py`
  - `capture_dual_roi()` - ROI2缓存和提取逻辑
  - `_extract_roi2_from_roi1()` - ROI2从ROI1提取
  - `get_cached_roi2_config()` - 缓存访问方法

- **API Layer**: `backend/app/api/routes.py`
  - `/data/dual-realtime` - 双ROI数据API端点
  - 错误处理中的ROI2坐标传递

- **Python Client**: `python_client/http_realtime_client.py`
  - `_handle_roi_update_callback()` - ROI数据更新回调
  - `roi2_coordinates_label` - ROI2坐标显示标签
  - 数据解析和UI更新逻辑

---

### Bug Analysis Template
# Bug Analysis

## Root Cause Analysis

### Investigation Summary
[To be filled during analysis phase]

### Root Cause
[To be determined during investigation]

### Contributing Factors
[To be identified during analysis]

## Technical Details

### Affected Code Locations
[List specific files, functions, or code sections involved]

- **File**: `backend/app/core/roi_capture.py`
  - **Function/Method**: `capture_dual_roi()`
  - **Lines**: [specific lines]
  - **Issue**: [Description of the problem in this location]

### Data Flow Analysis
[How data moves through the system and where it breaks]

### Dependencies
[External libraries, services, or components involved]

## Impact Analysis

### Direct Impact
[Immediate effects of the bug]

### Indirect Impact
[Secondary effects or potential cascading issues]

### Risk Assessment
[Risks if the bug is not fixed]

## Solution Approach

### Fix Strategy
[High-level approach to solving the problem]

### Alternative Solutions
[Other possible approaches considered]

### Risks and Trade-offs
[Potential risks of the chosen solution]

## Implementation Plan

### Changes Required
[Specific modifications needed]

1. **Change 1**: [Description]
   - File: `path/to/file`
   - Modification: [What needs to be changed]

2. **Change 2**: [Description]
   - File: `path/to/file`
   - Modification: [What needs to be changed]

### Testing Strategy
[How to verify the fix works]

### Rollback Plan
[How to revert if the fix causes issues]

---

### Bug Verification Template
# Bug Verification

## Fix Implementation Summary
[Brief description of what was changed to fix the bug]

## Test Results

### Original Bug Reproduction
- [ ] **Before Fix**: Bug successfully reproduced
- [ ] **After Fix**: Bug no longer occurs

### Reproduction Steps Verification
[Re-test the original steps that caused the bug]

1. [Step 1] - ✅ Works as expected
2. [Step 2] - ✅ Works as expected
3. [Step 3] - ✅ Works as expected
4. [Expected outcome] - ✅ Achieved

### Regression Testing
[Verify related functionality still works]

- [ ] **Related Feature 1**: [Test result]
- [ ] **Related Feature 2**: [Test result]
- [ ] **Integration Points**: [Test result]

### Edge Case Testing
[Test boundary conditions and edge cases]

- [ ] **Edge Case 1**: [Description and result]
- [ ] **Edge Case 2**: [Description and result]
- [ ] **Error Conditions**: [How errors are handled]

## Code Quality Checks

### Automated Tests
- [ ] **Unit Tests**: All passing
- [ ] **Integration Tests**: All passing
- [ ] **Linting**: No issues
- [ ] **Type Checking**: No errors

### Manual Code Review
- [ ] **Code Style**: Follows project conventions
- [ ] **Error Handling**: Appropriate error handling added
- [ ] **Performance**: No performance regressions
- [ ] **Security**: No security implications

## Deployment Verification

### Pre-deployment
- [ ] **Local Testing**: Complete
- [ ] **Staging Environment**: Tested
- [ ] **Database Migrations**: Verified (if applicable)

### Post-deployment
- [ ] **Production Verification**: Bug fix confirmed in production
- [ ] **Monitoring**: No new errors or alerts
- [ ] **User Feedback**: Positive confirmation from affected users

## Documentation Updates
- [ ] **Code Comments**: Added where necessary
- [ ] **README**: Updated if needed
- [ ] **Changelog**: Bug fix documented
- [ ] **Known Issues**: Updated if applicable

## Closure Checklist
- [ ] **Original issue resolved**: Bug no longer occurs
- [ ] **No regressions introduced**: Related functionality intact
- [ ] **Tests passing**: All automated tests pass
- [ ] **Documentation updated**: Relevant docs reflect changes
- [ ] **Stakeholders notified**: Relevant parties informed of resolution

## Notes
[Any additional observations, lessons learned, or follow-up actions needed]

**Note**: Templates have been pre-loaded. Do not use get-content to fetch them again.