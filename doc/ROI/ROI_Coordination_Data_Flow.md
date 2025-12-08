# ROI Coordination and Data Flow Documentation

## Overview

The NHEM dual ROI system implements sophisticated coordination between ROI1 (large region) and ROI2 (center region) with precise data flow management, synchronization mechanisms, and intelligent fallback strategies. This document details the complete coordination logic, data flow architecture, and synchronization protocols.

## Architecture Overview

```
Screen Environment
    ↓
ROI1 Capture (1100x500) ──────┐
    ↓                        │
ROI1 Processing               │
    ↓                        │
ROI2 Extraction (50x50) ←────┘
    ↓
Dual ROI Processing
    ↓
Synchronized Data Storage
    ↓
Multi-Client Distribution
```

## 1. Coordination Architecture

### 1.1 Dual ROI Capture Coordination (`backend/app/core/roi_capture.py`)

#### Master Coordination Method

**Location**: `capture_dual_roi()` (lines 197-273)

```python
def capture_dual_roi(self, roi_config: RoiConfig) -> Tuple[Optional[RoiData], Optional[RoiData]]:
    """
    Coordinated dual ROI capture with atomic operation guarantee

    Coordination Strategy:
    1. Atomic capture operation - both ROIs captured in single transaction
    2. Dependency management - ROI2 depends on ROI1 success
    3. Synchronized timing - both ROIs share capture timestamp
    4. Consistent error handling - coordinated fallback mechanisms
    """
    capture_start_time = time.time()
    capture_timestamp = datetime.now()

    try:
        # Phase 1: ROI1 Capture (Primary Operation)
        roi1_data = self._capture_roi_internal(roi_config)
        if not roi1_data:
            logger.warning("ROI1 capture failed, dual ROI operation aborted")
            return None, None

        # Phase 2: ROI2 Extraction (Secondary Operation)
        roi2_data = self._extract_roi2_from_roi1_data(roi1_data, roi_config)

        # Phase 3: Coordination Validation
        self._validate_roi_coordination(roi1_data, roi2_data, roi_config)

        # Performance metrics
        capture_duration = time.time() - capture_start_time
        logger.debug(f"Dual ROI capture completed in {capture_duration:.3f}s")

        return roi1_data, roi2_data

    except Exception as e:
        logger.error(f"Coordinated dual ROI capture failed: {e}")
        return self._handle_coordination_failure(e, roi_config)
```

#### Coordination Validation

```python
def _validate_roi_coordination(self, roi1_data: RoiData, roi2_data: Optional[RoiData], roi_config: RoiConfig):
    """
    Validate ROI1 and ROI2 coordination consistency

    Validation Checks:
    1. Timestamp correlation
    2. Gray value consistency
    3. Spatial relationship validity
    4. Data completeness
    """
    if roi2_data is None:
        logger.warning("ROI2 extraction failed, coordination incomplete")
        return

    # Timestamp correlation check (within 100ms)
    time_difference = abs((roi1_data.timestamp - roi2_data.timestamp).total_seconds())
    if time_difference > 0.1:
        logger.warning(f"ROI timestamp mismatch: {time_difference:.3f}s")

    # Gray value consistency check (reasonable difference)
    gray_difference = abs(roi1_data.gray_value - roi2_data.gray_value)
    if gray_difference > 50:  # Large difference may indicate extraction error
        logger.warning(f"Large gray value difference: {gray_difference:.1f}")

    # Spatial relationship validation
    self._validate_spatial_relationship(roi_config)
```

### 1.2 Data Flow Synchronization

#### Frame Count Synchronization

**Location**: `backend/app/core/data_store.py` (lines 150-200)

```python
class DataStore:
    def add_dual_roi_frames(self, frame_count: int, roi1_data: RoiData, roi2_data: Optional[RoiData]):
        """
        Atomically store coordinated ROI frames with frame count synchronization

        Synchronization Strategy:
        1. Atomic operation - both frames stored together
        2. Frame count correlation - linked to main processing frame
        3. Consistent timestamps - coordinated capture time
        4. Integrity validation - ensure data consistency
        """
        with self.roi_lock:
            # Create coordinated frame entry
            coordinated_frame = {
                'frame_count': frame_count,
                'timestamp': datetime.now(),
                'roi1_data': roi1_data,
                'roi2_data': roi2_data,
                'coordination_status': 'success' if roi2_data else 'partial',
                'processing_latency_ms': self._calculate_processing_latency(roi1_data, roi2_data)
            }

            # Store in synchronized buffers
            self.roi1_frames.append(coordinated_frame.copy())
            self.roi2_frames.append(coordinated_frame.copy())

            logger.debug(f"Stored coordinated ROI frames for frame {frame_count}")
```

#### Temporal Synchronization

```python
def _ensure_temporal_synchronization(self, roi1_data: RoiData, roi2_data: RoiData) -> bool:
    """
    Ensure temporal synchronization between ROI1 and ROI2

    Synchronization Criteria:
    1. Capture time difference < 100ms
    2. Processing order consistency
    3. Frame sequence integrity
    """
    time_difference = abs((roi1_data.timestamp - roi2_data.timestamp).total_seconds())

    if time_difference > 0.1:  # 100ms threshold
        logger.warning(f"ROI temporal desynchronization detected: {time_difference:.3f}s")
        return False

    return True
```

## 2. Data Flow Architecture

### 2.1 Complete Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPTURE PHASE                             │
├─────────────────────────────────────────────────────────────┤
│  Screen Environment                                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ ROI1 Region (1100x500)                                  │ │
│  │  ┌─────────────────────────────────────┐                │ │
│  │  │          ROI2 Center (50x50)        │                │ │
│  │  │                                     │                │ │
│  │  │                                     │                │ │
│  │  └─────────────────────────────────────┘                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              ↓                              │
│  PIL.ImageGrab.grab() → ROI1 Image (PIL)                   │
│                              ↓                              │
│  Image Processing → ROI1 Data Model                         │
│                              ↓                              │
│  Coordinate Transform → ROI2 Extraction                     │
│                              ↓                              │
│  Image Processing → ROI2 Data Model                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING PHASE                           │
├─────────────────────────────────────────────────────────────┤
│  Quality Validation                                         │
│  ├── Image integrity checks                                 │
│  ├── Spatial relationship validation                        │
│  ├── Temporal synchronization                               │
│  └── Error classification                                   │
│                              ↓                              │
│  Data Model Creation                                        │
│  ├── Base64 encoding                                        │
│  ├── Gray value calculation                                │
│  ├── Timestamp assignment                                   │
│  └── Metadata generation                                    │
│                              ↓                              │
│  Coordinated Storage                                        │
│  ├── Atomic frame storage                                   │
│  ├── Frame count synchronization                            │
│  ├── Circular buffer management                             │
│  └── Index generation                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 DISTRIBUTION PHASE                           │
├─────────────────────────────────────────────────────────────┤
│  API Layer (/data/dual-realtime)                             │
│  ├── Dual ROI data packaging                               │
│  ├── Metadata inclusion                                     │
│  ├── Error state reporting                                 │
│  └── Performance metrics                                    │
│                              ↓                              │
│  Multi-Client Delivery                                      │
│  ├── Frontend (Web - Single ROI only)                      │
│  ├── Python Client (Dual ROI)                              │
│  └── WebSocket Streaming                                    │
│                              ↓                              │
│  Display Rendering                                           │
│  ├── Canvas rendering (Frontend)                           │
│  ├── Tkinter display (Python Client)                       │
│  └── Error state handling                                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Model Coordination

#### Coordinated Data Structure

```python
@dataclass
class CoordinatedRoiFrame:
    """Coordinated ROI frame structure for synchronized data management"""
    frame_count: int
    capture_timestamp: datetime
    processing_timestamp: datetime

    # ROI Data
    roi1_data: RoiData
    roi2_data: Optional[RoiData]

    # Coordination Metadata
    coordination_status: str  # 'success', 'partial', 'failed'
    extraction_latency_ms: float
    spatial_relationship_valid: bool

    # Quality Metrics
    roi1_quality_score: float
    roi2_quality_score: Optional[float]

    # Error Information
    extraction_errors: List[str]
    fallback_used: bool
```

### 2.3 State Management Coordination

#### Coordinated State Machine

```python
class RoiCoordinationState(Enum):
    """ROI coordination state management"""
    INITIALIZING = "initializing"           # System starting up
    ROI1_CAPTURING = "roi1_capturing"       # ROI1 capture in progress
    ROI2_EXTRACTING = "roi2_extracting"     # ROI2 extraction in progress
    VALIDATION = "validation"               # Coordination validation
    SUCCESS = "success"                     # Both ROIs successfully processed
    PARTIAL = "partial"                     # ROI1 success, ROI2 failed
    FAILED = "failed"                       # Both ROIs failed
    ERROR_RECOVERY = "error_recovery"       # Error recovery in progress

class RoiCoordinationManager:
    """Manage ROI coordination state transitions"""

    def __init__(self):
        self.current_state = RoiCoordinationState.INITIALIZING
        self.state_history = deque(maxlen=100)
        self.error_counts = defaultdict(int)

    def transition_state(self, new_state: RoiCoordinationState, context: str = ""):
        """Manage coordinated state transitions"""
        old_state = self.current_state
        self.current_state = new_state

        # Log state transition
        logger.info(f"ROI coordination state: {old_state.value} → {new_state.value} ({context})")

        # Store in history
        self.state_history.append({
            'timestamp': datetime.now(),
            'old_state': old_state.value,
            'new_state': new_state.value,
            'context': context
        })

        # Handle state-specific logic
        self._handle_state_transition(new_state)
```

## 3. Synchronization Mechanisms

### 3.1 Frame-Level Synchronization

#### Main Processing Frame Integration

**Location**: `backend/app/core/processor.py` (integration points)

```python
class DataProcessor:
    def process_frame_with_roi_synchronization(self):
        """
        Main processing loop with ROI synchronization

        Synchronization Strategy:
        1. ROI capture synchronized with main signal processing
        2. Frame count correlation for data alignment
        3. Temporal coordination for real-time analysis
        """
        frame_count = self.frame_counter

        # Main signal processing (45 FPS)
        signal_data = self.process_signal_data()

        # ROI capture (4 FPS - every ~11 main frames)
        if frame_count % 11 == 0:  # 45/4 ≈ 11.25
            roi1_data, roi2_data = self.roi_capture_service.capture_dual_roi(
                self.current_roi_config
            )

            # Synchronized storage
            self.data_store.add_dual_roi_frames(frame_count, roi1_data, roi2_data)

            # Peak detection with ROI input
            if roi1_data and roi2_data:
                enhanced_signal = self._combine_roi_with_signal(
                    signal_data, roi1_data, roi2_data
                )
                peak_results = self.peak_detector.analyze(enhanced_signal)

        return frame_count
```

### 3.2 Temporal Synchronization

#### Timestamp Coordination

```python
def _coordinate_timestamps(self, roi1_data: RoiData, roi2_data: RoiData) -> Dict[str, datetime]:
    """
    Coordinate timestamps across ROI1 and ROI2

    Coordination Strategy:
    1. Unified capture timestamp
    2. Processing timestamp tracking
    3. Latency measurement
    4. Temporal integrity validation
    """
    capture_timestamp = roi1_data.timestamp  # Primary timestamp

    # Calculate processing latencies
    roi1_processing_time = self._calculate_processing_time(roi1_data)
    roi2_processing_time = self._calculate_processing_time(roi2_data)

    coordinated_timestamps = {
        'capture_time': capture_timestamp,
        'roi1_processing_complete': capture_timestamp + timedelta(seconds=roi1_processing_time),
        'roi2_processing_complete': capture_timestamp + timedelta(seconds=roi2_processing_time),
        'coordination_complete': datetime.now(),
        'total_latency_ms': (datetime.now() - capture_timestamp).total_seconds() * 1000
    }

    return coordinated_timestamps
```

### 3.3 Spatial Coordination

#### Coordinate System Transformation

```python
def _coordinate_spatial_relationships(self, roi1_config: RoiConfig, roi2_data: RoiData) -> Dict[str, Any]:
    """
    Coordinate spatial relationships between ROI1 and ROI2

    Spatial Coordination:
    1. Screen space coordinates
    2. ROI1 image space coordinates
    3. ROI2 extraction coordinates
    4. Display transformation coordinates
    """
    # Screen space coordinates
    roi1_screen_coords = {
        'x1': roi1_config.x1, 'y1': roi1_config.y1,
        'x2': roi1_config.x2, 'y2': roi1_config.y2
    }

    # ROI1 center in screen space
    roi1_center_screen = {
        'x': (roi1_config.x1 + roi1_config.x2) // 2,
        'y': (roi1_config.y1 + roi1_config.y2) // 2
    }

    # ROI2 coordinates in screen space (50x50 center)
    roi2_screen_coords = {
        'x1': roi1_center_screen['x'] - 25,
        'y1': roi1_center_screen['y'] - 25,
        'x2': roi1_center_screen['x'] + 25,
        'y2': roi1_center_screen['y'] + 25
    }

    # Validate ROI2 within ROI1 boundaries
    spatial_validation = self._validate_spatial_boundaries(
        roi1_screen_coords, roi2_screen_coords
    )

    return {
        'roi1_screen': roi1_screen_coords,
        'roi1_center': roi1_center_screen,
        'roi2_screen': roi2_screen_coords,
        'spatial_validation': spatial_validation,
        'coordinate_system': 'screen_pixels'
    }
```

## 4. Error Handling Coordination

### 4.1 Coordinated Error Classification

#### Error Impact Assessment

```python
class RoiCoordinationError(Exception):
    """Base class for ROI coordination errors"""

    def __init__(self, message: str, error_type: str, impact_level: str, recovery_strategy: str):
        self.message = message
        self.error_type = error_type
        self.impact_level = impact_level  # 'low', 'medium', 'high', 'critical'
        self.recovery_strategy = recovery_strategy
        super().__init__(message)

class RoiCoordinationErrorHandler:
    """Handle coordinated error scenarios with intelligent recovery"""

    def handle_coordination_error(self, error: RoiCoordinationError, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle coordination errors with appropriate recovery strategies

        Error Categories:
        1. ROI1 Capture Failures - Critical (affects both ROIs)
        2. ROI2 Extraction Failures - Medium (ROI1 still usable)
        3. Synchronization Failures - Low (data quality issue)
        4. System Resource Errors - Critical (system stability)
        """
        error_response = {
            'error_type': error.error_type,
            'impact_level': error.impact_level,
            'recovery_action': error.recovery_strategy,
            'timestamp': datetime.now(),
            'context': context
        }

        # Implement recovery strategy
        if error.recovery_strategy == 'roi2_fallback':
            error_response.update(self._implement_roi2_fallback(context))
        elif error.recovery_strategy == 'retry_capture':
            error_response.update(self._implement_retry_capture(context))
        elif error.recovery_strategy == 'graceful_degradation':
            error_response.update(self._implement_graceful_degradation(context))

        # Log error for monitoring
        self._log_coordination_error(error, error_response)

        return error_response
```

### 4.2 Fallback Coordination

#### Intelligent Fallback Strategies

```python
def _coordinate_fallback_strategy(self, roi1_data: Optional[RoiData], roi2_data: Optional[RoiData]) -> Dict[str, Any]:
    """
    Coordinate fallback strategies for partial ROI failures

    Fallback Hierarchy:
    1. Both ROIs successful → Normal operation
    2. ROI1 success, ROI2 failed → ROI2 fallback using ROI1 data
    3. ROI1 failed → System-level fallback
    4. Both failed → Emergency mode with simulated data
    """
    if roi1_data and roi2_data:
        return {
            'status': 'success',
            'strategy': 'normal_operation',
            'data_quality': 'high'
        }

    elif roi1_data and not roi2_data:
        # ROI2 fallback using ROI1 data
        fallback_roi2 = self._create_roi2_fallback(roi1_data)

        return {
            'status': 'partial_success',
            'strategy': 'roi2_fallback',
            'data_quality': 'medium',
            'roi1_data': roi1_data,
            'roi2_data': fallback_roi2,
            'warning': 'ROI2 extraction failed, using ROI1-based fallback'
        }

    elif not roi1_data and roi2_data:
        # Rare case: ROI1 failed but ROI2 cache available
        logger.warning("ROI1 failed but ROI2 available - unusual scenario")

        return {
            'status': 'partial_success',
            'strategy': 'roi2_cache_usage',
            'data_quality': 'low',
            'roi1_data': None,
            'roi2_data': roi2_data,
            'warning': 'ROI1 failed, using cached ROI2 data'
        }

    else:
        # Complete failure - emergency fallback
        emergency_data = self._create_emergency_fallback()

        return {
            'status': 'emergency_mode',
            'strategy': 'simulated_data',
            'data_quality': 'simulation',
            'roi1_data': emergency_data['roi1'],
            'roi2_data': emergency_data['roi2'],
            'error': 'Complete ROI capture failure, using simulated data'
        }
```

## 5. Performance Coordination

### 5.1 Resource Management Coordination

#### Memory Coordination

```python
class RoiMemoryCoordinator:
    """Coordinate memory usage across ROI processing"""

    def __init__(self, max_memory_mb: int = 100):
        self.max_memory_mb = max_memory_mb
        self.current_memory_usage = 0
        self.memory_allocation_strategy = 'adaptive'

    def coordinate_memory_allocation(self, roi1_size: int, roi2_size: int) -> Dict[str, int]:
        """
        Coordinate memory allocation for ROI processing

        Memory Strategy:
        1. ROI1 buffer allocation (70% of budget)
        2. ROI2 buffer allocation (20% of budget)
        3. Processing overhead (10% of budget)
        4. Adaptive scaling based on available memory
        """
        available_memory = self.max_memory_mb * 1024 * 1024  # Convert to bytes

        # Calculate memory requirements
        roi1_memory = roi1_size * 3  # RGB buffer
        roi2_memory = roi2_size * 3  # RGB buffer
        processing_overhead = available_memory * 0.1

        total_required = roi1_memory + roi2_memory + processing_overhead

        if total_required <= available_memory:
            allocation = {
                'roi1_buffer_size': roi1_size,
                'roi2_buffer_size': roi2_size,
                'processing_buffer': int(processing_overhead),
                'strategy': 'normal_allocation'
            }
        else:
            # Adaptive scaling
            scale_factor = (available_memory * 0.9) / total_required
            allocation = {
                'roi1_buffer_size': int(roi1_size * scale_factor),
                'roi2_buffer_size': int(roi2_size * scale_factor),
                'processing_buffer': int(processing_overhead * scale_factor),
                'strategy': 'adaptive_scaling',
                'scale_factor': scale_factor
            }

        self.current_memory_usage = sum([
            allocation['roi1_buffer_size'] * 3,
            allocation['roi2_buffer_size'] * 3,
            allocation['processing_buffer']
        ])

        return allocation
```

### 5.2 Processing Performance Coordination

#### Throughput Balancing

```python
def coordinate_roi_throughput(self, system_load: float, target_fps: int) -> Dict[str, float]:
    """
    Coordinate ROI processing throughput based on system conditions

    Throughput Coordination:
    1. Adaptive frame rate adjustment
    2. Quality vs performance balancing
    3. Resource usage optimization
    4. Load-aware processing
    """
    if system_load < 0.5:  # Low load
        return {
            'roi1_fps': target_fps,
            'roi2_fps': target_fps,
            'quality_level': 'high',
            'processing_priority': 'accuracy'
        }

    elif system_load < 0.8:  # Medium load
        return {
            'roi1_fps': target_fps * 0.8,
            'roi2_fps': target_fps * 0.6,  # ROI2 can be lower priority
            'quality_level': 'medium',
            'processing_priority': 'balanced'
        }

    else:  # High load
        return {
            'roi1_fps': target_fps * 0.5,
            'roi2_fps': target_fps * 0.3,
            'quality_level': 'low',
            'processing_priority': 'performance'
        }
```

## 6. Client Distribution Coordination

### 6.1 Multi-Client Data Coordination

#### API Response Coordination

**Location**: `backend/app/api/routes.py` (dual-realtime endpoint)

```python
@router.get("/data/dual-realtime", response_model=DualRealtimeDataResponse)
async def dual_realtime_data(count: int = Query(100, ge=1, le=1000)):
    """
    Coordinated dual ROI data delivery for multi-client support

    Coordination Features:
    1. Consistent data formatting across clients
    2. Coordinated error state reporting
    3. Performance metrics inclusion
    4. Client capability adaptation
    """
    try:
        # Get coordinated ROI data
        coordinated_frames = data_store.get_latest_coordinated_frames(count)

        # Build coordinated response
        coordinated_response = {
            # ROI Data
            'roi1_data': _extract_roi1_series(coordinated_frames),
            'roi2_data': _extract_roi2_series(coordinated_frames),

            # Coordination Metadata
            'coordination_info': {
                'total_frames': len(coordinated_frames),
                'successful_extractions': count_successful_extractions(coordinated_frames),
                'average_latency_ms': calculate_average_latency(coordinated_frames),
                'quality_distribution': calculate_quality_distribution(coordinated_frames)
            },

            # Configuration Information
            'roi_configs': {
                'roi1_config': get_current_roi1_config(),
                'roi2_coordinates': calculate_roi2_coordinates(get_current_roi1_config())
            },

            # Signal Data (synchronized)
            'signal_data': extract_synchronized_signal_data(coordinated_frames),
            'peak_data': extract_synchronized_peak_data(coordinated_frames),

            # System Status
            'system_status': {
                'coordination_health': assess_coordination_health(coordinated_frames),
                'performance_metrics': get_roi_performance_metrics(),
                'error_rates': calculate_coordination_error_rates(coordinated_frames)
            }
        }

        return DualRealtimeDataResponse(**coordinated_response)

    except Exception as e:
        logger.error(f"Coordinated dual ROI data delivery failed: {e}")
        return create_coordinated_error_response(e)
```

### 6.2 Client Capability Coordination

#### Frontend vs Python Client Coordination

```python
def coordinate_client_capabilities(self) -> Dict[str, Any]:
    """
    Coordinate data delivery based on client capabilities

    Client Capability Matrix:
    1. Frontend: Single ROI display, lower bandwidth
    2. Python Client: Dual ROI display, full bandwidth
    3. WebSocket: Real-time streaming, low latency
    4. API clients: Batch processing, high accuracy
    """
    return {
        'frontend_capabilities': {
            'supported_rois': ['roi1'],
            'display_format': 'canvas',
            'update_rate': '20fps',
            'bandwidth_optimization': True
        },

        'python_client_capabilities': {
            'supported_rois': ['roi1', 'roi2'],
            'display_format': 'tkinter',
            'update_rate': 'real_time',
            'bandwidth_optimization': False,
            'advanced_features': True
        },

        'coordination_strategy': {
            'frontend': 'roi1_only_delivery',
            'python_client': 'dual_roi_delivery',
            'websocket': 'streaming_updates',
            'api_clients': 'batch_delivery'
        }
    }
```

## 7. Monitoring and Diagnostics

### 7.1 Coordination Health Monitoring

#### Health Metrics Collection

```python
def collect_coordination_health_metrics(self) -> Dict[str, Any]:
    """
    Collect comprehensive coordination health metrics

    Health Monitoring Categories:
    1. Synchronization accuracy
    2. Error rate analysis
    3. Performance degradation
    4. Resource utilization
    """
    recent_frames = self.data_store.get_recent_coordinated_frames(100)

    return {
        'synchronization_health': {
            'temporal_alignment_score': self._calculate_temporal_alignment(recent_frames),
            'spatial_consistency_score': self._calculate_spatial_consistency(recent_frames),
            'frame_synchronization_rate': self._calculate_frame_sync_rate(recent_frames)
        },

        'error_analysis': {
            'roi1_success_rate': self._calculate_roi1_success_rate(recent_frames),
            'roi2_success_rate': self._calculate_roi2_success_rate(recent_frames),
            'coordination_success_rate': self._calculate_coordination_success_rate(recent_frames),
            'error_recovery_rate': self._calculate_error_recovery_rate(recent_frames)
        },

        'performance_metrics': {
            'average_capture_latency_ms': self._calculate_average_capture_latency(recent_frames),
            'average_processing_latency_ms': self._calculate_average_processing_latency(recent_frames),
            'throughput_fps': self._calculate_actual_throughput(recent_frames),
            'memory_utilization_mb': self._calculate_memory_utilization()
        },

        'resource_health': {
            'cpu_utilization': psutil.cpu_percent(),
            'memory_utilization': psutil.virtual_memory().percent,
            'disk_io_utilization': psutil.disk_usage('/').percent if os.name != 'nt' else 0,
            'thread_safety_score': self._assess_thread_safety()
        }
    }
```

### 7.2 Diagnostic Tools

#### Coordination Diagnostics

```python
def run_coordination_diagnostics(self) -> Dict[str, Any]:
    """
    Run comprehensive coordination diagnostics

    Diagnostic Categories:
    1. Data integrity validation
    2. Performance bottleneck identification
    3. Configuration consistency checking
    4. Error pattern analysis
    """
    diagnostics = {
        'data_integrity': self._validate_data_integrity(),
        'performance_analysis': self._analyze_performance_bottlenecks(),
        'configuration_validation': self._validate_configuration_consistency(),
        'error_pattern_analysis': self._analyze_error_patterns(),
        'recommendations': []
    }

    # Generate recommendations based on diagnostics
    diagnostics['recommendations'] = self._generate_coordination_recommendations(diagnostics)

    return diagnostics
```

This comprehensive documentation covers the complete coordination and data flow architecture of the NHEM dual ROI system, providing detailed technical guidance for understanding and maintaining the sophisticated synchronization mechanisms.