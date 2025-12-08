"""
API Configuration Module
This module contains all API-related configuration constants and settings.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
from enum import Enum

class APIEnvironment(Enum):
    """API deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

# Default API Endpoints
DEFAULT_ENDPOINTS = {
    'base_url': 'http://localhost:8421',
    'health_check': '/health',
    'system_status': '/status',
    'dual_realtime': '/data/dual-realtime',
    'enhanced_dual_realtime': '/data/dual-realtime/enhanced',
    'line_intersection': '/api/roi/line-intersection',
    'line_intersection_stats': '/api/roi/line-intersection/stats',
    'detection_status': '/api/detection/status',
    'start_detection': '/api/detection/start',
    'stop_detection': '/api/detection/stop',
    'reset_detection': '/api/detection/reset',
    'roi_config': '/api/roi/config',
    'update_roi': '/api/roi/config/update',
    'screenshot': '/api/roi/screenshot',
    'peak_data': '/data/peaks',
    'realtime_data': '/data/realtime',
    'calibration': '/api/calibration',
}

# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    APIEnvironment.DEVELOPMENT: {
        'base_url': 'http://localhost:8421',
        'timeout': 30.0,
        'retry_attempts': 3,
        'retry_delay': 1.0,
        'verify_ssl': False,
        'debug_mode': True,
    },
    APIEnvironment.TESTING: {
        'base_url': 'http://test-server:8421',
        'timeout': 15.0,
        'retry_attempts': 2,
        'retry_delay': 0.5,
        'verify_ssl': False,
        'debug_mode': True,
    },
    APIEnvironment.STAGING: {
        'base_url': 'https://staging-server:8421',
        'timeout': 10.0,
        'retry_attempts': 2,
        'retry_delay': 1.0,
        'verify_ssl': True,
        'debug_mode': False,
    },
    APIEnvironment.PRODUCTION: {
        'base_url': 'https://production-server:8421',
        'timeout': 5.0,
        'retry_attempts': 1,
        'retry_delay': 0.5,
        'verify_ssl': True,
        'debug_mode': False,
    }
}

# Request Configuration
REQUEST_CONFIG = {
    'default_timeout': 5.0,
    'connection_timeout': 2.0,
    'read_timeout': 5.0,
    'max_retries': 3,
    'retry_delay': 1.0,
    'retry_backoff_factor': 2.0,
    'status_codes_to_retry': [408, 429, 500, 502, 503, 504],
    'max_redirects': 5,
}

# Header Configuration
DEFAULT_HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'User-Agent': 'NHEM-LineDetection/1.0',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

# Authentication Configuration
AUTH_CONFIG = {
    'password_header': 'X-Detection-Password',
    'default_password': '31415',
    'api_key_header': 'X-API-Key',
    'token_header': 'Authorization',
    'auth_bearer_prefix': 'Bearer',
}

# Data Request Parameters
DATA_REQUEST_CONFIG = {
    'default_count': 100,
    'max_count': 1000,
    'min_count': 1,
    'count_step': 10,
    'default_roi_count': 2,
    'max_roi_count': 4,
    'data_format': 'json',
    'include_metadata': True,
    'include_timestamps': True,
}

# Line Intersection Detection Configuration
LINE_DETECTION_CONFIG = {
    'default_threshold': 104.0,
    'default_margin_frames': 5,
    'default_difference_threshold': 1.1,
    'default_min_region_length': 5,
    'max_processing_time': 10.0,  # seconds
    'batch_size': 10,
    'enable_validation': True,
    'enable_statistics': True,
}

# ROI Configuration
ROI_CONFIG = {
    'default_width': 100,
    'default_height': 100,
    'min_width': 10,
    'min_height': 10,
    'max_width': 1000,
    'max_height': 1000,
    'default_x1': 1480,
    'default_y1': 480,
    'default_x2': 1580,
    'default_y2': 580,
    'coordinate_precision': 2,
    'enable_snapping': True,
    'snap_threshold': 5.0,
}

# WebSocket Configuration
WEBSOCKET_CONFIG = {
    'default_port': 30415,
    'path': '/ws',
    'ping_interval': 20,
    'ping_timeout': 10,
    'close_timeout': 10,
    'max_queue': 32,
    'compression': None,
    'permessage_deflate': False,
}

# Response Validation Configuration
RESPONSE_VALIDATION = {
    'required_fields': {
        'dual_realtime': ['success', 'data'],
        'line_intersection': ['success', 'intersections'],
        'health_check': ['status', 'timestamp'],
        'system_status': ['status', 'data'],
    },
    'data_types': {
        'timestamp': (int, float, str),
        'confidence': (int, float),
        'coordinates': (list, tuple),
        'roi_data': dict,
    },
    'coordinate_constraints': {
        'min_x': 0,
        'min_y': 0,
        'max_x': 10000,
        'max_y': 10000,
    },
}

# Error Handling Configuration
ERROR_HANDLING = {
    'log_errors': True,
    'log_level': 'WARNING',
    'raise_on_error': False,
    'error_retry_limit': 3,
    'circuit_breaker_threshold': 5,
    'circuit_breaker_timeout': 60,  # seconds
    'fallback_to_cache': True,
    'cache_ttl': 300,  # seconds
}

# Cache Configuration
CACHE_CONFIG = {
    'enable_cache': True,
    'default_ttl': 60,  # seconds
    'max_cache_size': 1000,
    'cache_key_prefix': 'nhem_ld_',
    'cache_invalidations': {
        'line_intersection': ['roi_config_update', 'detection_start', 'detection_stop'],
        'dual_realtime': ['data_update', 'processing_change'],
    },
}

# Rate Limiting Configuration
RATE_LIMITING = {
    'enable_rate_limiting': True,
    'requests_per_minute': 100,
    'requests_per_hour': 1000,
    'burst_size': 10,
    'rate_limit_headers': True,
    'rate_limit_response': {
        'status': 429,
        'message': 'Rate limit exceeded',
        'retry_after': 60,
    }
}

# Monitoring and Metrics Configuration
MONITORING_CONFIG = {
    'enable_metrics': True,
    'metrics_interval': 60,  # seconds
    'track_response_times': True,
    'track_error_rates': True,
    'track_request_counts': True,
    'health_check_interval': 30,  # seconds
    'alert_threshold': {
        'response_time': 5.0,  # seconds
        'error_rate': 0.05,    # 5%
        'unavailability': 0.01,  # 1%
    }
}

# Development and Debug Configuration
DEBUG_CONFIG = {
    'enable_request_logging': False,
    'enable_response_logging': False,
    'log_request_body': False,
    'log_response_body': False,
    'pretty_print_json': True,
    'save_raw_responses': False,
    'mock_mode': False,
    'mock_data_file': 'mock_api_responses.json',
}

# Environment Variable Configuration
ENV_VAR_MAPPING = {
    'NHEM_API_BASE_URL': ('base_url', str),
    'NHEM_API_PASSWORD': ('password', str),
    'NHEM_API_TIMEOUT': ('timeout', float),
    'NHEM_API_DEBUG': ('debug_mode', bool),
    'NHEM_API_ENVIRONMENT': ('environment', str),
    'NHEM_API_SSL_VERIFY': ('verify_ssl', bool),
}

def get_api_config(environment: APIEnvironment = APIEnvironment.DEVELOPMENT,
                  **overrides) -> Dict[str, Any]:
    """
    Get API configuration for a specific environment with optional overrides.

    Args:
        environment: Target environment
        **overrides: Configuration overrides

    Returns:
        Complete API configuration dictionary
    """
    config = ENVIRONMENT_CONFIGS.get(environment, ENVIRONMENT_CONFIGS[APIEnvironment.DEVELOPMENT]).copy()

    # Add endpoint configuration
    config['endpoints'] = DEFAULT_ENDPOINTS.copy()

    # Add other configurations
    config['headers'] = DEFAULT_HEADERS.copy()
    config['auth'] = AUTH_CONFIG.copy()
    config['request'] = REQUEST_CONFIG.copy()
    config['data'] = DATA_REQUEST_CONFIG.copy()
    config['line_detection'] = LINE_DETECTION_CONFIG.copy()
    config['roi'] = ROI_CONFIG.copy()
    config['websocket'] = WEBSOCKET_CONFIG.copy()
    config['validation'] = RESPONSE_VALIDATION.copy()
    config['error_handling'] = ERROR_HANDLING.copy()
    config['cache'] = CACHE_CONFIG.copy()
    config['rate_limiting'] = RATE_LIMITING.copy()
    config['monitoring'] = MONITORING_CONFIG.copy()
    config['debug'] = DEBUG_CONFIG.copy()

    # Apply environment variable overrides
    config.update(_load_environment_overrides())

    # Apply function parameter overrides
    config.update(overrides)

    return config

def _load_environment_overrides() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    overrides = {}

    for env_var, (config_key, value_type) in ENV_VAR_MAPPING.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                if value_type == bool:
                    overrides[config_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif value_type == int:
                    overrides[config_key] = int(env_value)
                elif value_type == float:
                    overrides[config_key] = float(env_value)
                else:
                    overrides[config_key] = env_value
            except (ValueError, TypeError):
                # Skip invalid environment variable values
                continue

    return overrides

def get_endpoint_url(base_url: str, endpoint_key: str, environment: APIEnvironment = APIEnvironment.DEVELOPMENT) -> str:
    """
    Construct full endpoint URL.

    Args:
        base_url: Base API URL
        endpoint_key: Key from DEFAULT_ENDPOINTS
        environment: Target environment

    Returns:
        Full endpoint URL
    """
    endpoints = DEFAULT_ENDPOINTS
    endpoint_path = endpoints.get(endpoint_key)

    if not endpoint_path:
        raise ValueError(f"Unknown endpoint key: {endpoint_key}")

    # Ensure base_url doesn't have trailing slash and endpoint doesn't have leading slash
    base_url = base_url.rstrip('/')
    endpoint_path = endpoint_path.lstrip('/')

    return f"{base_url}/{endpoint_path}"

def validate_api_response(response_data: Dict[str, Any], endpoint_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate API response structure and data.

    Args:
        response_data: Response data to validate
        endpoint_type: Type of endpoint for validation rules

    Returns:
        Tuple of (is_valid, error_message)
    """
    validation_config = RESPONSE_VALIDATION

    # Check required fields
    required_fields = validation_config['required_fields'].get(endpoint_type, [])
    for field in required_fields:
        if field not in response_data:
            return False, f"Missing required field: {field}"

    # Validate data types
    for field, expected_types in validation_config['data_types'].items():
        if field in response_data and not isinstance(response_data[field], expected_types):
            return False, f"Invalid type for field {field}: expected {expected_types}, got {type(response_data[field])}"

    return True, None