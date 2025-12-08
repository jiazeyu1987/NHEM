"""
API Integration Component
This module handles all external API communications for line detection operations.
"""

import logging
import requests
import base64
import json
import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import traceback

from ..config import get_api_config, APIEnvironment

logger = logging.getLogger(__name__)

class APIStatus(Enum):
    """API connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

@dataclass
class APIRequest:
    """Represents an API request."""
    method: str
    endpoint: str
    data: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: float = 5.0

@dataclass
class APIResponse:
    """Represents an API response."""
    success: bool
    status_code: int
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    response_time: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ROIConfiguration:
    """ROI configuration for detection."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence_threshold: float = 0.5
    line_length_threshold: float = 10.0
    enable_intersection_detection: bool = True

class APIIntegration:
    """
    Handles API communications for line detection operations.
    Provides a clean interface for external API calls with error handling and retry logic.
    """

    def __init__(self, base_url: str = None, config: Dict[str, Any] = None):
        """
        Initialize the API integration.

        Args:
            base_url: Base URL for the API
            config: Configuration dictionary
        """
        self.config = get_api_config(APIEnvironment.DEVELOPMENT, **(config or {}))
        self.base_url = base_url or self.config.get('base_url', 'http://localhost:8421')

        # Connection state
        self.api_status: APIStatus = APIStatus.DISCONNECTED
        self.session: Optional[requests.Session] = None
        self.connection_retries: int = 0
        self.last_connection_attempt: Optional[datetime] = None
        self.last_successful_request: Optional[datetime] = None

        # Request configuration
        self.default_headers = self.config.get('headers', {}).copy()
        self.request_timeout = self.config.get('request', {}).get('default_timeout', 5.0)
        self.max_retries = self.config.get('request', {}).get('max_retries', 3)
        self.retry_delay = self.config.get('request', {}).get('retry_delay', 1.0)

        # Authentication
        self.auth_config = self.config.get('auth', {})
        self.detection_password = self.auth_config.get('default_password', '31415')

        # API endpoints
        self.endpoints = self.config.get('endpoints', {})

        # Statistics and monitoring
        self.request_count: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.total_response_time: float = 0.0
        self.error_history: List[str] = []

        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'status_changed': [],
            'request_completed': [],
            'error_occurred': [],
        }

        # Threading
        self.request_lock = threading.Lock()
        self.background_requests: Dict[str, threading.Thread] = {}

        logger.info(f"APIIntegration initialized with base_url: {self.base_url}")

    def connect(self) -> bool:
        """
        Establish connection to the API.

        Returns:
            True if connection was successful
        """
        try:
            with self.request_lock:
                if self.api_status == APIStatus.CONNECTED:
                    return True

                self.api_status = APIStatus.CONNECTING
                self._trigger_callbacks('status_changed', self.api_status)

                # Create session
                self.session = requests.Session()
                self.session.headers.update(self.default_headers)

                # Test connection with health check
                health_response = self._make_request(
                    method='GET',
                    endpoint=self.endpoints.get('health_check', '/health'),
                    timeout=self.request_timeout
                )

                if health_response.success:
                    self.api_status = APIStatus.CONNECTED
                    self.last_successful_request = datetime.now()
                    self.connection_retries = 0

                    logger.info("Successfully connected to API")
                    self._trigger_callbacks('status_changed', self.api_status)
                    return True
                else:
                    self.api_status = APIStatus.ERROR
                    self._track_error(f"Health check failed: {health_response.error_message}")
                    return False

        except Exception as e:
            self.api_status = APIStatus.ERROR
            self._track_error(f"Connection error: {str(e)}")
            logger.error(f"Error connecting to API: {e}")
            return False

    def disconnect(self):
        """Close the API connection."""
        try:
            with self.request_lock:
                if self.session:
                    self.session.close()
                    self.session = None

                # Cancel background requests
                for request_id, thread in self.background_requests.items():
                    if thread.is_alive():
                        logger.warning(f"Background request {request_id} is still running")

                self.background_requests.clear()
                self.api_status = APIStatus.DISCONNECTED

                logger.info("Disconnected from API")
                self._trigger_callbacks('status_changed', self.api_status)

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def manual_detection(self, roi_coordinates: Optional[Dict[str, Any]] = None,
                        force_refresh: bool = False) -> APIResponse:
        """
        Trigger manual line detection.

        Args:
            roi_coordinates: ROI coordinates for detection
            force_refresh: Whether to force refresh of data

        Returns:
            API response with detection results
        """
        try:
            # Prepare request data
            request_data = {
                'force_refresh': force_refresh,
                'timestamp': datetime.now().isoformat()
            }

            if roi_coordinates:
                request_data['roi_coordinates'] = roi_coordinates

            # Make API request
            response = self._make_request(
                method='POST',
                endpoint=self.endpoints.get('line_intersection', '/api/roi/line-intersection'),
                data=request_data
            )

            if response.success:
                logger.info("Manual detection completed successfully")
            else:
                logger.warning(f"Manual detection failed: {response.error_message}")

            return response

        except Exception as e:
            logger.error(f"Error in manual detection: {e}")
            return APIResponse(
                success=False,
                status_code=500,
                error_message=str(e)
            )

    def get_dual_realtime_data(self, count: int = 100) -> APIResponse:
        """
        Get dual realtime data from the API.

        Args:
            count: Number of data points to retrieve

        Returns:
            API response with realtime data
        """
        try:
            params = {'count': min(count, self.config.get('data', {}).get('max_count', 1000))}

            response = self._make_request(
                method='GET',
                endpoint=self.endpoints.get('dual_realtime', '/data/dual-realtime'),
                params=params
            )

            if response.success:
                logger.debug(f"Retrieved dual realtime data: {count} points")
            else:
                logger.warning(f"Failed to get dual realtime data: {response.error_message}")

            return response

        except Exception as e:
            logger.error(f"Error getting dual realtime data: {e}")
            return APIResponse(
                success=False,
                status_code=500,
                error_message=str(e)
            )

    def get_enhanced_realtime_data(self, count: int = 100) -> APIResponse:
        """
        Get enhanced realtime data from the API.

        Args:
            count: Number of data points to retrieve

        Returns:
            API response with enhanced realtime data
        """
        try:
            params = {'count': min(count, self.config.get('data', {}).get('max_count', 1000))}

            response = self._make_request(
                method='GET',
                endpoint=self.endpoints.get('enhanced_dual_realtime', '/data/dual-realtime/enhanced'),
                params=params
            )

            if response.success:
                logger.debug(f"Retrieved enhanced realtime data: {count} points")
            else:
                logger.warning(f"Failed to get enhanced realtime data: {response.error_message}")

            return response

        except Exception as e:
            logger.error(f"Error getting enhanced realtime data: {e}")
            return APIResponse(
                success=False,
                status_code=500,
                error_message=str(e)
            )

    def get_detection_status(self) -> APIResponse:
        """
        Get current detection status from the API.

        Returns:
            API response with status information
        """
        try:
            response = self._make_request(
                method='GET',
                endpoint=self.endpoints.get('detection_status', '/api/detection/status')
            )

            return response

        except Exception as e:
            logger.error(f"Error getting detection status: {e}")
            return APIResponse(
                success=False,
                status_code=500,
                error_message=str(e)
            )

    def update_roi_configuration(self, roi_config: ROIConfiguration) -> APIResponse:
        """
        Update ROI configuration on the server.

        Args:
            roi_config: ROI configuration to update

        Returns:
            API response
        """
        try:
            response = self._make_request(
                method='POST',
                endpoint=self.endpoints.get('update_roi', '/api/roi/config/update'),
                data=asdict(roi_config)
            )

            if response.success:
                logger.info("ROI configuration updated successfully")
            else:
                logger.warning(f"Failed to update ROI configuration: {response.error_message}")

            return response

        except Exception as e:
            logger.error(f"Error updating ROI configuration: {e}")
            return APIResponse(
                success=False,
                status_code=500,
                error_message=str(e)
            )

    def _make_request(self, method: str, endpoint: str,
                     data: Optional[Dict[str, Any]] = None,
                     params: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     timeout: Optional[float] = None) -> APIResponse:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            params: URL parameters
            headers: Additional headers
            timeout: Request timeout

        Returns:
            API response
        """
        start_time = time.time()

        try:
            # Ensure connection
            if self.api_status != APIStatus.CONNECTED:
                if not self.connect():
                    return APIResponse(
                        success=False,
                        status_code=0,
                        error_message="Failed to connect to API"
                    )

            # Prepare request
            url = self.base_url.rstrip('/') + '/' + endpoint.lstrip('/')
            request_timeout = timeout or self.request_timeout

            # Merge headers
            request_headers = self.default_headers.copy()
            if headers:
                request_headers.update(headers)

            # Add authentication
            if self.detection_password:
                request_headers[self.auth_config.get('password_header', 'X-Detection-Password')] = self.detection_password

            # Make request with retry logic
            response_data = None
            status_code = 500
            error_message = None

            for attempt in range(self.max_retries + 1):
                try:
                    if method.upper() == 'GET':
                        response = self.session.get(
                            url,
                            params=params,
                            headers=request_headers,
                            timeout=request_timeout
                        )
                    elif method.upper() == 'POST':
                        response = self.session.post(
                            url,
                            json=data,
                            params=params,
                            headers=request_headers,
                            timeout=request_timeout
                        )
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                    status_code = response.status_code
                    response_time = time.time() - start_time

                    # Check response status
                    if response.status_code == 200:
                        try:
                            response_data = response.json()
                        except ValueError:
                            response_data = {'raw_response': response.text}

                        # Update statistics
                        self.request_count += 1
                        self.successful_requests += 1
                        self.total_response_time += response_time
                        self.last_successful_request = datetime.now()

                        # Create successful response
                        api_response = APIResponse(
                            success=True,
                            status_code=status_code,
                            data=response_data,
                            response_time=response_time
                        )

                        self._trigger_callbacks('request_completed', api_response)
                        return api_response

                    elif response.status_code in [429, 500, 502, 503, 504]:
                        # Retry on these status codes
                        if attempt < self.max_retries:
                            time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                            continue

                    # Non-retryable error
                    error_message = f"HTTP {status_code}: {response.text}"
                    break

                except requests.exceptions.Timeout:
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    error_message = "Request timeout"
                    break

                except requests.exceptions.ConnectionError:
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    error_message = "Connection error"
                    break

                except Exception as e:
                    error_message = f"Request error: {str(e)}"
                    break

            # If we get here, the request failed
            response_time = time.time() - start_time
            self.request_count += 1
            self.failed_requests += 1
            self._track_error(error_message)

            api_response = APIResponse(
                success=False,
                status_code=status_code,
                error_message=error_message,
                response_time=response_time
            )

            self._trigger_callbacks('request_completed', api_response)
            return api_response

        except Exception as e:
            response_time = time.time() - start_time
            error_message = f"Request preparation error: {str(e)}"
            self._track_error(error_message)

            api_response = APIResponse(
                success=False,
                status_code=0,
                error_message=error_message,
                response_time=response_time
            )

            self._trigger_callbacks('error_occurred', e)
            return api_response

    def _track_error(self, error_message: str):
        """Track error occurrences."""
        self.error_history.append(f"{datetime.now().isoformat()}: {error_message}")

        # Maintain error history size
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-50:]

        logger.error(f"API Error: {error_message}")

    def async_request(self, request_id: str, method: str, endpoint: str,
                      data: Optional[Dict[str, Any]] = None,
                      callback: Optional[Callable] = None) -> bool:
        """
        Make an asynchronous API request.

        Args:
            request_id: Unique identifier for the request
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            callback: Callback function for completion

        Returns:
            True if request was started successfully
        """
        try:
            # Cancel existing request with same ID
            if request_id in self.background_requests:
                if self.background_requests[request_id].is_alive():
                    logger.warning(f"Cancelling existing background request: {request_id}")
                del self.background_requests[request_id]

            # Start new request thread
            thread = threading.Thread(
                target=self._execute_async_request,
                args=(request_id, method, endpoint, data, callback),
                daemon=True
            )

            self.background_requests[request_id] = thread
            thread.start()

            logger.info(f"Started async request: {request_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting async request {request_id}: {e}")
            return False

    def _execute_async_request(self, request_id: str, method: str, endpoint: str,
                              data: Optional[Dict[str, Any]], callback: Optional[Callable]):
        """Execute asynchronous request in background thread."""
        try:
            response = self._make_request(method, endpoint, data)

            # Call callback if provided
            if callback:
                try:
                    callback(response)
                except Exception as e:
                    logger.error(f"Error in async request callback: {e}")

        except Exception as e:
            logger.error(f"Error in async request {request_id}: {e}")
            if callback:
                error_response = APIResponse(
                    success=False,
                    status_code=0,
                    error_message=str(e)
                )
                try:
                    callback(error_response)
                except Exception as cb_e:
                    logger.error(f"Error in error callback: {cb_e}")

        finally:
            # Clean up
            if request_id in self.background_requests:
                del self.background_requests[request_id]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get API statistics.

        Returns:
            Dictionary with API statistics
        """
        success_rate = 0.0
        if self.request_count > 0:
            success_rate = (self.successful_requests / self.request_count) * 100

        avg_response_time = 0.0
        if self.successful_requests > 0:
            avg_response_time = self.total_response_time / self.successful_requests

        return {
            'api_status': self.api_status.value,
            'base_url': self.base_url,
            'request_count': self.request_count,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'average_response_time': avg_response_time,
            'total_response_time': self.total_response_time,
            'last_successful_request': self.last_successful_request.isoformat() if self.last_successful_request else None,
            'connection_retries': self.connection_retries,
            'active_background_requests': len(self.background_requests),
            'recent_errors': self.error_history[-10:],  # Last 10 errors
        }

    def add_callback(self, event_type: str, callback: Callable):
        """
        Add a callback for API events.

        Args:
            event_type: Type of event ('status_changed', 'request_completed', etc.)
            callback: Callback function to call
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown callback event type: {event_type}")

    def remove_callback(self, event_type: str, callback: Callable):
        """
        Remove a callback for API events.

        Args:
            event_type: Type of event
            callback: Callback function to remove
        """
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)

    def _trigger_callbacks(self, event_type: str, *args, **kwargs):
        """Trigger all callbacks for a specific event type."""
        try:
            for callback in self.callbacks.get(event_type, []):
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {event_type} callback: {e}")
        except Exception as e:
            logger.error(f"Error triggering {event_type} callbacks: {e}")

    def health_check(self) -> APIResponse:
        """
        Perform a health check on the API.

        Returns:
            API response with health status
        """
        try:
            response = self._make_request(
                method='GET',
                endpoint=self.endpoints.get('health_check', '/health'),
                timeout=3.0
            )

            if response.success:
                self.api_status = APIStatus.CONNECTED
            else:
                self.api_status = APIStatus.ERROR

            self._trigger_callbacks('status_changed', self.api_status)
            return response

        except Exception as e:
            self.api_status = APIStatus.ERROR
            self._trigger_callbacks('status_changed', self.api_status)
            return APIResponse(
                success=False,
                status_code=0,
                error_message=str(e)
            )

    def test_connection(self) -> bool:
        """
        Test connection to the API.

        Returns:
            True if connection test was successful
        """
        try:
            health_response = self.health_check()
            return health_response.success
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def reset_statistics(self):
        """Reset API statistics."""
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        self.error_history = []
        logger.info("API statistics reset")