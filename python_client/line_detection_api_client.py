"""
Line Detection API Client - 线条检测API客户端集成
提供完整的线条检测API访问功能，支持启用/禁用、手动检测、配置管理等操作
"""

import logging
import json
import base64
import io
from typing import Dict, Any, Optional, List, Tuple, Union
import requests
from requests import Session, Response
from urllib.parse import urljoin
import time
from enum import Enum

logger = logging.getLogger(__name__)


class LineDetectionAPIError(Exception):
    """线条检测API专用异常类"""
    def __init__(self, message: str, status_code: int = None, response_data: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class LineDetectionStatus(Enum):
    """线条检测状态枚举"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    PROCESSING = "processing"
    ERROR = "error"


class LineDetectionAPIClient:
    """
    线条检测API客户端

    提供与后端线条检测API的完整集成，支持：
    - 启用/禁用线条检测
    - 手动检测请求
    - 配置管理
    - 状态查询
    - 增强实时数据获取
    """

    def __init__(self, base_url: str, password: str, timeout: int = 10):
        """
        初始化API客户端

        Args:
            base_url: 后端服务器基础URL (例如: http://localhost:8421)
            password: 认证密码
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.password = password
        self.timeout = timeout

        # 创建会话以支持连接池和重用
        self.session = Session()

        # 设置默认请求头
        self.session.headers.update({
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'User-Agent': 'NHEM-LineDetection/1.0'
        })

        # API端点映射
        self.endpoints = {
            'enable': '/api/roi/line-intersection/enable',
            'disable': '/api/roi/line-intersection/disable',
            'status': '/api/roi/line-intersection/status',
            'manual_detection': '/api/roi/line-intersection',
            'config_get': '/api/roi/line-intersection/config',
            'config_update': '/api/roi/line-intersection/config',
            'realtime_enhanced': '/data/realtime/enhanced',
            'health': '/health',
            'roi_data': '/data/realtime',
            'dual_roi_data': '/data/dual-realtime'
        }

        # 重试配置
        self.max_retries = 3
        self.retry_delay = 1.0  # 秒

        # 统计信息
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None

        logger.info(f"LineDetectionAPIClient initialized for {base_url}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Response:
        """
        发送HTTP请求的通用方法

        Args:
            method: HTTP方法 ('GET', 'POST', 'PUT', 'DELETE')
            endpoint: API端点
            **kwargs: 请求参数

        Returns:
            Response对象

        Raises:
            LineDetectionAPIError: 请求失败时抛出
        """
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))

        # 更新统计信息
        self.request_count += 1
        self.last_request_time = time.time()

        # 添加认证参数
        if method.upper() in ['POST', 'PUT', 'DELETE']:
            if 'data' in kwargs:
                kwargs['data']['password'] = self.password
            else:
                kwargs['data'] = {'password': self.password}
        elif method.upper() == 'GET':
            if 'params' in kwargs:
                kwargs['params']['password'] = self.password
            else:
                kwargs['params'] = {'password': self.password}

        # 执行重试逻辑
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making {method} request to {url} (attempt {attempt + 1})")

                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )

                # 记录请求
                logger.debug(f"Response status: {response.status_code}")

                # 检查HTTP状态码
                if response.status_code >= 400:
                    error_msg = f"HTTP {response.status_code}: {response.reason}"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('error', error_data.get('detail', error_msg))
                        error_msg = f"{error_msg} - {error_detail}"
                    except (ValueError, KeyError):
                        pass

                    if response.status_code >= 500 and attempt < self.max_retries:
                        # 服务器错误，可以重试
                        last_exception = LineDetectionAPIError(error_msg, response.status_code)
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue

                    self.error_count += 1
                    raise LineDetectionAPIError(error_msg, response.status_code)

                return response

            except requests.exceptions.Timeout:
                last_exception = LineDetectionAPIError(f"Request timeout after {self.timeout}s")
                if attempt < self.max_retries:
                    logger.warning(f"Request timeout, retrying... (attempt {attempt + 1})")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                break

            except requests.exceptions.ConnectionError:
                last_exception = LineDetectionAPIError("Connection error")
                if attempt < self.max_retries:
                    logger.warning(f"Connection error, retrying... (attempt {attempt + 1})")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                break

            except requests.exceptions.RequestException as e:
                last_exception = LineDetectionAPIError(f"Request exception: {str(e)}")
                break

        self.error_count += 1
        raise last_exception or LineDetectionAPIError("Unknown request error")

    def _parse_response(self, response: Response) -> Dict[str, Any]:
        """
        解析API响应

        Args:
            response: Response对象

        Returns:
            解析后的数据字典
        """
        try:
            data = response.json()
            logger.debug(f"Response data: {data}")
            return data
        except ValueError as e:
            raise LineDetectionAPIError(f"Invalid JSON response: {str(e)}")

    def enable_line_detection(self) -> Dict[str, Any]:
        """
        启用线条检测

        Returns:
            API响应数据

        Raises:
            LineDetectionAPIError: 启用失败时抛出
        """
        try:
            logger.info("Enabling line detection")

            response = self._make_request('POST', self.endpoints['enable'])
            data = self._parse_response(response)

            if data.get('success', False):
                logger.info("Line detection enabled successfully")
            else:
                error_msg = data.get('error', 'Failed to enable line detection')
                raise LineDetectionAPIError(error_msg, response.status_code, data)

            return data

        except Exception as e:
            logger.error(f"Error enabling line detection: {e}")
            raise

    def disable_line_detection(self) -> Dict[str, Any]:
        """
        禁用线条检测

        Returns:
            API响应数据

        Raises:
            LineDetectionAPIError: 禁用失败时抛出
        """
        try:
            logger.info("Disabling line detection")

            response = self._make_request('POST', self.endpoints['disable'])
            data = self._parse_response(response)

            if data.get('success', False):
                logger.info("Line detection disabled successfully")
            else:
                error_msg = data.get('error', 'Failed to disable line detection')
                raise LineDetectionAPIError(error_msg, response.status_code, data)

            return data

        except Exception as e:
            logger.error(f"Error disabling line detection: {e}")
            raise

    def get_detection_status(self) -> Dict[str, Any]:
        """
        获取线条检测状态

        Returns:
            包含检测状态的字典
        """
        try:
            logger.debug("Getting line detection status")

            response = self._make_request('GET', self.endpoints['status'])
            data = self._parse_response(response)

            # 解析状态信息
            status_data = {
                'enabled': data.get('enabled', False),
                'status': data.get('status', 'unknown'),
                'last_detection_time': data.get('last_detection_time'),
                'detection_count': data.get('detection_count', 0),
                'error_count': data.get('error_count', 0),
                'last_error': data.get('last_error'),
                'config': data.get('config', {}),
                'raw_response': data
            }

            logger.debug(f"Line detection status: {status_data}")
            return status_data

        except Exception as e:
            logger.error(f"Error getting detection status: {e}")
            raise

    def manual_detection(self,
                        roi_coordinates: Optional[Dict[str, int]] = None,
                        image_data: Optional[str] = None,
                        force_refresh: bool = False) -> Dict[str, Any]:
        """
        执行手动线条检测

        Args:
            roi_coordinates: ROI坐标 {'x1': int, 'y1': int, 'x2': int, 'y2': int}
            image_data: Base64编码的图像数据
            force_refresh: 是否强制刷新缓存

        Returns:
            包含检测结果的字典
        """
        try:
            logger.info("Executing manual line detection")

            # 准备请求数据
            request_data = {}

            if roi_coordinates:
                request_data.update({
                    'x1': roi_coordinates['x1'],
                    'y1': roi_coordinates['y1'],
                    'x2': roi_coordinates['x2'],
                    'y2': roi_coordinates['y2']
                })

            if image_data:
                request_data['image_data'] = image_data

            request_data['force_refresh'] = force_refresh

            # 发送请求
            response = self._make_request('POST', self.endpoints['manual_detection'], data=request_data)
            data = self._parse_response(response)

            # 解析检测结果
            if data.get('success', False):
                detection_result = {
                    'success': True,
                    'lines': data.get('lines', []),
                    'intersections': data.get('intersections', []),
                    'processing_time_ms': data.get('processing_time_ms', 0),
                    'detection_confidence': data.get('detection_confidence', 0.0),
                    'roi_info': data.get('roi_info', {}),
                    'raw_response': data
                }

                logger.info(f"Manual detection completed: {len(detection_result['lines'])} lines, "
                           f"{len(detection_result['intersections'])} intersections")
                return detection_result
            else:
                error_msg = data.get('error', 'Manual detection failed')
                raise LineDetectionAPIError(error_msg, response.status_code, data)

        except Exception as e:
            logger.error(f"Error in manual detection: {e}")
            raise

    def get_enhanced_realtime_data(self,
                                  count: int = 100,
                                  include_line_intersection: bool = True) -> Dict[str, Any]:
        """
        获取增强的实时数据（包含线条交点信息）

        Args:
            count: 获取的数据点数量
            include_line_intersection: 是否包含线条交点数据

        Returns:
            增强的实时数据字典
        """
        try:
            logger.debug(f"Getting enhanced realtime data: count={count}, include_lines={include_line_intersection}")

            params = {
                'count': count,
                'include_line_intersection': str(include_line_intersection).lower()
            }

            response = self._make_request('GET', self.endpoints['realtime_enhanced'], params=params)
            data = self._parse_response(response)

            # 解析增强数据
            enhanced_data = {
                'type': data.get('type', 'enhanced_realtime_data'),
                'timestamp': data.get('timestamp'),
                'data_points': data.get('data_points', []),
                'line_intersection_data': data.get('line_intersection_data', {}),
                'roi_data': data.get('roi_data', {}),
                'peak_detection_results': data.get('peak_detection_results', {}),
                'processing_info': data.get('processing_info', {}),
                'raw_response': data
            }

            logger.debug(f"Enhanced data retrieved: {len(enhanced_data['data_points'])} points")
            return enhanced_data

        except Exception as e:
            logger.error(f"Error getting enhanced realtime data: {e}")
            raise

    def get_line_detection_config(self) -> Dict[str, Any]:
        """
        获取线条检测配置

        Returns:
            配置数据字典
        """
        try:
            logger.debug("Getting line detection configuration")

            response = self._make_request('GET', self.endpoints['config_get'])
            data = self._parse_response(response)

            # 解析配置数据
            config_data = {
                'enabled': data.get('enabled', False),
                'hsv_green_lower': data.get('hsv_green_lower', [40, 50, 50]),
                'hsv_green_upper': data.get('hsv_green_upper', [80, 255, 255]),
                'canny_low_threshold': data.get('canny_low_threshold', 25),
                'canny_high_threshold': data.get('canny_high_threshold', 80),
                'hough_threshold': data.get('hough_threshold', 50),
                'hough_min_line_length': data.get('hough_min_line_length', 15),
                'hough_max_line_gap': data.get('hough_max_line_gap', 8),
                'min_confidence': data.get('min_confidence', 0.4),
                'roi_processing_mode': data.get('roi_processing_mode', 'roi1_only'),
                'cache_timeout_ms': data.get('cache_timeout_ms', 100),
                'max_processing_time_ms': data.get('max_processing_time_ms', 300),
                'raw_response': data
            }

            logger.debug("Line detection configuration retrieved")
            return config_data

        except Exception as e:
            logger.error(f"Error getting line detection config: {e}")
            raise

    def update_line_detection_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新线条检测配置

        Args:
            config_updates: 配置更新字典

        Returns:
            更新后的配置数据
        """
        try:
            logger.info("Updating line detection configuration")

            # 验证配置参数
            self._validate_config_updates(config_updates)

            response = self._make_request('POST', self.endpoints['config_update'], data=config_updates)
            data = self._parse_response(response)

            if data.get('success', False):
                logger.info("Configuration updated successfully")
                return data
            else:
                error_msg = data.get('error', 'Failed to update configuration')
                raise LineDetectionAPIError(error_msg, response.status_code, data)

        except Exception as e:
            logger.error(f"Error updating line detection config: {e}")
            raise

    def get_current_roi_data(self, dual_roi: bool = False) -> Dict[str, Any]:
        """
        获取当前ROI数据

        Args:
            dual_roi: 是否获取双ROI数据

        Returns:
            ROI数据字典
        """
        try:
            logger.debug(f"Getting current ROI data (dual_roi={dual_roi})")

            endpoint = self.endpoints['dual_roi_data'] if dual_roi else self.endpoints['roi_data']
            params = {'count': 1}

            response = self._make_request('GET', endpoint, params=params)
            data = self._parse_response(response)

            if dual_roi:
                roi_data = {
                    'type': 'dual_realtime_data',
                    'dual_roi_data': data.get('dual_roi_data', {}),
                    'timestamp': data.get('timestamp'),
                    'raw_response': data
                }
            else:
                roi_data = {
                    'type': 'realtime_data',
                    'roi_data': data.get('roi_data', {}),
                    'timestamp': data.get('timestamp'),
                    'raw_response': data
                }

            logger.debug("ROI data retrieved successfully")
            return roi_data

        except Exception as e:
            logger.error(f"Error getting ROI data: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查

        Returns:
            健康检查结果
        """
        try:
            logger.debug("Performing health check")

            response = self._make_request('GET', self.endpoints['health'])
            data = self._parse_response(response)

            health_data = {
                'status': data.get('status', 'unknown'),
                'timestamp': data.get('timestamp'),
                'uptime': data.get('uptime'),
                'version': data.get('version'),
                'raw_response': data
            }

            logger.debug(f"Health check result: {health_data['status']}")
            return health_data

        except Exception as e:
            logger.error(f"Error in health check: {e}")
            raise

    def _validate_config_updates(self, config: Dict[str, Any]):
        """验证配置更新参数"""
        try:
            # HSV阈值验证
            if 'hsv_green_lower' in config:
                lower = config['hsv_green_lower']
                if not isinstance(lower, list) or len(lower) != 3:
                    raise ValueError("hsv_green_lower must be a list of 3 integers [H, S, V]")
                if not all(0 <= val <= 255 for val in lower):
                    raise ValueError("HSV values must be between 0 and 255")

            if 'hsv_green_upper' in config:
                upper = config['hsv_green_upper']
                if not isinstance(upper, list) or len(upper) != 3:
                    raise ValueError("hsv_green_upper must be a list of 3 integers [H, S, V]")
                if not all(0 <= val <= 255 for val in upper):
                    raise ValueError("HSV values must be between 0 and 255")

            # Canny阈值验证
            if 'canny_low_threshold' in config:
                if not isinstance(config['canny_low_threshold'], (int, float)) or config['canny_low_threshold'] < 0:
                    raise ValueError("canny_low_threshold must be a non-negative number")

            if 'canny_high_threshold' in config:
                high = config['canny_high_threshold']
                if not isinstance(high, (int, float)) or high < 0:
                    raise ValueError("canny_high_threshold must be a non-negative number")

            # 霍夫变换参数验证
            if 'hough_threshold' in config:
                if not isinstance(config['hough_threshold'], (int, float)) or config['hough_threshold'] < 0:
                    raise ValueError("hough_threshold must be non-negative")

            if 'hough_min_line_length' in config:
                if not isinstance(config['hough_min_line_length'], (int, float)) or config['hough_min_line_length'] < 0:
                    raise ValueError("hough_min_line_length must be non-negative")

            if 'hough_max_line_gap' in config:
                if not isinstance(config['hough_max_line_gap'], (int, float)) or config['hough_max_line_gap'] < 0:
                    raise ValueError("hough_max_line_gap must be non-negative")

            # 置信度验证
            if 'min_confidence' in config:
                conf = config['min_confidence']
                if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                    raise ValueError("min_confidence must be between 0.0 and 1.0")

            # 处理模式验证
            if 'roi_processing_mode' in config:
                mode = config['roi_processing_mode']
                valid_modes = ['roi1_only', 'roi2_only', 'dual_roi', 'auto']
                if mode not in valid_modes:
                    raise ValueError(f"roi_processing_mode must be one of: {valid_modes}")

            # 超时参数验证
            if 'cache_timeout_ms' in config:
                timeout = config['cache_timeout_ms']
                if not isinstance(timeout, (int, float)) or timeout < 0:
                    raise ValueError("cache_timeout_ms must be non-negative")

            if 'max_processing_time_ms' in config:
                max_time = config['max_processing_time_ms']
                if not isinstance(max_time, (int, float)) or max_time < 0:
                    raise ValueError("max_processing_time_ms must be non-negative")

        except ValueError as e:
            raise LineDetectionAPIError(f"Configuration validation failed: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取客户端统计信息

        Returns:
            统计信息字典
        """
        stats = {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'success_rate': (self.request_count - self.error_count) / max(self.request_count, 1),
            'last_request_time': self.last_request_time,
            'session_info': {
                'timeout': self.timeout,
                'max_retries': self.max_retries,
                'retry_delay': self.retry_delay
            }
        }

        if self.last_request_time:
            stats['time_since_last_request'] = time.time() - self.last_request_time

        return stats

    def reset_statistics(self):
        """重置统计信息"""
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        logger.info("Statistics reset")

    def close(self):
        """关闭客户端会话"""
        if self.session:
            self.session.close()
            logger.info("LineDetectionAPIClient session closed")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 便捷函数

def create_line_detection_client(base_url: str, password: str, timeout: int = 10) -> LineDetectionAPIClient:
    """
    创建线条检测API客户端的便捷函数

    Args:
        base_url: 后端服务器基础URL
        password: 认证密码
        timeout: 请求超时时间

    Returns:
        LineDetectionAPIClient实例
    """
    return LineDetectionAPIClient(base_url, password, timeout)


def toggle_line_detection(client: LineDetectionAPIClient, enabled: bool) -> bool:
    """
    切换线条检测状态的便捷函数

    Args:
        client: API客户端实例
        enabled: 目标状态

    Returns:
        操作是否成功
    """
    try:
        if enabled:
            result = client.enable_line_detection()
        else:
            result = client.disable_line_detection()

        return result.get('success', False)
    except Exception:
        return False


def get_detection_status_simple(client: LineDetectionAPIClient) -> Tuple[bool, str]:
    """
    获取检测状态的便捷函数

    Args:
        client: API客户端实例

    Returns:
        (是否启用, 状态描述)
    """
    try:
        status_data = client.get_detection_status()
        enabled = status_data.get('enabled', False)
        status_desc = status_data.get('status', 'unknown')
        return enabled, status_desc
    except Exception as e:
        return False, f"Error: {str(e)}"