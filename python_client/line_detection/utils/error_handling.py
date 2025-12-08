"""
Error Handling Utilities
This module provides centralized error handling and user-friendly error management.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import sys

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    API = "api"
    DATA = "data"
    UI = "ui"
    PROCESSING = "processing"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for an error."""
    component: str
    operation: str
    user_action: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception]
    context: Optional[ErrorContext]
    timestamp: datetime
    stack_trace: Optional[str] = None
    user_friendly_message: Optional[str] = None
    suggested_actions: List[str] = None
    resolved: bool = False

    def __post_init__(self):
        if self.suggested_actions is None:
            self.suggested_actions = []

class ErrorHandlingSystem:
    """
    Centralized error handling system for line detection operations.
    Provides error classification, user-friendly messages, and recovery suggestions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the error handling system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Error storage
        self.error_history: List[ErrorRecord] = []
        self.max_history_size = self.config.get('max_error_history', 100)
        self.error_statistics: Dict[str, Any] = {
            'total_errors': 0,
            'by_category': {},
            'by_severity': {},
            'by_component': {},
            'recent_errors': []
        }

        # Error message translations
        self.error_translations = self._load_error_translations()

        # Error handlers
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = {
            category: [] for category in ErrorCategory
        }

        # Recovery strategies
        self.recovery_strategies = self._load_recovery_strategies()

        # Logging configuration
        self.log_errors = self.config.get('log_errors', True)
        self.log_level = self.config.get('log_level', 'WARNING')

        # Error notifications
        self.notification_callbacks: List[Callable] = []

        logger.info("ErrorHandlingSystem initialized")

    def _load_error_translations(self) -> Dict[str, Dict[str, Any]]:
        """Load error message translations and recovery suggestions."""
        return {
            # Network errors
            'ConnectionError': {
                'category': ErrorCategory.NETWORK,
                'severity': ErrorSeverity.ERROR,
                'user_message': 'Network connection failed. Please check your internet connection.',
                'suggested_actions': [
                    'Check internet connection',
                    'Verify server address',
                    'Try again in a few moments',
                    'Contact support if problem persists'
                ]
            },
            'TimeoutError': {
                'category': ErrorCategory.NETWORK,
                'severity': ErrorSeverity.WARNING,
                'user_message': 'Operation timed out. The server took too long to respond.',
                'suggested_actions': [
                    'Try again',
                    'Check if server is responsive',
                    'Increase timeout settings if possible'
                ]
            },
            'requests.exceptions.ConnectionError': {
                'category': ErrorCategory.NETWORK,
                'severity': ErrorSeverity.ERROR,
                'user_message': 'Failed to connect to the detection server.',
                'suggested_actions': [
                    'Check server is running',
                    'Verify server address and port',
                    'Check firewall settings'
                ]
            },

            # API errors
            'HTTPError': {
                'category': ErrorCategory.API,
                'severity': ErrorSeverity.ERROR,
                'user_message': 'Server returned an error response.',
                'suggested_actions': [
                    'Check request parameters',
                    'Verify API permissions',
                    'Try again later'
                ]
            },
            'ValidationError': {
                'category': ErrorCategory.API,
                'severity': ErrorSeverity.WARNING,
                'user_message': 'Invalid data provided to the API.',
                'suggested_actions': [
                    'Check input data format',
                    'Verify required fields',
                    'Consult API documentation'
                ]
            },

            # Data errors
            'ValueError': {
                'category': ErrorCategory.DATA,
                'severity': ErrorSeverity.ERROR,
                'user_message': 'Invalid data value encountered.',
                'suggested_actions': [
                    'Check input data format',
                    'Verify data ranges',
                    'Ensure data is not corrupted'
                ]
            },
            'JSONDecodeError': {
                'category': ErrorCategory.DATA,
                'severity': ErrorSeverity.ERROR,
                'user_message': 'Failed to parse JSON data.',
                'suggested_actions': [
                    'Check data format',
                    'Ensure valid JSON structure',
                    'Verify data encoding'
                ]
            },

            # Image processing errors
            'PIL.UnidentifiedImageError': {
                'category': ErrorCategory.PROCESSING,
                'severity': ErrorSeverity.ERROR,
                'user_message': 'Invalid or corrupted image data.',
                'suggested_actions': [
                    'Check image format',
                    'Verify image data integrity',
                    'Try with a different image'
                ]
            },

            # Configuration errors
            'FileNotFoundError': {
                'category': ErrorCategory.CONFIGURATION,
                'severity': ErrorSeverity.ERROR,
                'user_message': 'Required configuration file not found.',
                'suggested_actions': [
                    'Check file path',
                    'Verify file permissions',
                    'Reinstall if necessary'
                ]
            },

            # UI errors
            'Tkinter.TclError': {
                'category': ErrorCategory.UI,
                'severity': ErrorSeverity.WARNING,
                'user_message': 'UI component error occurred.',
                'suggested_actions': [
                    'Refresh the interface',
                    'Restart the application',
                    'Check display settings'
                ]
            }
        }

    def _load_recovery_strategies(self) -> Dict[ErrorCategory, List[Callable]]:
        """Load recovery strategies for different error categories."""
        return {
            ErrorCategory.NETWORK: [
                self._retry_connection,
                self._check_server_status,
                self._reset_connection
            ],
            ErrorCategory.API: [
                self._refresh_api_client,
                self._validate_api_parameters
            ],
            ErrorCategory.DATA: [
                self._clear_data_cache,
                self._reset_data_processor
            ],
            ErrorCategory.UI: [
                self._refresh_ui_components,
                self._reset_ui_state
            ],
            ErrorCategory.PROCESSING: [
                self._reset_processing_pipeline,
                self._clear_processing_cache
            ]
        }

    def handle_error(self,
                    exception: Exception,
                    component: str = "unknown",
                    operation: str = "unknown",
                    user_action: Optional[str] = None,
                    additional_data: Optional[Dict[str, Any]] = None) -> ErrorRecord:
        """
        Handle an error with comprehensive logging and user-friendly messaging.

        Args:
            exception: The exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            user_action: User action that triggered the error
            additional_data: Additional context data

        Returns:
            ErrorRecord with all error information
        """
        try:
            # Generate error ID
            error_id = f"ERR_{int(datetime.now().timestamp())}_{len(self.error_history)}"

            # Create error context
            context = ErrorContext(
                component=component,
                operation=operation,
                user_action=user_action,
                additional_data=additional_data
            )

            # Classify error
            category, severity, user_message, suggested_actions = self._classify_error(exception)

            # Get stack trace
            stack_trace = traceback.format_exc() if self.log_errors else None

            # Create error record
            error_record = ErrorRecord(
                error_id=error_id,
                category=category,
                severity=severity,
                message=str(exception),
                original_exception=exception,
                context=context,
                timestamp=datetime.now(),
                stack_trace=stack_trace,
                user_friendly_message=user_message,
                suggested_actions=suggested_actions
            )

            # Store error
            self._store_error(error_record)

            # Log error
            self._log_error(error_record)

            # Trigger error handlers
            self._trigger_error_handlers(error_record)

            # Send notifications
            self._send_notifications(error_record)

            logger.info(f"Error handled: {error_id} - {category.value}/{severity.value}")

            return error_record

        except Exception as e:
            # Fallback error handling
            logger.critical(f"Error in error handling system: {e}")
            return ErrorRecord(
                error_id=f"CRITICAL_{int(datetime.now().timestamp())}",
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.CRITICAL,
                message=f"Error handling system failure: {str(e)}",
                original_exception=e,
                context=ErrorContext("ErrorHandlingSystem", "handle_error"),
                timestamp=datetime.now()
            )

    def _classify_error(self, exception: Exception) -> Tuple[ErrorCategory, ErrorSeverity, str, List[str]]:
        """Classify error and generate user-friendly message."""
        exception_type = type(exception).__name__
        full_exception_name = f"{exception.__class__.__module__}.{exception_type}"

        # Look for exact match first
        for error_key, error_info in self.error_translations.items():
            if error_key in [exception_type, full_exception_name]:
                return (
                    error_info['category'],
                    error_info['severity'],
                    error_info['user_message'],
                    error_info['suggested_actions'].copy()
                )

        # Look for partial matches
        for error_key, error_info in self.error_translations.items():
            if error_key.lower() in exception_type.lower() or exception_type.lower() in error_key.lower():
                return (
                    error_info['category'],
                    error_info['severity'],
                    error_info['user_message'],
                    error_info['suggested_actions'].copy()
                )

        # Default classification
        if isinstance(exception, (ConnectionError, OSError)):
            return ErrorCategory.NETWORK, ErrorSeverity.ERROR, "A network error occurred.", ["Check network connection"]
        elif isinstance(exception, ValueError):
            return ErrorCategory.DATA, ErrorSeverity.ERROR, "Invalid data provided.", ["Check input data"]
        elif isinstance(exception, KeyError):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.ERROR, "Missing required configuration.", ["Check settings"]
        else:
            return ErrorCategory.UNKNOWN, ErrorSeverity.ERROR, f"An error occurred: {str(exception)}", ["Try again"]

    def _store_error(self, error_record: ErrorRecord):
        """Store error record in history."""
        self.error_history.append(error_record)

        # Maintain history size
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)

        # Update statistics
        self._update_statistics(error_record)

    def _update_statistics(self, error_record: ErrorRecord):
        """Update error statistics."""
        self.error_statistics['total_errors'] += 1

        # By category
        category = error_record.category.value
        self.error_statistics['by_category'][category] = self.error_statistics['by_category'].get(category, 0) + 1

        # By severity
        severity = error_record.severity.value
        self.error_statistics['by_severity'][severity] = self.error_statistics['by_severity'].get(severity, 0) + 1

        # By component
        component = error_record.context.component
        self.error_statistics['by_component'][component] = self.error_statistics['by_component'].get(component, 0) + 1

        # Recent errors (last 10)
        self.error_statistics['recent_errors'].append({
            'error_id': error_record.error_id,
            'category': category,
            'severity': severity,
            'message': error_record.message,
            'timestamp': error_record.timestamp.isoformat()
        })

        if len(self.error_statistics['recent_errors']) > 10:
            self.error_statistics['recent_errors'].pop(0)

    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level."""
        if not self.log_errors:
            return

        log_message = f"{error_record.error_id} [{error_record.category.value}] {error_record.message}"

        if error_record.context:
            log_message += f" (Component: {error_record.context.component}, Operation: {error_record.context.operation})"

        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error_record.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # Log stack trace for debugging
        if error_record.stack_trace and error_record.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            logger.debug(f"Stack trace for {error_record.error_id}:\n{error_record.stack_trace}")

    def _trigger_error_handlers(self, error_record: ErrorRecord):
        """Trigger error handlers for the error category."""
        handlers = self.error_handlers.get(error_record.category, [])
        for handler in handlers:
            try:
                handler(error_record)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")

    def _send_notifications(self, error_record: ErrorRecord):
        """Send error notifications to registered callbacks."""
        for callback in self.notification_callbacks:
            try:
                callback(error_record)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")

    # Recovery strategies
    def _retry_connection(self, error_record: ErrorRecord):
        """Attempt to retry network connection."""
        logger.info(f"Attempting connection retry for {error_record.error_id}")
        # Implementation would depend on specific network client

    def _check_server_status(self, error_record: ErrorRecord):
        """Check if server is responding."""
        logger.info(f"Checking server status for {error_record.error_id}")
        # Implementation would make health check request

    def _reset_connection(self, error_record: ErrorRecord):
        """Reset network connection."""
        logger.info(f"Resetting connection for {error_record.error_id}")
        # Implementation would reset connection state

    def _refresh_api_client(self, error_record: ErrorRecord):
        """Refresh API client state."""
        logger.info(f"Refreshing API client for {error_record.error_id}")

    def _validate_api_parameters(self, error_record: ErrorRecord):
        """Validate API parameters."""
        logger.info(f"Validating API parameters for {error_record.error_id}")

    def _clear_data_cache(self, error_record: ErrorRecord):
        """Clear data processing cache."""
        logger.info(f"Clearing data cache for {error_record.error_id}")

    def _reset_data_processor(self, error_record: ErrorRecord):
        """Reset data processor state."""
        logger.info(f"Resetting data processor for {error_record.error_id}")

    def _refresh_ui_components(self, error_record: ErrorRecord):
        """Refresh UI components."""
        logger.info(f"Refreshing UI components for {error_record.error_id}")

    def _reset_ui_state(self, error_record: ErrorRecord):
        """Reset UI to default state."""
        logger.info(f"Resetting UI state for {error_record.error_id}")

    def _reset_processing_pipeline(self, error_record: ErrorRecord):
        """Reset processing pipeline."""
        logger.info(f"Resetting processing pipeline for {error_record.error_id}")

    def _clear_processing_cache(self, error_record: ErrorRecord):
        """Clear processing cache."""
        logger.info(f"Clearing processing cache for {error_record.error_id}")

    # Public API methods
    def add_error_handler(self, category: ErrorCategory, handler: Callable):
        """Add an error handler for a specific category."""
        self.error_handlers[category].append(handler)

    def add_notification_callback(self, callback: Callable):
        """Add a callback for error notifications."""
        self.notification_callbacks.append(callback)

    def get_error_history(self, limit: Optional[int] = None,
                         category: Optional[ErrorCategory] = None,
                         severity: Optional[ErrorSeverity] = None) -> List[ErrorRecord]:
        """
        Get error history with optional filtering.

        Args:
            limit: Maximum number of errors to return
            category: Filter by error category
            severity: Filter by error severity

        Returns:
            Filtered list of error records
        """
        errors = self.error_history.copy()

        # Apply filters
        if category:
            errors = [e for e in errors if e.category == category]

        if severity:
            errors = [e for e in errors if e.severity == severity]

        # Apply limit
        if limit:
            errors = errors[-limit:]

        return errors

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self.error_statistics.copy()

    def clear_error_history(self):
        """Clear error history and statistics."""
        self.error_history = []
        self.error_statistics = {
            'total_errors': 0,
            'by_category': {},
            'by_severity': {},
            'by_component': {},
            'recent_errors': []
        }
        logger.info("Error history cleared")

    def resolve_error(self, error_id: str) -> bool:
        """
        Mark an error as resolved.

        Args:
            error_id: ID of error to resolve

        Returns:
            True if error was found and resolved
        """
        for error in self.error_history:
            if error.error_id == error_id:
                error.resolved = True
                logger.info(f"Error resolved: {error_id}")
                return True
        return False

    def get_user_friendly_message(self, error_record: ErrorRecord) -> str:
        """Get user-friendly message for an error record."""
        if error_record.user_friendly_message:
            return error_record.user_friendly_message
        else:
            return f"An error occurred in {error_record.context.component}: {error_record.message}"

    def get_recovery_actions(self, error_record: ErrorRecord) -> List[str]:
        """Get suggested recovery actions for an error."""
        if error_record.suggested_actions:
            return error_record.suggested_actions.copy()

        # Try to get recovery strategies for the category
        strategies = self.recovery_strategies.get(error_record.category, [])
        if strategies:
            return [f"Attempt recovery strategy: {strategy.__name__}" for strategy in strategies]

        return ["Contact support if the problem persists"]

    def export_error_data(self, include_stack_traces: bool = False) -> Dict[str, Any]:
        """
        Export error data for analysis.

        Args:
            include_stack_traces: Whether to include stack traces

        Returns:
            Dictionary containing all error data
        """
        error_data = {
            'export_timestamp': datetime.now().isoformat(),
            'statistics': self.get_error_statistics(),
            'errors': []
        }

        for error in self.error_history:
            error_dict = {
                'error_id': error.error_id,
                'category': error.category.value,
                'severity': error.severity.value,
                'message': error.message,
                'user_friendly_message': error.user_friendly_message,
                'suggested_actions': error.suggested_actions,
                'timestamp': error.timestamp.isoformat(),
                'resolved': error.resolved,
                'context': {
                    'component': error.context.component,
                    'operation': error.context.operation,
                    'user_action': error.context.user_action,
                    'additional_data': error.context.additional_data
                }
            }

            if include_stack_traces and error.stack_trace:
                error_dict['stack_trace'] = error.stack_trace

            error_data['errors'].append(error_dict)

        return error_data

    def auto_resolve_errors(self, max_age_hours: int = 24) -> int:
        """
        Automatically resolve old errors.

        Args:
            max_age_hours: Maximum age in hours for errors to remain unresolved

        Returns:
            Number of errors resolved
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        resolved_count = 0

        for error in self.error_history:
            if not error.resolved and error.timestamp < cutoff_time:
                error.resolved = True
                resolved_count += 1

        if resolved_count > 0:
            logger.info(f"Auto-resolved {resolved_count} old errors")

        return resolved_count