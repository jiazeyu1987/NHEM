"""
Status Display Component
This module handles status display and user feedback for the LineDetectionWidget.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk

from ..config import get_status_colors, get_widget_config

logger = logging.getLogger(__name__)

class StatusState(Enum):
    """Status states for line detection."""
    DISABLED = "disabled"
    ENABLED_NO_DETECTION = "enabled_no_detection"
    DETECTION_SUCCESS = "detection_success"
    DETECTION_ERROR = "detection_error"
    PROCESSING = "processing"

@dataclass
class StatusMessage:
    """Represents a status message with metadata."""
    state: StatusState
    message: str
    details: Optional[str] = None
    timestamp: datetime = None
    confidence: Optional[float] = None
    coordinates: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class StatusDisplay:
    """
    Manages status display and user feedback for line detection operations.
    Provides a clean interface for status updates and visualization.
    """

    def __init__(self, parent_frame: ttk.Frame, config: Dict[str, Any] = None):
        """
        Initialize the status display.

        Args:
            parent_frame: Parent Tkinter frame
            config: Configuration dictionary
        """
        self.parent_frame = parent_frame
        self.config = get_widget_config(**(config or {}))

        # Status colors
        self.status_colors = get_status_colors()

        # Font configuration
        self.font_config = self.config.get('font', {})
        self.font_family = self.font_config.get('family', 'Microsoft YaHei')
        self.font_size = self.font_config.get('size', 10)
        self.font_bold_size = self.font_config.get('title_size', 10)
        self.timestamp_font_size = self.font_config.get('annotation_size', 8)

        self.fonts = {
            'normal': (self.font_family, self.font_size, 'normal'),
            'bold': (self.font_family, self.font_bold_size, 'bold'),
            'timestamp': (self.font_family, self.timestamp_font_size, 'normal'),
        }

        # Current state
        self.current_state: StatusState = StatusState.DISABLED
        self.current_message: Optional[str] = None
        self.current_details: Optional[str] = None
        self.last_update: Optional[datetime] = None

        # Status history
        self.status_history: List[StatusMessage] = []
        self.max_history_size = self.config.get('max_status_history', 10)

        # UI components
        self.status_frame: Optional[ttk.Frame] = None
        self.status_label: Optional[ttk.Label] = None
        self.details_label: Optional[ttk.Label] = None
        self.timestamp_label: Optional[ttk.Label] = None

        # Internationalization
        self.messages = {
            'disabled': {
                'en': 'Detection Disabled',
                'zh': '检测已禁用',
                'color': self.status_colors['disabled']
            },
            'enabled_no_detection': {
                'en': 'Detection Enabled - No Results',
                'zh': '检测已启用 - 无结果',
                'color': self.status_colors['enabled']
            },
            'detection_success': {
                'en': 'Detection Successful',
                'zh': '检测成功',
                'color': self.status_colors['success']
            },
            'detection_error': {
                'en': 'Detection Error',
                'zh': '检测错误',
                'color': self.status_colors['error']
            },
            'processing': {
                'en': 'Processing...',
                'zh': '处理中...',
                'color': self.status_colors['enabled']
            }
        }

        # Initialize UI
        self._setup_ui()
        self._update_display()

        logger.info("StatusDisplay initialized")

    def _setup_ui(self):
        """Set up the UI components."""
        # Create main status frame
        self.status_frame = ttk.LabelFrame(
            self.parent_frame,
            text="Detection Status",
            padding=10
        )
        self.status_frame.pack(fill='x', padx=5, pady=5)

        # Status label (main status text)
        self.status_label = ttk.Label(
            self.status_frame,
            text="",
            font=self.fonts['bold'],
            anchor='center'
        )
        self.status_label.pack(fill='x', pady=(0, 5))

        # Details label (additional information)
        self.details_label = ttk.Label(
            self.status_frame,
            text="",
            font=self.fonts['normal'],
            anchor='center',
            wraplength=300
        )
        self.details_label.pack(fill='x', pady=(0, 5))

        # Timestamp label
        self.timestamp_label = ttk.Label(
            self.status_frame,
            text="",
            font=self.fonts['timestamp'],
            anchor='center'
        )
        self.timestamp_label.pack(fill='x')

        # Configure colors
        self._configure_colors()

    def _configure_colors(self):
        """Configure colors for status display."""
        # Use configured colors
        fg_color = self.config.get('colors', {}).get('text', '#000000')
        bg_color = self.config.get('colors', {}).get('background', '#ffffff')

        # Apply colors to labels
        for label in [self.status_label, self.details_label, self.timestamp_label]:
            if hasattr(label, 'configure'):
                label.configure(foreground=fg_color)

        # Configure frame background if using themed widget
        if hasattr(self.status_frame, 'configure'):
            try:
                self.status_frame.configure(background=bg_color)
            except:
                pass  # Some themes don't support background color

    def update_status(self, state: StatusState, details: Optional[str] = None,
                     confidence: Optional[float] = None,
                     coordinates: Optional[Tuple[float, float]] = None):
        """
        Update the status display.

        Args:
            state: New status state
            details: Optional detailed message
            confidence: Optional confidence value (0.0-1.0)
            coordinates: Optional intersection coordinates
        """
        try:
            # Create status message
            message_key = state.value
            message_data = self.messages.get(message_key, self.messages['disabled'])
            message_text = message_data.get('en', 'Unknown Status')

            status_message = StatusMessage(
                state=state,
                message=message_text,
                details=details,
                confidence=confidence,
                coordinates=coordinates
            )

            # Update current state
            self.current_state = state
            self.current_message = message_text
            self.current_details = details
            self.last_update = datetime.now()

            # Add to history
            self._add_to_history(status_message)

            # Update display
            self._update_display()

            logger.debug(f"Status updated: {state.value} - {details}")

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _update_display(self):
        """Update the UI display."""
        try:
            if not self.status_label:
                return

            # Get status message data
            message_key = self.current_state.value
            message_data = self.messages.get(message_key, self.messages['disabled'])

            # Update status label
            display_text = message_data.get('en', 'Unknown Status')
            color = message_data.get('color', '#808080')

            self.status_label.config(text=display_text, foreground=color)

            # Update details label
            if self.current_details:
                self.details_label.config(text=self.current_details)
            else:
                # Generate default details based on state
                default_details = self._generate_default_details()
                self.details_label.config(text=default_details)

            # Update timestamp
            if self.last_update:
                timestamp_str = self.last_update.strftime("%H:%M:%S")
                self.timestamp_label.config(text=f"Last updated: {timestamp_str}")
            else:
                self.timestamp_label.config(text="")

        except Exception as e:
            logger.error(f"Error updating display: {e}")

    def _generate_default_details(self) -> str:
        """Generate default details text based on current state."""
        try:
            if self.current_state == StatusState.DISABLED:
                return "Click 'Enable Detection' to start"
            elif self.current_state == StatusState.ENABLED_NO_DETECTION:
                return "Waiting for detection results..."
            elif self.current_state == StatusState.DETECTION_SUCCESS:
                if self.current_message and self.current_message != "":
                    return f"Result: {self.current_message}"
                return "Detection completed successfully"
            elif self.current_state == StatusState.DETECTION_ERROR:
                return self.current_details or "An error occurred during detection"
            elif self.current_state == StatusState.PROCESSING:
                return "Please wait..."
            else:
                return ""
        except Exception as e:
            logger.error(f"Error generating default details: {e}")
            return ""

    def _add_to_history(self, status_message: StatusMessage):
        """Add status message to history."""
        self.status_history.append(status_message)

        # Maintain history size
        if len(self.status_history) > self.max_history_size:
            self.status_history.pop(0)

    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current status information.

        Returns:
            Dictionary with current status information
        """
        return {
            'state': self.current_state.value,
            'message': self.current_message,
            'details': self.current_details,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'history_count': len(self.status_history),
        }

    def get_status_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get status history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of status history entries
        """
        history = self.status_history.copy()

        if limit is not None:
            history = history[-limit:]

        return [
            {
                'state': msg.state.value,
                'message': msg.message,
                'details': msg.details,
                'timestamp': msg.timestamp.isoformat(),
                'confidence': msg.confidence,
                'coordinates': msg.coordinates,
            }
            for msg in history
        ]

    def set_language(self, language: str = 'en'):
        """
        Set the display language.

        Args:
            language: Language code ('en' or 'zh')
        """
        try:
            # Update messages for selected language
            for key in self.messages:
                if language in self.messages[key]:
                    # Update current display
                    if self.current_state and self.current_state.value == key:
                        if self.status_label:
                            self.status_label.config(text=self.messages[key][language])
            logger.info(f"Language set to: {language}")
        except Exception as e:
            logger.error(f"Error setting language: {e}")

    def set_colors(self, colors: Dict[str, str]):
        """
        Set custom colors for status display.

        Args:
            colors: Dictionary of status colors
        """
        try:
            # Update message colors
            for state_key, color in colors.items():
                if state_key in self.messages:
                    self.messages[state_key]['color'] = color

            # Update current display
            self._update_display()
            logger.info("Status colors updated")
        except Exception as e:
            logger.error(f"Error setting colors: {e}")

    def clear_history(self):
        """Clear status history."""
        self.status_history = []
        logger.info("Status history cleared")

    def reset(self):
        """Reset status display to initial state."""
        try:
            self.current_state = StatusState.DISABLED
            self.current_message = None
            self.current_details = None
            self.last_update = None
            self.clear_history()
            self._update_display()
            logger.info("Status display reset")
        except Exception as e:
            logger.error(f"Error resetting status display: {e}")

    def show_error(self, error_message: str, details: Optional[str] = None):
        """
        Show an error status.

        Args:
            error_message: Error message to display
            details: Optional additional details
        """
        self.update_status(
            StatusState.DETECTION_ERROR,
            details=details or error_message
        )

    def show_success(self, message: str, confidence: Optional[float] = None):
        """
        Show a success status.

        Args:
            message: Success message
            confidence: Optional confidence value
        """
        details = message
        if confidence is not None:
            details = f"{message} (Confidence: {confidence:.2f})"

        self.update_status(
            StatusState.DETECTION_SUCCESS,
            details=details,
            confidence=confidence
        )

    def show_processing(self, message: str = "Processing..."):
        """
        Show a processing status.

        Args:
            message: Processing message
        """
        self.update_status(StatusState.PROCESSING, details=message)

    def enable(self):
        """Enable detection status."""
        self.update_status(StatusState.ENABLED_NO_DETECTION)

    def disable(self):
        """Disable detection status."""
        self.update_status(StatusState.DISABLED)

    def get_state_duration(self) -> timedelta:
        """
        Get duration of current state.

        Returns:
            Duration since last state change
        """
        if self.last_update:
            return datetime.now() - self.last_update
        else:
            return timedelta(0)

    def get_state_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about status states.

        Returns:
            Dictionary with state statistics
        """
        state_counts = {}
        state_durations = {}

        for message in self.status_history:
            state = message.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        # Calculate total time in each state (simplified)
        if self.status_history:
            for message in self.status_history:
                state = message.state.value
                state_durations[state] = state_durations.get(state, 0)

        return {
            'total_state_changes': len(self.status_history),
            'state_counts': state_counts,
            'current_state_duration': self.get_state_duration().total_seconds(),
            'current_state': self.current_state.value,
        }

    def export_status_data(self) -> Dict[str, Any]:
        """
        Export status data for analysis.

        Returns:
            Dictionary containing all status data
        """
        return {
            'current_status': self.get_current_status(),
            'history': self.get_status_history(),
            'statistics': self.get_state_statistics(),
            'configuration': {
                'max_history_size': self.max_history_size,
                'font_family': self.font_family,
                'font_size': self.font_size,
            },
            'export_timestamp': datetime.now().isoformat(),
        }

    def set_visibility(self, visible: bool):
        """
        Set visibility of status display.

        Args:
            visible: Whether status display should be visible
        """
        try:
            if self.status_frame:
                if visible:
                    self.status_frame.pack(fill='x', padx=5, pady=5)
                else:
                    self.status_frame.pack_forget()
        except Exception as e:
            logger.error(f"Error setting visibility: {e}")

    def update_font_size(self, size: int):
        """
        Update font size for status display.

        Args:
            size: New font size
        """
        try:
            self.font_size = size
            self.font_bold_size = size
            self.timestamp_font_size = max(6, size - 2)

            self.fonts = {
                'normal': (self.font_family, self.font_size, 'normal'),
                'bold': (self.font_family, self.font_bold_size, 'bold'),
                'timestamp': (self.font_family, self.timestamp_font_size, 'normal'),
            }

            # Update existing labels
            if self.status_label:
                self.status_label.config(font=self.fonts['bold'])
            if self.details_label:
                self.details_label.config(font=self.fonts['normal'])
            if self.timestamp_label:
                self.timestamp_label.config(font=self.fonts['timestamp'])

            logger.info(f"Font size updated to {size}")
        except Exception as e:
            logger.error(f"Error updating font size: {e}")