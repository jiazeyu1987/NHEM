"""
Controls Manager Component
This module manages UI controls and user interactions for the LineDetectionWidget.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import tkinter as tk
from tkinter import ttk
import threading
import time

from ..config import get_widget_config

logger = logging.getLogger(__name__)

class ControlState(Enum):
    """States for UI controls."""
    NORMAL = "normal"
    LOADING = "loading"
    DISABLED = "disabled"
    ERROR = "error"

class ControlType(Enum):
    """Types of UI controls."""
    BUTTON = "button"
    TOGGLE = "toggle"
    SLIDER = "slider"
    COMBOBOX = "combobox"
    CHECKBOX = "checkbox"
    TEXT_ENTRY = "text_entry"
    LABEL = "label"

@dataclass
class ControlConfig:
    """Configuration for a UI control."""
    control_type: ControlType
    text: str
    command: Optional[Callable] = None
    width: Optional[int] = None
    height: Optional[int] = None
    enabled: bool = True
    visible: bool = True
    tooltip: Optional[str] = None
    state: ControlState = ControlState.NORMAL

@dataclass
class LoadingState:
    """Represents a loading state for a control."""
    loading: bool = False
    original_text: Optional[str] = None
    loading_text: str = "Processing..."

class ControlsManager:
    """
    Manages UI controls and user interactions for line detection operations.
    Provides a clean interface for control creation and state management.
    """

    def __init__(self, parent_frame: ttk.Frame, config: Dict[str, Any] = None):
        """
        Initialize the controls manager.

        Args:
            parent_frame: Parent Tkinter frame
            config: Configuration dictionary
        """
        self.parent_frame = parent_frame
        self.config = get_widget_config(**(config or {}))

        # UI controls storage
        self.controls: Dict[str, ttk.Widget] = {}
        self.control_configs: Dict[str, ControlConfig] = {}
        self.loading_states: Dict[str, LoadingState] = {}

        # Main frames
        self.main_frame: Optional[ttk.Frame] = None
        self.button_frame: Optional[ttk.Frame] = None
        self.settings_frame: Optional[ttk.Frame] = None

        # Font configuration
        self.font_config = self.config.get('font', {})
        self.font_family = self.font_config.get('family', 'Microsoft YaHei')
        self.font_size = self.font_config.get('size', 10)

        self.fonts = {
            'normal': (self.font_family, self.font_size, 'normal'),
            'bold': (self.font_family, self.font_size, 'bold'),
            'small': (self.font_family, max(6, self.font_size - 2), 'normal'),
        }

        # Colors
        self.colors = self.config.get('colors', {})

        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {}

        # Initialize UI
        self._setup_ui()

        logger.info("ControlsManager initialized")

    def _setup_ui(self):
        """Set up the main UI structure."""
        # Create main frame
        self.main_frame = ttk.Frame(self.parent_frame)
        self.main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create button frame
        self.button_frame = ttk.LabelFrame(
            self.main_frame,
            text="Detection Controls",
            padding=10
        )
        self.button_frame.pack(fill='x', pady=(0, 5))

        # Create settings frame
        self.settings_frame = ttk.LabelFrame(
            self.main_frame,
            text="Settings",
            padding=10
        )
        self.settings_frame.pack(fill='x', pady=(5, 0))

        # Setup default controls
        self._setup_default_controls()

    def _setup_default_controls(self):
        """Setup default detection controls."""
        # Detection toggle button
        self.add_control(
            'toggle_detection',
            ControlConfig(
                control_type=ControlType.TOGGLE,
                text="Enable Detection",
                command=self._on_toggle_detection,
                width=15,
                tooltip="Enable or disable line detection"
            )
        )

        # Manual detection button
        self.add_control(
            'manual_detection',
            ControlConfig(
                control_type=ControlType.BUTTON,
                text="Manual Detection",
                command=self._on_manual_detection,
                width=15,
                tooltip="Trigger manual detection"
            )
        )

        # Refresh button
        self.add_control(
            'refresh',
            ControlConfig(
                control_type=ControlType.BUTTON,
                text="Refresh",
                command=self._on_refresh,
                width=10,
                tooltip="Refresh detection results"
            )
        )

        # Clear overlays button
        self.add_control(
            'clear_overlays',
            ControlConfig(
                control_type=ControlType.BUTTON,
                text="Clear Overlays",
                command=self._on_clear_overlays,
                width=15,
                tooltip="Clear all overlay elements"
            )
        )

        # Confidence threshold slider
        self.add_control(
            'confidence_threshold',
            ControlConfig(
                control_type=ControlType.SLIDER,
                text="Confidence Threshold",
                width=200,
                tooltip="Set minimum confidence for detection results"
            )
        )

        # Auto-refresh checkbox
        self.add_control(
            'auto_refresh',
            ControlConfig(
                control_type=ControlType.CHECKBOX,
                text="Auto Refresh",
                tooltip="Automatically refresh detection results"
            )
        )

        # Layout controls
        self._layout_controls()

    def _layout_controls(self):
        """Layout controls in appropriate frames."""
        # Button row 1 (Toggle and Manual)
        button_row1 = ttk.Frame(self.button_frame)
        button_row1.pack(fill='x', pady=(0, 5))

        if 'toggle_detection' in self.controls:
            self.controls['toggle_detection'].pack(side='left', padx=(0, 5))

        if 'manual_detection' in self.controls:
            self.controls['manual_detection'].pack(side='left', padx=(0, 5))

        # Button row 2 (Refresh and Clear)
        button_row2 = ttk.Frame(self.button_frame)
        button_row2.pack(fill='x')

        if 'refresh' in self.controls:
            self.controls['refresh'].pack(side='left', padx=(0, 5))

        if 'clear_overlays' in self.controls:
            self.controls['clear_overlays'].pack(side='left', padx=(0, 5))

        # Settings controls
        if 'confidence_threshold' in self.controls:
            self._setup_slider_control('confidence_threshold', from_=0.0, to=1.0, value=0.5)

        if 'auto_refresh' in self.controls:
            self._setup_checkbox_control('auto_refresh')

    def add_control(self, control_id: str, config: ControlConfig) -> bool:
        """
        Add a new control to the manager.

        Args:
            control_id: Unique identifier for the control
            config: Control configuration

        Returns:
            True if control was added successfully
        """
        try:
            # Store configuration
            self.control_configs[control_id] = config
            self.loading_states[control_id] = LoadingState()

            # Create control based on type
            control = self._create_control(config)

            if control:
                self.controls[control_id] = control
                logger.info(f"Control added: {control_id}")
                return True
            else:
                logger.error(f"Failed to create control: {control_id}")
                return False

        except Exception as e:
            logger.error(f"Error adding control {control_id}: {e}")
            return False

    def _create_control(self, config: ControlConfig) -> Optional[ttk.Widget]:
        """Create a control widget based on configuration."""
        try:
            if config.control_type == ControlType.BUTTON:
                control = ttk.Button(
                    self.button_frame,
                    text=config.text,
                    command=config.command,
                    width=config.width,
                    state=self._get_tkinter_state(config.state)
                )

            elif config.control_type == ControlType.TOGGLE:
                control = ttk.Button(
                    self.button_frame,
                    text=config.text,
                    command=config.command,
                    width=config.width,
                    state=self._get_tkinter_state(config.state)
                )

            elif config.control_type == ControlType.SLIDER:
                # For sliders, we need to create a frame with label and slider
                frame = ttk.Frame(self.settings_frame)
                ttk.Label(frame, text=config.text).pack(side='left', padx=(0, 10))
                control = ttk.Scale(
                    frame,
                    from_=0.0,
                    to=1.0,
                    orient='horizontal',
                    state=self._get_tkinter_state(config.state)
                )
                control.pack(side='left', fill='x', expand=True)

            elif config.control_type == ControlType.CHECKBOX:
                control = ttk.Checkbutton(
                    self.settings_frame,
                    text=config.text,
                    state=self._get_tkinter_state(config.state)
                )

            elif config.control_type == ControlType.LABEL:
                control = ttk.Label(
                    self.settings_frame,
                    text=config.text,
                    font=self.fonts['normal']
                )

            else:
                logger.warning(f"Unsupported control type: {config.control_type}")
                return None

            return control

        except Exception as e:
            logger.error(f"Error creating control: {e}")
            return None

    def _setup_slider_control(self, control_id: str, from_=0.0, to=1.0, value=0.5):
        """Setup a slider control with label and value display."""
        try:
            config = self.control_configs[control_id]
            frame = ttk.Frame(self.settings_frame)
            frame.pack(fill='x', pady=2)

            # Label
            label = ttk.Label(frame, text=config.text, font=self.fonts['normal'])
            label.pack(side='left', padx=(0, 10))

            # Slider
            slider = ttk.Scale(
                frame,
                from_=from_,
                to=to,
                orient='horizontal',
                value=value,
                state=self._get_tkinter_state(config.state)
            )
            slider.pack(side='left', fill='x', expand=True, padx=(0, 10))

            # Value label
            value_label = ttk.Label(frame, text=f"{value:.2f}", width=5)
            value_label.pack(side='left')

            # Update control reference to the frame
            self.controls[control_id] = frame

            # Bind slider update
            def on_slider_change(val):
                value_label.config(text=f"{float(val):.2f}")
                self._trigger_callbacks(f'{control_id}_changed', float(val))

            slider.config(command=on_slider_change)

        except Exception as e:
            logger.error(f"Error setting up slider control {control_id}: {e}")

    def _setup_checkbox_control(self, control_id: str):
        """Setup a checkbox control."""
        try:
            config = self.control_configs[control_id]
            frame = ttk.Frame(self.settings_frame)
            frame.pack(fill='x', pady=2)

            checkbox = ttk.Checkbutton(
                frame,
                text=config.text,
                command=lambda: self._on_checkbox_change(control_id),
                state=self._get_tkinter_state(config.state)
            )
            checkbox.pack(side='left')

            self.controls[control_id] = frame

        except Exception as e:
            logger.error(f"Error setting up checkbox control {control_id}: {e}")

    def _on_checkbox_change(self, control_id: str):
        """Handle checkbox change event."""
        try:
            # Get checkbox state
            frame = self.controls[control_id]
            for child in frame.winfo_children():
                if isinstance(child, ttk.Checkbutton):
                    state = child.instate(['selected'])
                    self._trigger_callbacks(f'{control_id}_changed', state)
                    break
        except Exception as e:
            logger.error(f"Error handling checkbox change: {e}")

    def _get_tkinter_state(self, control_state: ControlState) -> str:
        """Convert ControlState to tkinter state."""
        mapping = {
            ControlState.NORMAL: 'normal',
            ControlState.DISABLED: 'disabled',
            ControlState.LOADING: 'normal',
            ControlState.ERROR: 'normal'
        }
        return mapping.get(control_state, 'normal')

    def set_loading_state(self, control_id: str, loading: bool, loading_text: str = "Processing..."):
        """
        Set loading state for a control.

        Args:
            control_id: Control identifier
            loading: Whether control is in loading state
            loading_text: Text to display during loading
        """
        try:
            if control_id not in self.controls:
                return

            control = self.controls[control_id]
            config = self.control_configs[control_id]
            loading_state = self.loading_states[control_id]

            if loading:
                # Store original text and set loading text
                if hasattr(control, 'config') and 'text' in config.__dict__:
                    loading_state.original_text = config.text
                    control.config(text=loading_text)
                    config.text = loading_text

                # Update state
                config.state = ControlState.LOADING
                loading_state.loading = True
                loading_state.loading_text = loading_text

                # Disable control
                if hasattr(control, 'config'):
                    control.config(state='disabled')

            else:
                # Restore original text
                if loading_state.original_text and hasattr(control, 'config'):
                    control.config(text=loading_state.original_text)
                    config.text = loading_state.original_text

                # Update state
                config.state = ControlState.NORMAL
                loading_state.loading = False
                loading_state.original_text = None

                # Enable control
                if hasattr(control, 'config'):
                    control.config(state='normal')

            logger.debug(f"Loading state updated for {control_id}: {loading}")

        except Exception as e:
            logger.error(f"Error setting loading state for {control_id}: {e}")

    def set_control_enabled(self, control_id: str, enabled: bool):
        """
        Enable or disable a control.

        Args:
            control_id: Control identifier
            enabled: Whether control should be enabled
        """
        try:
            if control_id not in self.controls:
                return

            control = self.controls[control_id]
            config = self.control_configs[control_id]

            # Update configuration
            config.enabled = enabled
            config.state = ControlState.NORMAL if enabled else ControlState.DISABLED

            # Update control state
            if hasattr(control, 'config'):
                control.config(state=self._get_tkinter_state(config.state))

            logger.debug(f"Control {control_id} enabled: {enabled}")

        except Exception as e:
            logger.error(f"Error setting control enabled state: {e}")

    def set_control_visible(self, control_id: str, visible: bool):
        """
        Set visibility of a control.

        Args:
            control_id: Control identifier
            visible: Whether control should be visible
        """
        try:
            if control_id not in self.controls:
                return

            control = self.controls[control_id]
            config = self.control_configs[control_id]

            # Update configuration
            config.visible = visible

            # Update visibility
            if visible:
                control.pack(fill='x', pady=2)
            else:
                control.pack_forget()

            logger.debug(f"Control {control_id} visible: {visible}")

        except Exception as e:
            logger.error(f"Error setting control visibility: {e}")

    def update_control_text(self, control_id: str, text: str):
        """
        Update text for a control.

        Args:
            control_id: Control identifier
            text: New text
        """
        try:
            if control_id not in self.controls:
                return

            control = self.controls[control_id]
            config = self.control_configs[control_id]

            # Update configuration
            config.text = text

            # Update control text
            if hasattr(control, 'config') and isinstance(control, (ttk.Button, ttk.Label)):
                control.config(text=text)

            # For toggle buttons, update the text as well
            if control_id == 'toggle_detection':
                # Update toggle state based on text
                if 'Enable' in text:
                    config.state = ControlState.NORMAL
                else:
                    config.state = ControlState.NORMAL

            logger.debug(f"Control {control_id} text updated: {text}")

        except Exception as e:
            logger.error(f"Error updating control text: {e}")

    def get_control_value(self, control_id: str) -> Any:
        """
        Get current value of a control.

        Args:
            control_id: Control identifier

        Returns:
            Current control value or None
        """
        try:
            if control_id not in self.controls:
                return None

            control = self.controls[control_id]

            if isinstance(control, ttk.Scale):
                return control.get()
            elif isinstance(control, ttk.Checkbutton):
                # This is more complex due to ttk.Checkbutton structure
                frame = control
                for child in frame.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        return child.instate(['selected'])
            elif hasattr(control, 'config') and 'text' in str(type(control)):
                return control.cget('text')

            return None

        except Exception as e:
            logger.error(f"Error getting control value: {e}")
            return None

    def set_control_value(self, control_id: str, value: Any):
        """
        Set value for a control.

        Args:
            control_id: Control identifier
            value: New value
        """
        try:
            if control_id not in self.controls:
                return

            control = self.controls[control_id]

            if isinstance(control, ttk.Scale):
                control.set(value)
            elif isinstance(control, ttk.Checkbutton):
                # Update checkbox state
                frame = control
                for child in frame.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        if value:
                            child.state(['selected'])
                        else:
                            child.state(['!selected'])
                        break

            logger.debug(f"Control {control_id} value set: {value}")

        except Exception as e:
            logger.error(f"Error setting control value: {e}")

    def add_callback(self, event: str, callback: Callable):
        """
        Add a callback for control events.

        Args:
            event: Event name (e.g., 'toggle_detection', 'manual_detection')
            callback: Callback function
        """
        if event not in self.callbacks:
            self.callbacks[event] = []

        self.callbacks[event].append(callback)
        logger.debug(f"Callback added for event: {event}")

    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """Trigger callbacks for an event."""
        try:
            for callback in self.callbacks.get(event, []):
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in callback for {event}: {e}")
        except Exception as e:
            logger.error(f"Error triggering callbacks for {event}: {e}")

    # Default control handlers
    def _on_toggle_detection(self):
        """Handle toggle detection button click."""
        try:
            # In a real implementation, this would trigger the detection toggle
            current_text = self.control_configs['toggle_detection'].text
            if 'Enable' in current_text:
                self.update_control_text('toggle_detection', 'Disable Detection')
                self._trigger_callbacks('toggle_detection', True)
            else:
                self.update_control_text('toggle_detection', 'Enable Detection')
                self._trigger_callbacks('toggle_detection', False)
        except Exception as e:
            logger.error(f"Error in toggle detection handler: {e}")

    def _on_manual_detection(self):
        """Handle manual detection button click."""
        try:
            self.set_loading_state('manual_detection', True, 'Detecting...')
            self._trigger_callbacks('manual_detection')

            # Simulate processing delay
            def complete_detection():
                time.sleep(1.0)
                self.set_loading_state('manual_detection', False)
                self._trigger_callbacks('manual_detection_completed')

            thread = threading.Thread(target=complete_detection, daemon=True)
            thread.start()

        except Exception as e:
            logger.error(f"Error in manual detection handler: {e}")
            self.set_loading_state('manual_detection', False)

    def _on_refresh(self):
        """Handle refresh button click."""
        try:
            self.set_loading_state('refresh', True, 'Refreshing...')
            self._trigger_callbacks('refresh')

            # Simulate refresh delay
            def complete_refresh():
                time.sleep(0.5)
                self.set_loading_state('refresh', False)
                self._trigger_callbacks('refresh_completed')

            thread = threading.Thread(target=complete_refresh, daemon=True)
            thread.start()

        except Exception as e:
            logger.error(f"Error in refresh handler: {e}")
            self.set_loading_state('refresh', False)

    def _on_clear_overlays(self):
        """Handle clear overlays button click."""
        try:
            self._trigger_callbacks('clear_overlays')
        except Exception as e:
            logger.error(f"Error in clear overlays handler: {e}")

    def get_control_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current states of all controls.

        Returns:
            Dictionary with control states
        """
        states = {}
        for control_id, config in self.control_configs.items():
            states[control_id] = {
                'enabled': config.enabled,
                'visible': config.visible,
                'state': config.state.value,
                'text': config.text,
                'loading': self.loading_states[control_id].loading,
                'value': self.get_control_value(control_id)
            }
        return states

    def reset_all_controls(self):
        """Reset all controls to default state."""
        try:
            for control_id in self.controls:
                self.set_control_enabled(control_id, True)
                self.set_loading_state(control_id, False)
                self.set_control_visible(control_id, True)

            # Reset toggle detection text
            if 'toggle_detection' in self.controls:
                self.update_control_text('toggle_detection', 'Enable Detection')

            logger.info("All controls reset to default state")

        except Exception as e:
            logger.error(f"Error resetting controls: {e}")

    def enable_controls(self, control_ids: List[str]):
        """
        Enable specific controls.

        Args:
            control_ids: List of control IDs to enable
        """
        for control_id in control_ids:
            self.set_control_enabled(control_id, True)

    def disable_controls(self, control_ids: List[str]):
        """
        Disable specific controls.

        Args:
            control_ids: List of control IDs to disable
        """
        for control_id in control_ids:
            self.set_control_enabled(control_id, False)

    def get_controls_frame(self) -> ttk.Frame:
        """Get the main controls frame."""
        return self.main_frame