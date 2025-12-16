"""Operational event logging system.

This module provides high-level operational logging for important events like:
- New person/vehicle detected
- Armed person detected
- Track analysis completed
- Zone violations
- System events

Separate from technical/debug logs, these are product-level events for operators.
"""

import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict


class EventType(str, Enum):
    """Event types for operational logging."""

    DETECTION = "detection"  # New object detected
    ALERT = "alert"  # Alert triggered (armed person, zone violation, etc.)
    ANALYSIS = "analysis"  # AI analysis completed (Gemini, etc.)
    RECORDING = "recording"  # Recording started/stopped
    SYSTEM = "system"  # System events (startup, shutdown, error)


class EventSeverity(str, Enum):
    """Event severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class OperationalEvent:
    """Operational event record."""

    timestamp: float
    event_type: EventType
    severity: EventSeverity
    camera_id: Optional[str]
    track_id: Optional[int]
    message: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json_line(self) -> str:
        """Convert to JSON line (for append-only log file)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class OperationalLogger:
    """High-level operational event logger."""

    def __init__(
        self,
        log_dir: str = "output/logs",
        log_file: str = "operational.jsonl",
        console_output: bool = True,
    ):
        """Initialize operational logger.

        Args:
            log_dir: Directory for log files
            log_file: Log file name (JSONL format)
            console_output: Also print to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / log_file
        self.console_output = console_output

        # Python logger for technical logs
        self._logger = logging.getLogger(__name__)

        self._logger.info(f"OperationalLogger initialized: {self.log_file}")

    def log_event(
        self,
        event_type: EventType,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log an operational event.

        Args:
            event_type: Type of event
            message: Human-readable message
            severity: Event severity
            camera_id: Optional camera identifier
            track_id: Optional track ID
            details: Optional additional details
        """
        event = OperationalEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            camera_id=camera_id,
            track_id=track_id,
            message=message,
            details=details or {},
        )

        # Write to file (append-only JSONL)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(event.to_json_line() + "\n")
        except Exception as e:
            self._logger.error(f"Failed to write operational log: {e}")

        # Console output
        if self.console_output:
            self._print_event(event)

    def log_detection(
        self,
        camera_id: str,
        track_id: int,
        class_name: str,
        details: Optional[Dict] = None,
    ):
        """Log new object detection.

        Args:
            camera_id: Camera identifier
            track_id: Track ID
            class_name: Class name (person, car, etc.)
            details: Optional analysis details (Gemini response, etc.)
        """
        # Format details for display
        detail_str = ""
        if details:
            gemini_analysis = details.get("gemini_analysis")
            if gemini_analysis:
                # Format Hebrew analysis nicely
                if isinstance(gemini_analysis, dict):
                    detail_items = [f"{k}: {v}" for k, v in gemini_analysis.items() if v and k != "error" and k != "timestamp"]
                    detail_str = f", {', '.join(detail_items)}" if detail_items else ""

        message = f"אובייקט חדש זוהה במצלמה {camera_id}: {class_name} (ID: {track_id}){detail_str}"

        self.log_event(
            event_type=EventType.DETECTION,
            message=message,
            severity=EventSeverity.INFO,
            camera_id=camera_id,
            track_id=track_id,
            details=details or {},
        )

    def log_alert(
        self,
        camera_id: str,
        track_id: int,
        alert_type: str,
        alert_message: str,
        details: Optional[Dict] = None,
    ):
        """Log alert event.

        Args:
            camera_id: Camera identifier
            track_id: Track ID
            alert_type: Alert type (armed_person, zone_violation, etc.)
            alert_message: Alert message
            details: Optional additional details
        """
        severity = EventSeverity.CRITICAL if "armed" in alert_type else EventSeverity.WARNING

        message = f"התראה במצלמה {camera_id}: {alert_message} (ID: {track_id})"

        self.log_event(
            event_type=EventType.ALERT,
            message=message,
            severity=severity,
            camera_id=camera_id,
            track_id=track_id,
            details={**(details or {}), "alert_type": alert_type},
        )

    def log_analysis(
        self,
        camera_id: str,
        track_id: int,
        class_name: str,
        analysis_result: Dict,
    ):
        """Log AI analysis completion.

        Args:
            camera_id: Camera identifier
            track_id: Track ID
            class_name: Class name
            analysis_result: Gemini analysis result
        """
        # Format analysis result for Hebrew display
        if class_name == "car":
            model = analysis_result.get("דגם", "לא ידוע")
            plate = analysis_result.get("מספר_רישוי", "לא זוהה")
            color = analysis_result.get("צבע", "לא ידוע")
            message = f"זוהה רכב במצלמה {camera_id} (ID: {track_id}): דגם {model}, צבע {color}, מספר {plate}"
        elif class_name == "person":
            clothing = analysis_result.get("לבוש", "לא ידוע")
            items = analysis_result.get("פריטים_בידיים")
            items_str = f", מחזיק: {', '.join(items)}" if items else ""
            message = f"זוהה אדם במצלמה {camera_id} (ID: {track_id}): {clothing}{items_str}"
        else:
            message = f"ניתוח הושלם למצלמה {camera_id} (ID: {track_id})"

        self.log_event(
            event_type=EventType.ANALYSIS,
            message=message,
            severity=EventSeverity.INFO,
            camera_id=camera_id,
            track_id=track_id,
            details={"class_name": class_name, "analysis": analysis_result},
        )

    def log_recording(
        self,
        camera_id: str,
        action: str,  # "started" or "stopped"
        trigger_reason: Optional[str] = None,
        track_id: Optional[int] = None,
        output_path: Optional[str] = None,
    ):
        """Log recording event.

        Args:
            camera_id: Camera identifier
            action: "started" or "stopped"
            trigger_reason: Reason for recording
            track_id: Optional track ID
            output_path: Optional output file path
        """
        if action == "started":
            reason_str = f" (סיבה: {trigger_reason})" if trigger_reason else ""
            track_str = f", ID: {track_id}" if track_id else ""
            message = f"הקלטה החלה במצלמה {camera_id}{track_str}{reason_str}"
        else:
            path_str = f", שמור ב: {output_path}" if output_path else ""
            message = f"הקלטה הסתיימה במצלמה {camera_id}{path_str}"

        self.log_event(
            event_type=EventType.RECORDING,
            message=message,
            severity=EventSeverity.INFO,
            camera_id=camera_id,
            track_id=track_id,
            details={
                "action": action,
                "trigger_reason": trigger_reason,
                "output_path": output_path,
            },
        )

    def log_system_event(self, message: str, severity: EventSeverity = EventSeverity.INFO, details: Optional[Dict] = None):
        """Log system event.

        Args:
            message: System event message
            severity: Event severity
            details: Optional additional details
        """
        self.log_event(
            event_type=EventType.SYSTEM,
            message=message,
            severity=severity,
            details=details or {},
        )

    def _print_event(self, event: OperationalEvent):
        """Print event to console with formatting.

        Args:
            event: Event to print
        """
        # Color codes for severity
        colors = {
            EventSeverity.INFO: "\033[0m",  # Default
            EventSeverity.WARNING: "\033[93m",  # Yellow
            EventSeverity.CRITICAL: "\033[91m",  # Red
        }
        reset = "\033[0m"

        color = colors.get(event.severity, "")

        # Format timestamp
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp))

        # Build output
        parts = [
            f"{color}[{event.severity.upper()}]",
            f"[{time_str}]",
            f"[{event.event_type.upper()}]",
        ]

        if event.camera_id:
            parts.append(f"[{event.camera_id}]")

        if event.track_id is not None:
            parts.append(f"[Track:{event.track_id}]")

        parts.append(event.message)
        parts.append(reset)

        print(" ".join(parts))


# Global singleton instance
_operational_logger: Optional[OperationalLogger] = None


def get_operational_logger(
    log_dir: str = "output/logs",
    log_file: str = "operational.jsonl",
    console_output: bool = True,
) -> OperationalLogger:
    """Get or create global operational logger instance.

    Args:
        log_dir: Log directory (only used on first call)
        log_file: Log file name (only used on first call)
        console_output: Enable console output (only used on first call)

    Returns:
        OperationalLogger instance
    """
    global _operational_logger
    if _operational_logger is None:
        _operational_logger = OperationalLogger(
            log_dir=log_dir, log_file=log_file, console_output=console_output
        )
    return _operational_logger


def reset_operational_logger():
    """Reset global operational logger (for testing)."""
    global _operational_logger
    _operational_logger = None
