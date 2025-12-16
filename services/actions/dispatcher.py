"""Modular action dispatcher.

Loads action rules from config/actions.json and executes them when events occur
(e.g., armed_person_detected). Rules are declarative and easily extendable.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "actions.json",
)


def _safe_format(template: str, mapping: Dict[str, Any]) -> str:
    try:
        return str(template).format_map(
            {
                k: (", ".join(v) if isinstance(v, (list, tuple)) else v)
                for k, v in mapping.items()
            }
        )
    except Exception:
        return str(template)


class ActionDispatcher:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {"rules": []}

    def reload(self, path: Optional[str] = None):
        path = path or _CONFIG_PATH
        if not os.path.exists(path):
            logger.warning(f"Actions config not found at {path}; using empty rules")
            self._config = {"rules": []}
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
            if "rules" not in self._config:
                self._config = {"rules": []}
            logger.info(
                f"Loaded {len(self._config['rules'])} action rule(s) from {path}"
            )
        except Exception as e:
            logger.error(f"Failed to load actions config: {e}")
            self._config = {"rules": []}

    def dispatch(
        self, event_type: str, event: Dict[str, Any], track: Optional[Any] = None
    ):
        if not self._config:
            self.reload()
        rules: List[Dict[str, Any]] = self._config.get("rules", [])
        ev = dict(event)
        if isinstance(ev.get("weapon_types"), (list, tuple)):
            ev["weapon_count"] = len(ev["weapon_types"])  # computed count
        for rule in rules:
            if rule.get("on") != event_type:
                continue
            for action in rule.get("actions", []):
                self._execute_action(action, ev, track)

    def _execute_action(
        self, action: Dict[str, Any], event: Dict[str, Any], track: Optional[Any]
    ):
        atype = action.get("type")
        params = action.get("params", {})
        try:
            if atype == "add_tag" and track is not None:
                tag = _safe_format(params.get("tag", ""), event)
                if tag:
                    track.add_tag(tag)
            elif atype == "add_alert" and track is not None:
                alert_type = _safe_format(params.get("alert_type", "alert"), event)
                message = _safe_format(params.get("message", ""), event)
                severity = params.get("severity", "warning")
                track.add_alert(
                    alert_type=alert_type, message=message, severity=severity
                )
            elif atype == "set_attribute" and track is not None:
                key = params.get("key")
                value = params.get("value")
                if isinstance(value, str):
                    value = _safe_format(value, event)
                track.set_attribute(key, value)
            elif atype == "metadata_update" and track is not None:
                from services.tracker.metadata_manager import get_metadata_manager

                manager = get_metadata_manager()
                manager.update_track_metadata(
                    track.track_id, track.class_id, track.get_metadata_summary()
                )
            elif atype == "start_recording" and track is not None:
                from services.video.recording_integration import (
                    start_recording_for_track,
                )

                reason = params.get("reason", event.get("type"))
                camera_id = event.get("camera_id") or getattr(track, "camera_id", None)
                if camera_id:
                    start_recording_for_track(
                        track_id=track.track_id,
                        camera_id=camera_id,
                        event_type=reason,
                        context=event,
                    )
            elif atype == "stop_recording" and track is not None:
                from services.video.recording_integration import (
                    stop_recording_for_track,
                )

                camera_id = event.get("camera_id") or getattr(track, "camera_id", None)
                if camera_id:
                    stop_recording_for_track(
                        track_id=track.track_id, camera_id=camera_id
                    )
            elif atype == "log_event":
                from services.logging.operational_logger import (
                    get_operational_logger,
                    EventSeverity,
                )

                op = get_operational_logger()
                message = _safe_format(params.get("message", ""), event)
                severity_str = params.get("severity", "info").lower()
                severity = (
                    EventSeverity.CRITICAL
                    if severity_str == "critical"
                    else (
                        EventSeverity.WARNING
                        if severity_str == "warning"
                        else EventSeverity.INFO
                    )
                )
                camera_id = event.get("camera_id")
                track_id = event.get("track_id")
                if (
                    severity == EventSeverity.CRITICAL
                    and camera_id
                    and track_id is not None
                ):
                    op.log_alert(
                        camera_id=camera_id,
                        track_id=track_id,
                        alert_type=event.get("type", "event"),
                        alert_message=message,
                        details=event,
                    )
                else:
                    op.log_system_event(
                        message=message, severity=severity, details=event
                    )
            else:
                logger.debug(f"Unknown or no-op action: {atype}")
        except Exception as e:
            logger.error(f"Action '{atype}' failed: {e}")


_dispatcher: Optional[ActionDispatcher] = None


def get_dispatcher() -> ActionDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ActionDispatcher()
        _dispatcher.reload()
    return _dispatcher
