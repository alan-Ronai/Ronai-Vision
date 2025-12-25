"""
Scenario Hooks - Integration with Backend Scenario Manager and Rule Engine

This module provides hooks that are called from the detection pipeline
and transcription service to trigger scenario events.

The hooks communicate with:
1. Backend scenario API endpoints (for existing scenario manager)
2. Local ScenarioRuleEngine (for JSON-defined scenario rules)

Functions:
1. Report vehicle detections (with license plate)
2. Report armed person detections
3. Forward transcriptions for keyword detection
"""

import os
import logging
import asyncio
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")

# Import rule engine (lazy to avoid circular imports)
_rule_engine = None

def _get_rule_engine():
    """Get the scenario rule engine (lazy import)"""
    global _rule_engine
    if _rule_engine is None:
        from .scenario_rule_engine import get_scenario_rule_engine
        _rule_engine = get_scenario_rule_engine()
    return _rule_engine


@dataclass
class VehicleData:
    """Data structure for vehicle detection."""
    license_plate: str
    color: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    camera_id: Optional[str] = None
    track_id: Optional[int] = None
    confidence: float = 0.0
    bbox: Optional[List[float]] = None


@dataclass
class PersonData:
    """Data structure for armed person detection."""
    track_id: int
    armed: bool = False
    weapon_type: Optional[str] = None
    clothing: Optional[str] = None
    clothing_color: Optional[str] = None
    age_range: Optional[str] = None
    confidence: float = 0.0
    camera_id: Optional[str] = None
    bbox: Optional[List[float]] = None


class ScenarioHooks:
    """
    Hooks for integrating AI service detections with the scenario manager.

    This class provides methods to report detections to the backend,
    which will evaluate them against the scenario rules.
    """

    def __init__(self):
        self.backend_url = BACKEND_URL
        self.enabled = True
        self._client: Optional[httpx.AsyncClient] = None

        # Track what we've already reported to avoid duplicates
        self._reported_vehicles: set = set()  # track_ids
        self._reported_persons: set = set()  # track_ids

        logger.info(f"[ScenarioHooks] Initialized with backend: {self.backend_url}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def report_vehicle_detection(
        self,
        vehicle: VehicleData,
    ) -> Dict[str, Any]:
        """
        Report a vehicle detection to both the scenario manager and rule engine.

        The backend/rule engine will check if the license plate is in the stolen list
        and trigger the scenario if it matches.

        Args:
            vehicle: Vehicle detection data including license plate

        Returns:
            Response with scenario trigger status
        """
        if not self.enabled:
            return {"skipped": True, "reason": "hooks disabled"}

        # Check if already reported this vehicle
        if vehicle.track_id and vehicle.track_id in self._reported_vehicles:
            return {"skipped": True, "reason": "already reported"}

        if not vehicle.license_plate:
            return {"skipped": True, "reason": "no license plate"}

        payload = {
            "licensePlate": vehicle.license_plate,
            "color": vehicle.color,
            "make": vehicle.make,
            "model": vehicle.model,
            "cameraId": vehicle.camera_id,
            "trackId": vehicle.track_id,
            "confidence": vehicle.confidence,
            "bbox": vehicle.bbox,
        }

        # Try rule engine first (local, faster)
        try:
            rule_engine = _get_rule_engine()
            triggered = await rule_engine.handle_vehicle_detection(payload)

            if triggered:
                logger.info(
                    f"[ScenarioHooks] Vehicle triggered rule engine scenario: {vehicle.license_plate}"
                )
                if vehicle.track_id:
                    self._reported_vehicles.add(vehicle.track_id)
                return {"scenarioTriggered": True, "source": "rule_engine"}
        except Exception as e:
            logger.error(f"[ScenarioHooks] Rule engine error: {e}")

        # Also report to backend for legacy scenario manager
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.backend_url}/api/scenario/vehicle-detected",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()

                # Mark as reported if scenario was triggered
                if result.get("scenarioTriggered") and vehicle.track_id:
                    self._reported_vehicles.add(vehicle.track_id)

                logger.info(
                    f"[ScenarioHooks] Vehicle reported to backend: {vehicle.license_plate} "
                    f"- triggered: {result.get('scenarioTriggered')}"
                )

                return result
            else:
                logger.warning(
                    f"[ScenarioHooks] Vehicle report failed: {response.status_code}"
                )
                return {"error": f"HTTP {response.status_code}"}

        except httpx.ConnectError:
            logger.debug("[ScenarioHooks] Backend not available")
            return {"error": "backend not available"}
        except Exception as e:
            logger.error(f"[ScenarioHooks] Vehicle report error: {e}")
            return {"error": str(e)}

    async def report_armed_person(
        self,
        person: PersonData,
    ) -> Dict[str, Any]:
        """
        Report an armed person detection to both the scenario manager and rule engine.

        The backend/rule engine will track armed persons and trigger emergency mode
        when the threshold is reached.

        Args:
            person: Armed person detection data

        Returns:
            Response with threshold status
        """
        if not self.enabled:
            return {"skipped": True, "reason": "hooks disabled"}

        # Only report if armed
        if not person.armed:
            return {"skipped": True, "reason": "not armed"}

        # Check if already reported this person
        if person.track_id in self._reported_persons:
            return {"skipped": True, "reason": "already reported"}

        payload = {
            "trackId": person.track_id,
            "armed": person.armed,
            "weaponType": person.weapon_type,
            "clothing": person.clothing,
            "clothingColor": person.clothing_color,
            "ageRange": person.age_range,
            "confidence": person.confidence,
            "cameraId": person.camera_id,
            "bbox": person.bbox,
        }

        # Try rule engine first (local, faster)
        try:
            rule_engine = _get_rule_engine()
            triggered = await rule_engine.handle_armed_person(payload)

            if triggered:
                logger.info(
                    f"[ScenarioHooks] Armed person triggered rule engine threshold: track {person.track_id}"
                )
                self._reported_persons.add(person.track_id)
                return {"thresholdReached": True, "source": "rule_engine"}
            else:
                # Still mark as reported even if no threshold reached
                self._reported_persons.add(person.track_id)
        except Exception as e:
            logger.error(f"[ScenarioHooks] Rule engine error: {e}")

        # Also report to backend for legacy scenario manager
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.backend_url}/api/scenario/armed-person",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()

                # Mark as reported
                self._reported_persons.add(person.track_id)

                logger.info(
                    f"[ScenarioHooks] Armed person reported to backend: track {person.track_id} "
                    f"- threshold reached: {result.get('thresholdReached')}"
                )

                return result
            else:
                logger.warning(
                    f"[ScenarioHooks] Armed person report failed: {response.status_code}"
                )
                return {"error": f"HTTP {response.status_code}"}

        except httpx.ConnectError:
            logger.debug("[ScenarioHooks] Backend not available")
            return {"error": "backend not available"}
        except Exception as e:
            logger.error(f"[ScenarioHooks] Armed person report error: {e}")
            return {"error": str(e)}

    async def report_transcription(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Forward a transcription to both the rule engine and scenario manager for keyword detection.

        The rule engine/backend will check for stage-specific keywords and
        advance the scenario accordingly.

        Args:
            text: Transcription text

        Returns:
            Response with keyword match status
        """
        if not self.enabled:
            return {"skipped": True, "reason": "hooks disabled"}

        if not text or not text.strip():
            return {"skipped": True, "reason": "empty text"}

        # Try rule engine first (local, faster)
        try:
            rule_engine = _get_rule_engine()
            matched = await rule_engine.handle_transcription(text)

            if matched:
                logger.info(
                    f"[ScenarioHooks] Transcription keyword matched in rule engine: '{text[:50]}...'"
                )
                return {"keywordMatched": True, "source": "rule_engine"}
        except Exception as e:
            logger.error(f"[ScenarioHooks] Rule engine transcription error: {e}")

        # Also report to backend for legacy scenario manager
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.backend_url}/api/scenario/transcription",
                json={"text": text}
            )

            if response.status_code == 200:
                result = response.json()

                if result.get("keywordMatched"):
                    logger.info(
                        f"[ScenarioHooks] Transcription keyword matched in backend: '{text[:50]}...'"
                    )

                return result
            else:
                logger.warning(
                    f"[ScenarioHooks] Transcription report failed: {response.status_code}"
                )
                return {"error": f"HTTP {response.status_code}"}

        except httpx.ConnectError:
            logger.debug("[ScenarioHooks] Backend not available")
            return {"error": "backend not available"}
        except Exception as e:
            logger.error(f"[ScenarioHooks] Transcription report error: {e}")
            return {"error": str(e)}

    async def get_scenario_status(self) -> Dict[str, Any]:
        """
        Get current scenario status from backend.

        Returns:
            Current scenario state
        """
        try:
            client = await self._get_client()

            response = await client.get(
                f"{self.backend_url}/api/scenario/status"
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}

        except httpx.ConnectError:
            return {"active": False, "error": "backend not available"}
        except Exception as e:
            return {"active": False, "error": str(e)}

    async def check_plate_stolen(self, license_plate: str) -> Dict[str, Any]:
        """
        Check if a license plate is in the stolen vehicles list.

        Args:
            license_plate: License plate to check

        Returns:
            Dict with isStolen and stolenInfo fields
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.backend_url}/api/scenario/check-plate",
                json={"licensePlate": license_plate}
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"isStolen": False, "error": f"HTTP {response.status_code}"}

        except httpx.ConnectError:
            return {"isStolen": False, "error": "backend not available"}
        except Exception as e:
            return {"isStolen": False, "error": str(e)}

    def reset_reported(self):
        """Clear the reported tracks cache."""
        self._reported_vehicles.clear()
        self._reported_persons.clear()
        logger.info("[ScenarioHooks] Reported tracks cache cleared")

    def clear_camera_state(self, camera_id: str):
        """Clear all state related to a specific camera.

        Called when a camera is deleted/stopped to prevent:
        - Ghost tracks
        - Stale armed person counts
        - Events continuing after camera removal
        """
        # Clear reported persons/vehicles for this camera
        # Since we track by track_id (integer), we also need to clear the rule engine state

        # Reset all reported tracks (can't filter by camera since we only store track_id)
        # This is conservative but ensures clean state
        self._reported_vehicles.clear()
        self._reported_persons.clear()

        # Clear the rule engine scenario context for this camera
        try:
            rule_engine = _get_rule_engine()
            if rule_engine.active_scenario:
                # Filter out persons from this camera
                original_count = len(rule_engine.active_scenario.persons)
                rule_engine.active_scenario.persons = [
                    p for p in rule_engine.active_scenario.persons
                    if p.get('cameraId') != camera_id
                ]
                # Recalculate armed count
                rule_engine.active_scenario.armed_count = len([
                    p for p in rule_engine.active_scenario.persons
                    if p.get('armed')
                ])
                removed = original_count - len(rule_engine.active_scenario.persons)
                if removed > 0:
                    logger.info(f"[ScenarioHooks] Cleared {removed} persons from scenario for camera {camera_id}")
        except Exception as e:
            logger.debug(f"[ScenarioHooks] Rule engine clear error: {e}")

        logger.info(f"[ScenarioHooks] Cleared state for camera {camera_id}")

    def set_enabled(self, enabled: bool):
        """Enable or disable hooks."""
        self.enabled = enabled
        logger.info(f"[ScenarioHooks] Hooks {'enabled' if enabled else 'disabled'}")


# Global instance
_scenario_hooks: Optional[ScenarioHooks] = None


def get_scenario_hooks() -> ScenarioHooks:
    """Get or create the global scenario hooks instance."""
    global _scenario_hooks
    if _scenario_hooks is None:
        _scenario_hooks = ScenarioHooks()
    return _scenario_hooks


def init_scenario_hooks() -> ScenarioHooks:
    """Initialize the global scenario hooks instance."""
    global _scenario_hooks
    _scenario_hooks = ScenarioHooks()
    return _scenario_hooks
