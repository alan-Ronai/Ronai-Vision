"""
Scenario Rule Engine

Processes scenario rules defined in scenario_rules.json.
Manages multi-stage scenarios with automatic and manual transitions.
"""

import os
import json
import asyncio
import logging
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Backend API URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")


@dataclass
class ScenarioContext:
    """Runtime context for an active scenario"""
    scenario_id: str
    rule_id: str
    current_stage: str
    started_at: datetime
    vehicle: Optional[Dict] = None
    persons: List[Dict] = field(default_factory=list)
    armed_count: int = 0
    acknowledged: bool = False
    end_reason: Optional[str] = None
    stage_history: List[Dict] = field(default_factory=list)
    timers: Dict[str, asyncio.Task] = field(default_factory=dict)


class ScenarioRuleEngine:
    """
    Engine for processing scenario rules.

    Handles:
    - Loading scenario definitions from config
    - Managing scenario state machine
    - Processing triggers (detection, transcription, manual, timeout)
    - Executing stage actions
    - Coordinating with backend for UI updates
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._default_config_path()
        self.scenarios: Dict[str, Dict] = {}
        self.global_config: Dict = {}
        self.active_scenario: Optional[ScenarioContext] = None
        self.http_client = httpx.AsyncClient(timeout=10.0)
        self._load_config()

    def _default_config_path(self) -> str:
        """Get default config path"""
        return str(Path(__file__).parent.parent.parent / "config" / "scenario_rules.json")

    def _load_config(self):
        """Load scenario rules from config file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.scenarios = {}
            for scenario in data.get('scenarios', []):
                self.scenarios[scenario['_id']] = scenario

            # Load global config options
            self.global_config = data.get('config', {})

            logger.info(f"Loaded {len(self.scenarios)} scenario rules")
        except Exception as e:
            logger.error(f"Failed to load scenario rules: {e}")
            self.scenarios = {}
            self.global_config = {}

    @property
    def reannounce_threshold(self) -> int:
        """Get the threshold for re-announcing armed person count changes. Default: 2"""
        return self.global_config.get('reannounceThreshold', 2)

    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()

    def get_all_scenarios(self) -> List[Dict]:
        """Get all scenario definitions"""
        return list(self.scenarios.values())

    def get_scenario(self, scenario_id: str) -> Optional[Dict]:
        """Get a specific scenario definition"""
        return self.scenarios.get(scenario_id)

    def is_active(self) -> bool:
        """Check if any scenario is currently active"""
        return self.active_scenario is not None

    def get_active_scenario(self) -> Optional[Dict]:
        """Get the currently active scenario context"""
        if not self.active_scenario:
            return None
        return {
            'scenarioId': self.active_scenario.scenario_id,
            'ruleId': self.active_scenario.rule_id,
            'stage': self.active_scenario.current_stage,
            'startedAt': self.active_scenario.started_at.isoformat(),
            'vehicle': self.active_scenario.vehicle,
            'persons': self.active_scenario.persons,
            'armedCount': self.active_scenario.armed_count,
            'acknowledged': self.active_scenario.acknowledged,
            'stageHistory': self.active_scenario.stage_history
        }

    # =========================================================================
    # TRIGGER HANDLERS
    # =========================================================================

    async def handle_vehicle_detection(self, vehicle_data: Dict) -> bool:
        """
        Handle vehicle detection event.

        Returns True if this triggered a scenario start or transition.
        """
        # Check if plate is in stolen list for any enabled scenario
        for scenario_id, scenario in self.scenarios.items():
            if not scenario.get('enabled', True):
                continue

            # Check if scenario allows starting from vehicle detection
            if self.active_scenario and scenario.get('singleInstance', True):
                # Already have an active scenario
                continue

            # Check stolen vehicles list
            config = scenario.get('config', {})
            stolen_vehicles = config.get('stolenVehicles', [])

            plate = vehicle_data.get('licensePlate', '')
            for stolen in stolen_vehicles:
                if stolen.get('plate', '').upper() == plate.upper():
                    # Found a match! Start the scenario
                    logger.info(f"Stolen vehicle detected: {plate}, starting scenario {scenario_id}")

                    # Enrich vehicle data with stolen info
                    enriched_vehicle = {
                        **vehicle_data,
                        'color': stolen.get('color', vehicle_data.get('color')),
                        'make': stolen.get('make', vehicle_data.get('make')),
                        'model': stolen.get('model', vehicle_data.get('model')),
                        'stolenMatch': True
                    }

                    await self._start_scenario(scenario_id, enriched_vehicle)
                    return True

        return False

    async def handle_armed_person(self, person_data: Dict) -> bool:
        """
        Handle armed person detection.

        Can START a scenario if no scenario is active (alternative entry point).
        Can also trigger transitions within an active scenario.

        Returns True if this triggered a transition or scenario start.
        """
        # If no active scenario, check if we should start one from armed person detection
        if not self.active_scenario:
            started = await self._try_start_from_armed_person(person_data)
            if started:
                return True
            return False

        # Deduplicate: Check if this person (by track_id) is already in the list
        track_id = person_data.get('trackId') or person_data.get('track_id')
        existing_idx = None
        for i, p in enumerate(self.active_scenario.persons):
            p_track_id = p.get('trackId') or p.get('track_id')
            if p_track_id == track_id:
                existing_idx = i
                break

        if existing_idx is not None:
            # Update existing person data instead of adding duplicate
            self.active_scenario.persons[existing_idx] = person_data
            logger.debug(f"Updated existing armed person track {track_id}")
        else:
            # New person - add to context
            previous_count = self.active_scenario.armed_count
            self.active_scenario.persons.append(person_data)
            logger.debug(f"Added new armed person track {track_id}")

            # Update count
            self.active_scenario.armed_count = len([p for p in self.active_scenario.persons if p.get('armed')])

            logger.info(f"Armed person detected. Total: {self.active_scenario.armed_count}")

            # Re-announce if count increased significantly AFTER EMERGENCY_MODE was entered
            # (i.e., after the initial TTS announcement)
            # Uses configurable threshold from config (default: 2)
            if (self.active_scenario.current_stage == 'EMERGENCY_MODE' or
                self.active_scenario.current_stage in ['RESPONSE_INITIATED', 'DRONE_DISPATCHED']):
                count_increase = self.active_scenario.armed_count - previous_count
                if count_increase >= self.reannounce_threshold:
                    # Send updated TTS announcement
                    new_count = self.active_scenario.armed_count
                    persons_desc = self._build_persons_descriptions()
                    update_message = f"עדכון: זוהו {new_count} חמושים כעת. {persons_desc}"
                    await self._send_tts(update_message)
                    logger.info(f"Re-announced armed count: {previous_count} -> {new_count} (threshold: {self.reannounce_threshold})")

                    # Also send journal and backend update
                    await self._send_action_to_backend('journal', {
                        'severity': 'critical',
                        'title': f'עדכון: {new_count} חמושים זוהו',
                        'description': f'מספר החמושים עלה מ-{previous_count} ל-{new_count}\n{self._build_persons_summary()}'
                    })

        # Ensure count is current
        self.active_scenario.armed_count = len([p for p in self.active_scenario.persons if p.get('armed')])

        # Check for threshold transitions
        await self._check_threshold_transitions()

        # Notify backend of update
        await self._notify_backend('armed-person', {
            'person': person_data,
            'armedCount': self.active_scenario.armed_count
        })

        return True

    async def _try_start_from_armed_person(self, person_data: Dict) -> bool:
        """
        Try to start a scenario from an armed person detection.

        This is the alternative entry point that bypasses the stolen vehicle check.
        Looks for scenarios with IDLE -> ARMED_PERSON_DETECTED transition.

        Returns True if scenario was started.
        """
        for scenario_id, scenario in self.scenarios.items():
            if not scenario.get('enabled', True):
                continue

            # Skip if singleInstance and we already have an active scenario
            if self.active_scenario and scenario.get('singleInstance', True):
                continue

            # Find IDLE stage and check for armed person trigger
            stages = scenario.get('stages', [])
            idle_stage = next((s for s in stages if s.get('isInitial')), None)
            if not idle_stage:
                continue

            # Check transitions for armed person detection
            for transition in idle_stage.get('transitions', []):
                trigger = transition.get('trigger', {})
                if trigger.get('type') != 'detection':
                    continue

                conditions = trigger.get('conditions', {})
                if conditions.get('objectType') == 'person' and conditions.get('armed') == True:
                    # This scenario accepts armed person as entry point
                    target_stage = transition.get('to')
                    logger.info(f"Armed person detected, starting scenario {scenario_id} at stage {target_stage}")

                    # Start the scenario with armed person instead of vehicle
                    await self._start_scenario_from_armed(scenario_id, target_stage, person_data)
                    return True

        return False

    async def _start_scenario_from_armed(self, scenario_id: str, start_stage: str, person_data: Dict):
        """Start a scenario from armed person detection (alternative entry point)."""
        scenario = self.scenarios.get(scenario_id)
        if not scenario:
            return

        # Find the target stage
        stage = self._get_stage(scenario, start_stage)
        if not stage:
            logger.error(f"Start stage {start_stage} not found for scenario {scenario_id}")
            return

        # Create scenario context (no vehicle in this case)
        self.active_scenario = ScenarioContext(
            scenario_id=f"{scenario_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            rule_id=scenario_id,
            current_stage=start_stage,
            started_at=datetime.now(),
            vehicle=None  # No vehicle in armed person entry
        )

        # Add the first armed person to context
        self.active_scenario.persons.append(person_data)
        self.active_scenario.armed_count = 1

        logger.info(f"Started scenario {self.active_scenario.scenario_id} from armed person detection")

        # Notify backend
        await self._notify_backend('started', {
            'scenarioId': self.active_scenario.scenario_id,
            'ruleId': scenario_id,
            'vehicle': None,
            'entryPoint': 'armed_person',
            'firstArmedPerson': person_data
        })

        # Execute stage actions and check for auto-transition
        await self._enter_stage(stage)

    async def handle_transcription(self, text: str) -> bool:
        """
        Handle radio transcription for keyword matching.

        Returns True if a keyword triggered a transition.
        """
        if not self.active_scenario:
            return False

        scenario = self.scenarios.get(self.active_scenario.rule_id)
        if not scenario:
            return False

        # Get current stage
        current_stage = self._get_stage(scenario, self.active_scenario.current_stage)
        if not current_stage:
            return False

        # Check for transcription triggers
        for transition in current_stage.get('transitions', []):
            trigger = transition.get('trigger', {})
            if trigger.get('type') != 'transcription':
                continue

            keywords = trigger.get('keywords', [])
            text_lower = text.lower()

            for keyword in keywords:
                if keyword.lower() in text_lower:
                    logger.info(f"Keyword '{keyword}' detected, transitioning to {transition['to']}")
                    await self._transition_to(transition['to'])
                    return True

        return False

    async def handle_manual_action(self, action: str) -> bool:
        """
        Handle manual user action (acknowledge, false_alarm, etc.)

        Returns True if the action triggered a transition.
        """
        if not self.active_scenario:
            return False

        scenario = self.scenarios.get(self.active_scenario.rule_id)
        if not scenario:
            return False

        # Special handling for certain actions
        if action == 'acknowledge':
            self.active_scenario.acknowledged = True
        elif action == 'false_alarm':
            self.active_scenario.end_reason = 'false_alarm'
        elif action == 'end_scenario':
            self.active_scenario.end_reason = 'manual'

        # Get current stage and check for manual triggers
        current_stage = self._get_stage(scenario, self.active_scenario.current_stage)
        if not current_stage:
            return False

        for transition in current_stage.get('transitions', []):
            trigger = transition.get('trigger', {})
            if trigger.get('type') == 'manual' and trigger.get('action') == action:
                logger.info(f"Manual action '{action}' triggered transition to {transition['to']}")
                await self._transition_to(transition['to'], reason=action)
                return True

        return False

    async def handle_upload(self, upload_type: str, data: Dict) -> bool:
        """
        Handle file upload (e.g., soldier video).

        Returns True if this triggered a transition.
        """
        if not self.active_scenario:
            return False

        scenario = self.scenarios.get(self.active_scenario.rule_id)
        if not scenario:
            return False

        current_stage = self._get_stage(scenario, self.active_scenario.current_stage)
        if not current_stage:
            return False

        for transition in current_stage.get('transitions', []):
            trigger = transition.get('trigger', {})
            if trigger.get('type') == 'upload' and trigger.get('endpoint') == upload_type:
                logger.info(f"Upload '{upload_type}' triggered transition to {transition['to']}")
                await self._transition_to(transition['to'])
                return True

        return False

    # =========================================================================
    # SCENARIO LIFECYCLE
    # =========================================================================

    async def _start_scenario(self, scenario_id: str, vehicle_data: Dict):
        """Start a new scenario instance"""
        scenario = self.scenarios.get(scenario_id)
        if not scenario:
            logger.error(f"Scenario {scenario_id} not found")
            return

        # Find initial stage
        initial_stage = None
        for stage in scenario.get('stages', []):
            if stage.get('isInitial'):
                initial_stage = stage
                break

        if not initial_stage:
            logger.error(f"No initial stage found for scenario {scenario_id}")
            return

        # Create scenario context
        self.active_scenario = ScenarioContext(
            scenario_id=f"{scenario_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            rule_id=scenario_id,
            current_stage=initial_stage['id'],
            started_at=datetime.now(),
            vehicle=vehicle_data
        )

        logger.info(f"Started scenario {self.active_scenario.scenario_id}")

        # Notify backend
        await self._notify_backend('started', {
            'scenarioId': self.active_scenario.scenario_id,
            'ruleId': scenario_id,
            'vehicle': vehicle_data
        })

        # Execute initial stage actions and check for auto-transition
        await self._enter_stage(initial_stage)

    async def _transition_to(self, stage_id: str, reason: Optional[str] = None):
        """Transition to a new stage"""
        if not self.active_scenario:
            return

        scenario = self.scenarios.get(self.active_scenario.rule_id)
        if not scenario:
            return

        # Cancel any active timers
        self._cancel_timers()

        # Record history
        self.active_scenario.stage_history.append({
            'from': self.active_scenario.current_stage,
            'to': stage_id,
            'at': datetime.now().isoformat(),
            'reason': reason
        })

        # Update current stage
        old_stage = self.active_scenario.current_stage
        self.active_scenario.current_stage = stage_id

        if reason:
            self.active_scenario.end_reason = reason

        logger.info(f"Transitioned from {old_stage} to {stage_id}")

        # Notify backend of stage transition
        await self._notify_backend_transition(old_stage, stage_id, reason)

        # Get new stage and execute onEnter actions
        new_stage = self._get_stage(scenario, stage_id)
        if new_stage:
            await self._enter_stage(new_stage)

    async def _enter_stage(self, stage: Dict):
        """Execute onEnter actions for a stage"""
        logger.info(f"Entering stage: {stage['id']}")

        # Execute onEnter actions
        for action in stage.get('onEnter', []):
            await self._execute_action(action)

        # Set up timeout timers
        for transition in stage.get('transitions', []):
            trigger = transition.get('trigger', {})
            if trigger.get('type') == 'timeout':
                timeout_ms = trigger.get('timeout', 60000)
                reason = trigger.get('reason', 'timeout')
                self._start_timeout_timer(transition['to'], timeout_ms, reason)

        # Check for auto-transition
        if stage.get('autoTransition'):
            delay = stage.get('autoTransitionDelay', 0)
            for transition in stage.get('transitions', []):
                if transition.get('trigger', {}).get('type') == 'auto':
                    if delay > 0:
                        await asyncio.sleep(delay / 1000)
                    await self._transition_to(transition['to'])
                    break

        # Check if this is a final stage
        if stage.get('isFinal'):
            await self._end_scenario()

    async def _end_scenario(self):
        """Clean up and end the scenario"""
        if not self.active_scenario:
            return

        # Cancel timers
        self._cancel_timers()

        # Calculate duration
        duration = datetime.now() - self.active_scenario.started_at
        duration_ms = int(duration.total_seconds() * 1000)

        logger.info(f"Scenario ended: {self.active_scenario.scenario_id}, duration: {duration_ms}ms")

        # Clear armed status from all tracked persons
        # This ensures GID Store indicators update when alert is handled
        try:
            from services.reid import get_reid_tracker
            reid_tracker = get_reid_tracker()
            if reid_tracker:
                cleared = reid_tracker.clear_armed_status()
                if cleared > 0:
                    logger.info(f"Cleared armed status for {cleared} persons on scenario end")
        except Exception as e:
            logger.warning(f"Failed to clear armed status: {e}")

        # Also clear scenario hooks state
        try:
            from services.scenario import get_scenario_hooks
            hooks = get_scenario_hooks()
            if hooks:
                hooks.reset_reported()
        except Exception as e:
            logger.warning(f"Failed to reset scenario hooks: {e}")

        # Notify backend
        await self._notify_backend('ended', {
            'scenarioId': self.active_scenario.scenario_id,
            'duration': duration_ms,
            'reason': self.active_scenario.end_reason or 'completed',
            'vehicle': self.active_scenario.vehicle,
            'armedCount': self.active_scenario.armed_count
        })

        # Clear active scenario (with delay for cleanup)
        await asyncio.sleep(5)
        self.active_scenario = None

    async def reset(self):
        """Force reset the scenario engine"""
        self._cancel_timers()
        self.active_scenario = None
        await self._notify_backend('reset', {})
        logger.info("Scenario engine reset")

    # =========================================================================
    # ACTION EXECUTION
    # =========================================================================

    async def _execute_action(self, action: Dict):
        """Execute a single action from stage definition"""
        action_type = action.get('type')
        params = action.get('params', {})

        # Interpolate placeholders in params
        params = self._interpolate_params(params)

        logger.debug(f"Executing action: {action_type} with params: {params}")

        try:
            # Route to backend for UI actions via engine-event endpoint
            if action_type in ['alert_popup', 'danger_mode', 'emergency_modal',
                               'simulation', 'soldier_video_panel', 'new_camera_dialog',
                               'summary_popup', 'close_modal', 'play_sound',
                               'stop_all_sounds', 'camera_focus', 'cleanup']:
                await self._send_action_to_backend(action_type, params)

            # Handle recording (both local and send to backend)
            elif action_type == 'start_recording':
                await self._send_action_to_backend(action_type, params)
                await self._start_recording(params)

            elif action_type == 'stop_recording':
                await self._send_action_to_backend(action_type, params)
                await self._stop_recording(params)

            # Handle TTS (both local and send to backend)
            elif action_type == 'tts':
                await self._send_action_to_backend(action_type, params)
                await self._send_tts(params.get('message', ''))

            # Handle journal (via engine-event)
            elif action_type == 'journal':
                await self._send_action_to_backend(action_type, params)

            # Handle delays
            elif action_type == 'delay':
                await asyncio.sleep(params.get('ms', 0) / 1000)

            # Context storage
            elif action_type == 'store_context':
                # Already handled by detection handlers
                pass

            else:
                logger.warning(f"Unknown action type: {action_type}")

        except Exception as e:
            logger.error(f"Action execution error ({action_type}): {e}")

    def _interpolate_params(self, params: Dict) -> Dict:
        """Replace placeholders in params with actual values"""
        if not self.active_scenario:
            return params

        result = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Replace placeholders
                value = value.replace('{armedCount}', str(self.active_scenario.armed_count))

                # Vehicle info
                if self.active_scenario.vehicle:
                    for vkey, vval in self.active_scenario.vehicle.items():
                        value = value.replace(f'{{vehicle.{vkey}}}', str(vval or 'לא זוהה'))
                    # Vehicle info summary
                    if '{vehicleInfo}' in value:
                        v = self.active_scenario.vehicle
                        vehicle_info = f"רכב: {v.get('licensePlate', 'לא זוהה')} | צבע: {v.get('color', 'לא זוהה')} | {v.get('make', '')} {v.get('model', '')}".strip()
                        value = value.replace('{vehicleInfo}', vehicle_info)
                    value = value.replace('{hasVehicle}', 'true')
                else:
                    # No vehicle - clear vehicle placeholders
                    value = value.replace('{vehicleInfo}', '')
                    value = value.replace('{hasVehicle}', 'false')
                    # Clear any remaining vehicle.X placeholders
                    import re
                    value = re.sub(r'\{vehicle\.[^}]+\}', 'לא זוהה', value)

                # First armed person info (for armed-person variant)
                if self.active_scenario.persons and len(self.active_scenario.persons) > 0:
                    first_person = self.active_scenario.persons[0]
                    for pkey, pval in first_person.items():
                        value = value.replace(f'{{firstArmedPerson.{pkey}}}', str(pval or 'לא זוהה'))
                else:
                    import re
                    value = re.sub(r'\{firstArmedPerson\.[^}]+\}', 'לא זוהה', value)

                # Generate persons summary
                if '{persons_summary}' in value:
                    summary = self._build_persons_summary()
                    value = value.replace('{persons_summary}', summary if summary else 'אין נתונים')

                if '{persons_descriptions}' in value:
                    desc = self._build_persons_descriptions()
                    value = value.replace('{persons_descriptions}', desc if desc else 'אין תיאור')

                if '{end_reason}' in value:
                    reason_map = {
                        'false_alarm': 'אזעקת שווא',
                        'neutralized': 'נוטרל',
                        'manual': 'סיום ידני'
                    }
                    reason = self.active_scenario.end_reason or 'הסתיים'
                    value = value.replace('{end_reason}', reason_map.get(reason, reason))

                if '{duration}' in value:
                    if self.active_scenario.started_at:
                        duration = datetime.now() - self.active_scenario.started_at
                        minutes = int(duration.total_seconds() // 60)
                        seconds = int(duration.total_seconds() % 60)
                        value = value.replace('{duration}', f'{minutes} דקות, {seconds} שניות')

            result[key] = value

        return result

    def _build_persons_summary(self) -> str:
        """Build a text summary of detected persons"""
        if not self.active_scenario or not self.active_scenario.persons:
            return ''

        lines = []
        for i, person in enumerate(self.active_scenario.persons, 1):
            parts = [f'חמוש #{i}']
            if person.get('weaponType') and person['weaponType'] != 'לא זוהה':
                parts.append(f"נשק: {person['weaponType']}")
            if person.get('clothing') and person['clothing'] != 'לא זוהה':
                parts.append(f"לבוש: {person['clothing']}")
            if person.get('clothingColor') and person['clothingColor'] != 'לא זוהה':
                parts.append(f"צבע: {person['clothingColor']}")
            if person.get('ageRange') and person['ageRange'] != 'לא זוהה':
                parts.append(f"גיל: {person['ageRange']}")
            if person.get('cameraId'):
                parts.append(f"מצלמה: {person['cameraId']}")
            lines.append(' | '.join(parts))

        return '\n'.join(lines)

    def _build_persons_descriptions(self) -> str:
        """Build TTS-friendly descriptions of persons"""
        if not self.active_scenario or not self.active_scenario.persons:
            return ''

        descriptions = []
        for i, person in enumerate(self.active_scenario.persons, 1):
            parts = []
            if person.get('clothing'):
                parts.append(f"אדם ב{person['clothing']}")
            if person.get('weaponType'):
                parts.append(f"עם {person['weaponType']}")
            if parts:
                descriptions.append(' '.join(parts))

        return ', '.join(descriptions)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_stage(self, scenario: Dict, stage_id: str) -> Optional[Dict]:
        """Get stage definition by ID"""
        for stage in scenario.get('stages', []):
            if stage['id'] == stage_id:
                return stage
        return None

    async def _check_threshold_transitions(self):
        """Check if any threshold-based transitions should occur"""
        if not self.active_scenario:
            return

        scenario = self.scenarios.get(self.active_scenario.rule_id)
        if not scenario:
            return

        current_stage = self._get_stage(scenario, self.active_scenario.current_stage)
        if not current_stage:
            return

        for transition in current_stage.get('transitions', []):
            trigger = transition.get('trigger', {})
            if trigger.get('type') != 'threshold':
                continue

            conditions = trigger.get('conditions', {})

            # Check armed person count threshold
            armed_threshold = conditions.get('armedPersonCount', {})
            if 'gte' in armed_threshold:
                if self.active_scenario.armed_count >= armed_threshold['gte']:
                    logger.info(f"Armed threshold reached ({self.active_scenario.armed_count}), transitioning to {transition['to']}")
                    await self._transition_to(transition['to'])
                    return

    def _start_timeout_timer(self, target_stage: str, timeout_ms: int, reason: str):
        """Start a timeout timer for stage transition"""
        async def timeout_callback():
            await asyncio.sleep(timeout_ms / 1000)
            if self.active_scenario and self.active_scenario.current_stage != target_stage:
                logger.info(f"Timeout triggered, transitioning to {target_stage}")
                await self._transition_to(target_stage, reason=reason)

        timer = asyncio.create_task(timeout_callback())
        self.active_scenario.timers[f'timeout_{target_stage}'] = timer

    def _cancel_timers(self):
        """Cancel all active timers"""
        if not self.active_scenario:
            return

        for name, timer in self.active_scenario.timers.items():
            if not timer.done():
                timer.cancel()

        self.active_scenario.timers.clear()

    async def _send_action_to_backend(self, action: str, params: Dict):
        """Send action to backend engine-event endpoint"""
        try:
            url = f"{BACKEND_URL}/api/scenario/engine-event"

            # Build context for the backend
            context = {}
            if self.active_scenario:
                context = {
                    'vehicle': self.active_scenario.vehicle,
                    'persons': self.active_scenario.persons,
                    'armedCount': self.active_scenario.armed_count,
                    'endReason': self.active_scenario.end_reason,
                    'duration': None
                }
                if self.active_scenario.started_at:
                    duration = datetime.now() - self.active_scenario.started_at
                    context['duration'] = int(duration.total_seconds() * 1000)

            payload = {
                'action': action,
                'params': params,
                'scenarioId': self.active_scenario.scenario_id if self.active_scenario else None,
                'stage': self.active_scenario.current_stage if self.active_scenario else None,
                'context': context
            }

            response = await self.http_client.post(url, json=payload)
            if response.status_code != 200:
                logger.warning(f"Backend action {action} failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send action to backend: {e}")

    async def _notify_backend_transition(self, from_stage: str, to_stage: str, trigger: Optional[str] = None):
        """Send stage transition notification to backend"""
        try:
            url = f"{BACKEND_URL}/api/scenario/engine-transition"

            context = {}
            if self.active_scenario:
                context = {
                    'vehicle': self.active_scenario.vehicle,
                    'persons': self.active_scenario.persons,
                    'armedCount': self.active_scenario.armed_count,
                }

            payload = {
                'scenarioId': self.active_scenario.scenario_id if self.active_scenario else None,
                'fromStage': from_stage,
                'toStage': to_stage,
                'trigger': trigger,
                'context': context
            }

            response = await self.http_client.post(url, json=payload)
            if response.status_code != 200:
                logger.warning(f"Backend transition notification failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to notify backend of transition: {e}")

    async def _send_tts(self, message: str):
        """Generate TTS audio and transmit via radio"""
        try:
            # Import TTS service
            from ..tts_service import get_tts_service
            from ..radio.radio_transmit import transmit_audio_internal

            tts = get_tts_service()
            audio_path = await tts.generate(message)

            # Transmit via radio if audio was generated
            if audio_path:
                try:
                    # Read the audio file and transmit directly (no HTTP)
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()

                    if len(audio_data) > 44:  # More than just WAV header
                        # Use internal function for direct transmission
                        result = await transmit_audio_internal(
                            audio_data=audio_data,
                            format="wav",
                            priority="high"
                        )

                        if result.get("success"):
                            logger.info(f"📻 TTS transmitted via radio: {message[:50]}...")
                        else:
                            logger.warning(f"📻 Radio transmit failed: {result.get('error')}")
                    else:
                        logger.warning(f"Audio file too small, skipping radio transmit: {len(audio_data)} bytes")
                except Exception as radio_err:
                    logger.error(f"📻 Radio transmit error: {radio_err}")

        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def _start_recording(self, params: Dict):
        """Start video recording on the scenario camera"""
        try:
            from ..recording.recording_manager import get_recording_manager

            recording_manager = get_recording_manager()
            if not recording_manager:
                logger.warning("Recording manager not available")
                return

            pre_buffer = params.get('preBuffer', 30)
            duration = params.get('duration', 60)

            # Record on the camera where the scenario is happening
            # Try vehicle first (stolen-vehicle variant), then persons (armed-person variant)
            camera_id = None
            if self.active_scenario:
                if self.active_scenario.vehicle:
                    camera_id = self.active_scenario.vehicle.get('cameraId')
                elif self.active_scenario.persons and len(self.active_scenario.persons) > 0:
                    # Armed-person variant: get camera from first armed person
                    camera_id = self.active_scenario.persons[0].get('cameraId')

            if not camera_id:
                logger.warning("🎬 No camera ID available for recording (no vehicle or person)")
                return

            recording_id = recording_manager.start_recording(
                camera_id=camera_id,
                duration=duration,
                pre_buffer=pre_buffer,
                trigger_reason="scenario_emergency",
                metadata={
                    'scenario_id': self.active_scenario.scenario_id if self.active_scenario else None,
                    'stage': self.active_scenario.current_stage if self.active_scenario else None,
                    'vehicle': self.active_scenario.vehicle if self.active_scenario else None,
                    'armed_count': self.active_scenario.armed_count if self.active_scenario else 0
                }
            )
            if recording_id:
                logger.info(f"🎬 Started recording {recording_id} on camera {camera_id}")
            else:
                logger.warning(f"🎬 Could not start recording on camera {camera_id} (may already be recording)")

        except Exception as e:
            logger.error(f"Start recording error: {e}")

    async def _stop_recording(self, params: Dict):
        """Stop video recording on cameras"""
        try:
            from ..recording.recording_manager import get_recording_manager

            recording_manager = get_recording_manager()
            if not recording_manager:
                logger.warning("Recording manager not available")
                return

            # Stop all active recordings
            stats = recording_manager.get_stats()
            active_recordings = stats.get('active_recordings', {})

            for camera_id in list(active_recordings.keys()):
                if recording_manager.stop_recording(camera_id):
                    logger.info(f"🎬 Stopped recording on camera {camera_id}")

        except Exception as e:
            logger.error(f"Stop recording error: {e}")

    async def _create_journal_entry(self, params: Dict):
        """Create journal entry via backend"""
        try:
            url = f"{BACKEND_URL}/api/events"
            payload = {
                'type': 'scenario',
                'severity': params.get('severity', 'info'),
                'title': params.get('title', 'אירוע תרחיש'),
                'description': params.get('description', ''),
                'cameraId': self.active_scenario.vehicle.get('cameraId') if self.active_scenario and self.active_scenario.vehicle else None,
                'metadata': {
                    'scenarioId': self.active_scenario.scenario_id if self.active_scenario else None,
                    'stage': self.active_scenario.current_stage if self.active_scenario else None
                }
            }

            await self.http_client.post(url, json=payload)
        except Exception as e:
            logger.error(f"Journal entry error: {e}")

    async def close(self):
        """Close the engine and clean up resources"""
        self._cancel_timers()
        await self.http_client.aclose()


# Singleton instance
_engine: Optional[ScenarioRuleEngine] = None


def get_scenario_rule_engine() -> ScenarioRuleEngine:
    """Get or create scenario rule engine singleton"""
    global _engine
    if _engine is None:
        _engine = ScenarioRuleEngine()
    return _engine


async def init_scenario_rule_engine() -> ScenarioRuleEngine:
    """Initialize the scenario rule engine"""
    global _engine
    _engine = ScenarioRuleEngine()
    return _engine
