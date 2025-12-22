"""
Event Rule Engine - Evaluates conditions, runs pipeline, executes actions.

This is the core engine that processes event rules defined in the system.
Rules are loaded from the backend and evaluated against incoming events.

Flow:
1. Load rules from backend on startup
2. On event, find matching rules by condition type
3. Evaluate conditions (AND/OR logic)
4. If conditions met, run pipeline processors
5. Execute actions
"""

import asyncio
import json
import logging
import os
import time
import re
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")


@dataclass
class RuleContext:
    """
    Context passed through rule evaluation.
    Contains all data available for condition evaluation and action execution.
    """
    # Event identification
    event_type: str  # 'detection', 'new_track', 'track_lost', 'transcription', etc.

    # Camera info
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None

    # Object tracking info
    track_id: Optional[int] = None
    object_type: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[List[float]] = None

    # Object attributes (from Gemini analysis)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Detection counts
    person_count: int = 0
    vehicle_count: int = 0
    object_counts: Dict[str, int] = field(default_factory=dict)

    # For transcription events
    transcription: Optional[str] = None

    # Frame data for AI analysis
    frame: Optional[Any] = None  # numpy array

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    # Pipeline results accumulate here
    pipeline_results: Dict[str, Any] = field(default_factory=dict)

    # All detections in current frame
    detections: List[Dict] = field(default_factory=list)

    # Track duration (for track_lost events)
    track_duration: float = 0.0

    # Custom placeholders (for set_placeholder pipeline processor)
    placeholders: Dict[str, str] = field(default_factory=dict)

    # All tracked objects (for metadata_object_count condition)
    all_tracked_objects: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for template interpolation."""
        return {
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "cameraId": self.camera_id,  # Alias
            "camera_name": self.camera_name,
            "cameraName": self.camera_name,  # Alias
            "track_id": self.track_id,
            "trackId": self.track_id,  # Alias
            "object_type": self.object_type,
            "objectType": self.object_type,  # Alias
            "confidence": self.confidence,
            "bbox": self.bbox,
            "attributes": self.attributes,
            "person_count": self.person_count,
            "personCount": self.person_count,  # Alias
            "vehicle_count": self.vehicle_count,
            "vehicleCount": self.vehicle_count,  # Alias
            "object_counts": self.object_counts,
            "transcription": self.transcription,
            "timestamp": self.timestamp,
            "pipeline_results": self.pipeline_results,
            "detections": self.detections,
            "track_duration": self.track_duration,
            "placeholders": self.placeholders,
            # Flatten common attributes
            "armed": self.attributes.get("armed", False),
            "stolen": self.attributes.get("stolen", False),
            "threatLevel": self.attributes.get("threatLevel"),
            "weaponType": self.attributes.get("weaponType"),
            "count": self.person_count + self.vehicle_count,  # Total count
            # Object attributes for placeholders
            "object": self.attributes,
            "camera": {"id": self.camera_id, "name": self.camera_name},
        }


class RuleEngine:
    """
    Evaluates event rules against incoming events.

    The engine maintains a cache of rules loaded from the backend
    and provides methods to evaluate events against these rules.
    """

    def __init__(self):
        self.rules: List[Dict] = []
        self._debounce_cache: Dict[str, float] = {}
        self._aggregate_cache: Dict[str, List[Dict]] = {}
        self._periodic_cache: Dict[str, float] = {}  # Track last trigger time for periodic rules
        self._condition_handlers: Dict[str, Callable] = {}
        self._pipeline_handlers: Dict[str, Callable] = {}
        self._action_handlers: Dict[str, Callable] = {}
        self._last_load_time: float = 0
        self._rule_load_interval: float = 60.0  # Reload rules every 60 seconds

        # Background timer for periodic rules
        self._periodic_timer_task: Optional[asyncio.Task] = None
        self._periodic_timer_running: bool = False
        self._periodic_check_interval: float = 1.0  # Check every 1 second

        self._register_handlers()

    def _register_handlers(self):
        """Register built-in handlers for conditions, pipeline, and actions."""
        # Condition handlers
        self._condition_handlers = {
            "object_detected": self._eval_object_detected,
            "attribute_match": self._eval_attribute_match,
            "object_interaction": self._eval_object_interaction,
            "transcription_keyword": self._eval_transcription_keyword,
            "object_count": self._eval_object_count,
            "metadata_object_count": self._eval_metadata_object_count,
            "new_track": self._eval_new_track,
            "track_lost": self._eval_track_lost,
            "time_based": self._eval_time_based,
            "periodic_interval": self._eval_periodic_interval,
            "emergency_active": self._eval_emergency_active,
            "camera_status": self._eval_camera_status,
        }

        # Pipeline handlers
        self._pipeline_handlers = {
            "gemini_analysis": self._pipe_gemini_analysis,
            "filter": self._pipe_filter,
            "delay": self._pipe_delay,
            "debounce": self._pipe_debounce,
            "set_placeholder": self._pipe_set_placeholder,
            "transform": self._pipe_transform,
            "enrich": self._pipe_enrich,
            "aggregate": self._pipe_aggregate,
            "custom_script": self._pipe_custom_script,
        }

        # Action handlers
        self._action_handlers = {
            "system_alert": self._action_system_alert,
            "tts_radio": self._action_tts_radio,
            "start_recording": self._action_start_recording,
            "trigger_simulation": self._action_trigger_simulation,
            "emergency_mode": self._action_emergency_mode,
            "add_tag": self._action_add_tag,
            "set_attribute": self._action_set_attribute,
            "webhook": self._action_webhook,
            "log_event": self._action_log_event,
            "play_sound": self._action_play_sound,
            "send_notification": self._action_send_notification,
            "select_camera": self._action_select_camera,
            "auto_focus_camera": self._action_auto_focus_camera,
            "create_event": self._action_create_event,
        }

    async def load_rules(self, force: bool = False) -> bool:
        """
        Load rules from backend.

        Args:
            force: If True, force reload even if recently loaded

        Returns:
            True if rules were loaded successfully
        """
        now = time.time()

        # Skip if recently loaded (unless forced)
        if not force and (now - self._last_load_time) < self._rule_load_interval:
            return True

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{BACKEND_URL}/api/event-rules/active",
                    timeout=10.0
                )
                if response.status_code == 200:
                    self.rules = response.json()
                    self._last_load_time = now
                    logger.info(f"Loaded {len(self.rules)} active event rules")
                    return True
                else:
                    logger.warning(f"Failed to load rules: HTTP {response.status_code}")
                    return False
        except httpx.ConnectError:
            logger.debug("Backend not available for rule loading")
            return False
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            return False

    async def reload_rules(self):
        """Force reload rules from backend."""
        return await self.load_rules(force=True)

    # =========================================================================
    # PERIODIC TIMER - Background task for time-based rules
    # =========================================================================

    async def start_periodic_timer(self):
        """Start the background timer for periodic rules."""
        if self._periodic_timer_running:
            logger.debug("[RuleEngine] Periodic timer already running")
            return

        self._periodic_timer_running = True
        self._periodic_timer_task = asyncio.create_task(self._periodic_timer_loop())
        logger.info("[RuleEngine] Periodic timer started")

    async def stop_periodic_timer(self):
        """Stop the background timer for periodic rules."""
        self._periodic_timer_running = False
        if self._periodic_timer_task:
            self._periodic_timer_task.cancel()
            try:
                await self._periodic_timer_task
            except asyncio.CancelledError:
                pass
            self._periodic_timer_task = None
        logger.info("[RuleEngine] Periodic timer stopped")

    async def _periodic_timer_loop(self):
        """Background loop that checks and triggers periodic rules."""
        logger.info("[RuleEngine] Periodic timer loop started")

        while self._periodic_timer_running:
            try:
                # Note: Rules are loaded once at startup and reloaded via reload_rules()
                # when rules are created/updated/deleted. No need to reload here.

                # Find rules that have ONLY periodic_interval conditions
                periodic_rules = self._get_periodic_only_rules()

                for rule in periodic_rules:
                    try:
                        await self._process_periodic_rule(rule)
                    except Exception as e:
                        logger.error(f"[RuleEngine] Error processing periodic rule '{rule.get('name')}': {e}")

                # Sleep before next check
                await asyncio.sleep(self._periodic_check_interval)

            except asyncio.CancelledError:
                logger.info("[RuleEngine] Periodic timer loop cancelled")
                break
            except Exception as e:
                logger.error(f"[RuleEngine] Periodic timer error: {e}")
                await asyncio.sleep(self._periodic_check_interval)

        logger.info("[RuleEngine] Periodic timer loop ended")

    def _get_periodic_only_rules(self) -> List[Dict]:
        """Get rules that have ONLY periodic_interval (and optionally time_based) conditions."""
        time_conditions = {"periodic_interval", "time_based"}
        periodic_rules = []

        for rule in self.rules:
            if not rule.get("enabled", True):
                continue

            conditions = rule.get("conditions", {})
            items = conditions.get("items", [])

            if not items:
                continue

            # Check if ALL conditions are time-based AND at least one is periodic_interval
            has_periodic = False
            all_time_based = True

            for item in items:
                cond_type = item.get("type")
                if cond_type == "periodic_interval":
                    has_periodic = True
                if cond_type not in time_conditions:
                    all_time_based = False
                    break

            if has_periodic and all_time_based:
                periodic_rules.append(rule)

        return periodic_rules

    async def _process_periodic_rule(self, rule: Dict):
        """Process a single periodic rule - check conditions and execute actions."""
        # Create a minimal context for periodic rules (no detection data)
        context = RuleContext(
            event_type="periodic",
            camera_id=None,
            camera_name=None,
        )

        # Evaluate conditions
        if not self._evaluate_conditions(rule.get("conditions", {}), context):
            return  # Conditions not met (interval not elapsed)

        logger.info(f"[RuleEngine] Periodic rule '{rule.get('name')}' triggered!")

        # Run pipeline (if any)
        pipeline_passed = await self._run_pipeline(
            rule.get("pipeline", []),
            context,
            rule.get("_id", "unknown")
        )

        if not pipeline_passed:
            logger.debug(f"[RuleEngine] Periodic rule '{rule.get('name')}' stopped by pipeline")
            return

        # Execute actions
        action_results = await self._execute_actions(
            rule.get("actions", []),
            context
        )

        logger.info(f"[RuleEngine] Periodic rule '{rule.get('name')}' executed {len(action_results)} actions")

        # Record trigger
        await self._record_trigger(rule.get("_id"))

    def get_rules_for_event_type(self, event_type: str) -> List[Dict]:
        """
        Get rules that might match a specific event type.

        This is an optimization to avoid evaluating all rules for every event.
        """
        # Map event types to condition types
        type_mapping = {
            "detection": ["object_detected", "attribute_match", "object_count", "object_interaction"],
            "new_track": ["new_track", "object_detected"],
            "track_lost": ["track_lost"],
            "transcription": ["transcription_keyword"],
        }

        # Time-based conditions that should be evaluated on any event
        # These conditions control WHEN a rule fires, not WHAT triggers it
        time_conditions = ["periodic_interval", "time_based"]

        relevant_conditions = type_mapping.get(event_type, [])
        if not relevant_conditions:
            # Return all enabled rules if event type not mapped
            return [r for r in self.rules if r.get("enabled", True)]

        matching_rules = []
        for rule in self.rules:
            # Skip disabled rules (safety check - should already be filtered by backend)
            if not rule.get("enabled", True):
                continue

            conditions = rule.get("conditions", {})
            condition_items = conditions.get("items", [])

            for item in condition_items:
                cond_type = item.get("type")
                # Include rule if it has a relevant condition OR if it ONLY has time-based conditions
                if cond_type in relevant_conditions:
                    matching_rules.append(rule)
                    break
                # If rule only has time-based conditions, include it for all event types
                if cond_type in time_conditions:
                    # Check if ALL conditions are time-based
                    all_time_based = all(
                        c.get("type") in time_conditions
                        for c in condition_items
                    )
                    if all_time_based:
                        matching_rules.append(rule)
                        break

        return matching_rules

    async def process_event(self, context: RuleContext) -> List[Dict]:
        """
        Process an event through all matching rules.

        Args:
            context: Event context containing all relevant data

        Returns:
            List of action results from triggered rules
        """
        # Note: Rules are loaded once at startup and reloaded via reload_rules()
        # when rules are created/updated/deleted. No need to reload on every event.

        results = []

        # Get potentially matching rules
        candidate_rules = self.get_rules_for_event_type(context.event_type)

        for rule in candidate_rules:
            try:
                # Check if conditions match
                if not self._evaluate_conditions(rule.get("conditions", {}), context):
                    continue

                logger.info(f"Rule '{rule.get('name')}' conditions matched for {context.event_type}")

                # Run pipeline
                pipeline_passed = await self._run_pipeline(
                    rule.get("pipeline", []),
                    context,
                    rule.get("_id", "unknown")
                )

                if not pipeline_passed:
                    logger.debug(f"Rule '{rule.get('name')}' stopped by pipeline")
                    continue

                # Execute actions
                action_results = await self._execute_actions(
                    rule.get("actions", []),
                    context
                )

                results.append({
                    "rule_id": rule.get("_id"),
                    "rule_name": rule.get("name"),
                    "actions_executed": len(action_results),
                    "results": action_results
                })

                # Record trigger in backend
                await self._record_trigger(rule.get("_id"))

            except Exception as e:
                logger.error(f"Error processing rule '{rule.get('name')}': {e}", exc_info=True)

        return results

    def _evaluate_conditions(self, conditions: Dict, context: RuleContext) -> bool:
        """Evaluate rule conditions with AND/OR logic."""
        operator = conditions.get("operator", "AND")
        items = conditions.get("items", [])

        if not items:
            return True

        results = []
        for item in items:
            ctype = item.get("type")
            params = item.get("params", {})

            handler = self._condition_handlers.get(ctype)
            if handler:
                try:
                    result = handler(params, context)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Condition '{ctype}' error: {e}")
                    results.append(False)
            else:
                logger.warning(f"Unknown condition type: {ctype}")
                results.append(False)

        if operator == "AND":
            return all(results)
        else:  # OR
            return any(results)

    async def _run_pipeline(
        self,
        pipeline: List[Dict],
        context: RuleContext,
        rule_id: str
    ) -> bool:
        """
        Run pipeline processors in sequence.

        Returns False if pipeline should stop (e.g., filter failed, debounce active).
        """
        for step in pipeline:
            ptype = step.get("type")
            params = step.get("params", {})
            output_key = step.get("outputKey")

            handler = self._pipeline_handlers.get(ptype)
            if handler:
                try:
                    result = await handler(params, context, rule_id)

                    # Check if pipeline should stop
                    if result is False:
                        return False

                    # Store result if output key specified
                    if output_key and result is not None and result is not True:
                        context.pipeline_results[output_key] = result

                except Exception as e:
                    logger.error(f"Pipeline '{ptype}' error: {e}")
                    return False
            else:
                logger.warning(f"Unknown pipeline type: {ptype}")

        return True

    async def _execute_actions(
        self,
        actions: List[Dict],
        context: RuleContext
    ) -> List[Dict]:
        """Execute all actions for a rule."""
        results = []

        for action in actions:
            atype = action.get("type")
            params = action.get("params", {})

            handler = self._action_handlers.get(atype)
            if handler:
                try:
                    result = await handler(params, context)
                    results.append({
                        "type": atype,
                        "success": True,
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Action '{atype}' error: {e}")
                    results.append({
                        "type": atype,
                        "success": False,
                        "error": str(e)
                    })
            else:
                logger.warning(f"Unknown action type: {atype}")

        return results

    # =========================================================================
    # CONDITION HANDLERS
    # =========================================================================

    def _eval_object_detected(self, params: Dict, ctx: RuleContext) -> bool:
        """Check if specific object type is detected."""
        required_type = params.get("objectType")
        min_conf = params.get("minConfidence", 0)
        camera_id = params.get("cameraId")

        # Check camera filter
        if camera_id and ctx.camera_id != camera_id:
            return False

        # For single object context
        if ctx.object_type:
            type_match = self._match_object_type(ctx.object_type, required_type)
            if type_match and ctx.confidence >= min_conf:
                return True

        # Check in detections list
        for det in ctx.detections:
            det_type = det.get("class") or det.get("type")
            det_conf = det.get("confidence", 0)
            if self._match_object_type(det_type, required_type) and det_conf >= min_conf:
                return True

        return False

    def _match_object_type(self, actual: str, required: str) -> bool:
        """Match object type with support for groups like 'vehicle'."""
        if not actual or not required:
            return False

        actual = actual.lower()
        required = required.lower()

        if actual == required:
            return True

        # Vehicle group
        vehicle_types = {"car", "truck", "bus", "motorcycle", "bicycle"}
        if required == "vehicle" and actual in vehicle_types:
            return True

        return False

    def _eval_attribute_match(self, params: Dict, ctx: RuleContext) -> bool:
        """Check if attribute matches condition."""
        attr = params.get("attribute")
        operator = params.get("operator", "equals")
        value = params.get("value")

        actual = ctx.attributes.get(attr)

        # Handle exists/notExists operators
        if operator == "exists":
            return actual is not None
        if operator == "notExists":
            return actual is None

        # Compare values
        if operator == "equals":
            return actual == value
        elif operator == "notEquals":
            return actual != value
        elif operator == "contains":
            return value in str(actual) if actual else False
        elif operator == "greaterThan":
            try:
                return float(actual or 0) > float(value)
            except (ValueError, TypeError):
                return False
        elif operator == "lessThan":
            try:
                return float(actual or 0) < float(value)
            except (ValueError, TypeError):
                return False

        return False

    def _eval_object_interaction(self, params: Dict, ctx: RuleContext) -> bool:
        """Check if two objects are interacting."""
        # This requires spatial analysis of multiple detections
        obj_type_a = params.get("objectTypeA")
        obj_type_b = params.get("objectTypeB")
        interaction_type = params.get("interactionType", "proximity")
        threshold = params.get("threshold", 50)
        camera_id = params.get("cameraId")

        if camera_id and ctx.camera_id != camera_id:
            return False

        # Find objects of each type
        objects_a = [d for d in ctx.detections
                     if self._match_object_type(d.get("class", ""), obj_type_a)]
        objects_b = [d for d in ctx.detections
                     if self._match_object_type(d.get("class", ""), obj_type_b)]

        if not objects_a or not objects_b:
            return False

        # Check for interactions
        for a in objects_a:
            for b in objects_b:
                if a is b:
                    continue  # Skip same object

                bbox_a = a.get("bbox", [])
                bbox_b = b.get("bbox", [])

                if len(bbox_a) < 4 or len(bbox_b) < 4:
                    continue

                if interaction_type == "proximity":
                    # Check if centers are within threshold
                    center_a = ((bbox_a[0] + bbox_a[2]) / 2, (bbox_a[1] + bbox_a[3]) / 2)
                    center_b = ((bbox_b[0] + bbox_b[2]) / 2, (bbox_b[1] + bbox_b[3]) / 2)
                    distance = ((center_a[0] - center_b[0]) ** 2 +
                                (center_a[1] - center_b[1]) ** 2) ** 0.5
                    if distance <= threshold:
                        return True

                elif interaction_type == "overlap":
                    # Check if bboxes overlap
                    if (bbox_a[0] < bbox_b[2] and bbox_a[2] > bbox_b[0] and
                            bbox_a[1] < bbox_b[3] and bbox_a[3] > bbox_b[1]):
                        return True

        return False

    def _eval_transcription_keyword(self, params: Dict, ctx: RuleContext) -> bool:
        """Check if transcription contains keywords and/or word count threshold."""
        if not ctx.transcription:
            return False

        keywords = params.get("keywords", [])
        match_type = params.get("matchType", "any")
        case_sensitive = params.get("caseSensitive", False)

        # Word count parameters (Feature 5)
        count_mode = params.get("countMode", "disabled")
        count_operator = params.get("countOperator", "greaterOrEqual")
        count_threshold = params.get("countThreshold", 5)

        text = ctx.transcription if case_sensitive else ctx.transcription.lower()
        keywords_to_check = keywords if case_sensitive else [k.lower() for k in keywords]

        # Keyword matching result
        keyword_match = True  # Default to True if no keywords specified
        if keywords_to_check:
            matches = [kw in text for kw in keywords_to_check]
            if match_type == "any":
                keyword_match = any(matches)
            elif match_type == "all":
                keyword_match = all(matches)
            elif match_type == "exact":
                keyword_match = text in keywords_to_check
            elif match_type == "phrase":
                keyword_match = any(kw in text for kw in keywords_to_check)
            else:
                keyword_match = any(matches)

        # Word count matching result (Feature 5)
        count_match = True  # Default to True if count mode is disabled
        if count_mode != "disabled":
            words = text.split()

            if count_mode == "total_words":
                actual_count = len(words)
            elif count_mode == "keyword_occurrences":
                # Count occurrences of all keywords
                actual_count = sum(
                    words.count(kw) for kw in keywords_to_check
                ) if keywords_to_check else 0
            else:
                actual_count = 0

            # Compare against threshold
            count_match = self._compare_count(actual_count, count_operator, count_threshold)

        # Both conditions must pass (AND logic)
        # If no keywords, only count matters
        # If count mode disabled, only keywords matter
        if not keywords_to_check and count_mode == "disabled":
            return False  # At least one must be specified

        return keyword_match and count_match

    def _compare_count(self, actual: int, operator: str, threshold: int) -> bool:
        """Compare count against threshold using specified operator."""
        if operator == "greaterThan":
            return actual > threshold
        elif operator == "lessThan":
            return actual < threshold
        elif operator == "equals":
            return actual == threshold
        elif operator == "greaterOrEqual":
            return actual >= threshold
        elif operator == "lessOrEqual":
            return actual <= threshold
        return False

    def _eval_metadata_object_count(self, params: Dict, ctx: RuleContext) -> bool:
        """
        Check count of objects matching specific metadata criteria.
        Feature 2: Enables rules like "trigger when 3+ armed people detected".
        """
        obj_type = params.get("objectType", "")  # Empty = any type
        attribute = params.get("attribute")
        attribute_value = params.get("attributeValue")
        count_operator = params.get("countOperator", "greaterOrEqual")
        count_threshold = params.get("countThreshold", 1)
        scope = params.get("scope", "current_camera")

        if not attribute:
            return False

        # Determine which objects to check
        if scope == "all_cameras":
            objects_to_check = ctx.all_tracked_objects
        else:
            objects_to_check = ctx.detections

        # Filter and count matching objects
        matching_count = 0
        for obj in objects_to_check:
            # Check object type if specified
            if obj_type:
                detected_type = obj.get("class") or obj.get("type") or obj.get("objectType")
                if not self._match_object_type(detected_type or "", obj_type):
                    continue

            # Check attribute match
            # Try to get metadata from different possible locations
            metadata = obj.get("metadata", {})
            if not metadata:
                metadata = obj.get("attributes", {})
            if not metadata:
                # Attribute might be at the object level directly
                metadata = obj

            actual_value = metadata.get(attribute)

            # Handle different value types
            if isinstance(attribute_value, bool):
                if actual_value == attribute_value:
                    matching_count += 1
            elif isinstance(attribute_value, (int, float)):
                try:
                    if float(actual_value or 0) == float(attribute_value):
                        matching_count += 1
                except (ValueError, TypeError):
                    pass
            else:
                # String comparison
                if str(actual_value).lower() == str(attribute_value).lower():
                    matching_count += 1

        # Compare count against threshold
        return self._compare_count(matching_count, count_operator, count_threshold)

    def _eval_object_count(self, params: Dict, ctx: RuleContext) -> bool:
        """Check object count threshold."""
        obj_type = params.get("objectType", "any")
        operator = params.get("operator", "greaterThan")
        count = params.get("count", 0)
        camera_id = params.get("cameraId")

        if camera_id and ctx.camera_id != camera_id:
            return False

        # Get count based on type
        if obj_type == "any":
            actual_count = len(ctx.detections)
        elif obj_type == "person":
            actual_count = ctx.person_count or sum(
                1 for d in ctx.detections if d.get("class") == "person"
            )
        elif obj_type == "car":
            actual_count = sum(
                1 for d in ctx.detections if d.get("class") == "car"
            )
        elif obj_type == "vehicle":
            vehicle_types = {"car", "truck", "bus", "motorcycle", "bicycle"}
            actual_count = sum(
                1 for d in ctx.detections if d.get("class") in vehicle_types
            )
        else:
            actual_count = ctx.object_counts.get(obj_type, 0)

        # Compare
        if operator == "greaterThan":
            return actual_count > count
        elif operator == "lessThan":
            return actual_count < count
        elif operator == "equals":
            return actual_count == count
        elif operator == "greaterOrEqual":
            return actual_count >= count
        elif operator == "lessOrEqual":
            return actual_count <= count

        return False

    def _eval_new_track(self, params: Dict, ctx: RuleContext) -> bool:
        """Check if this is a new track."""
        obj_type = params.get("objectType", "any")
        camera_id = params.get("cameraId")

        if ctx.event_type != "new_track":
            return False
        if camera_id and ctx.camera_id != camera_id:
            return False
        if obj_type != "any":
            if not self._match_object_type(ctx.object_type or "", obj_type):
                return False

        return True

    def _eval_track_lost(self, params: Dict, ctx: RuleContext) -> bool:
        """Check if track was lost after minimum duration."""
        obj_type = params.get("objectType", "any")
        min_duration = params.get("minDuration", 0)
        camera_id = params.get("cameraId")

        if ctx.event_type != "track_lost":
            return False
        if camera_id and ctx.camera_id != camera_id:
            return False
        if obj_type != "any":
            if not self._match_object_type(ctx.object_type or "", obj_type):
                return False
        if ctx.track_duration < min_duration:
            return False

        return True

    def _eval_time_based(self, params: Dict, ctx: RuleContext) -> bool:
        """Check if current time is within allowed window."""
        start_time = params.get("startTime")
        end_time = params.get("endTime")
        days = params.get("days", ["0", "1", "2", "3", "4", "5", "6"])

        now = datetime.now()
        current_day = str(now.weekday())
        current_time = now.strftime("%H:%M")

        # Check day of week
        if current_day not in days:
            return False

        # Check time window
        if start_time and end_time:
            if start_time <= end_time:
                # Normal case: start before end
                if not (start_time <= current_time <= end_time):
                    return False
            else:
                # Overnight case: start after end (e.g., 22:00-06:00)
                if not (current_time >= start_time or current_time <= end_time):
                    return False

        return True

    def _eval_periodic_interval(self, params: Dict, ctx: RuleContext) -> bool:
        """
        Check if enough time has passed since last trigger.

        This condition is mainly used for testing - it triggers every X seconds/minutes/hours/days.
        Uses a cache to track when each rule was last triggered.

        Note: The cache key is based on interval params and camera_id to create unique timers
        per combination. Different rules with the same interval settings will share the same timer.
        """
        interval = params.get("interval", 30)
        unit = params.get("unit", "seconds")

        # Convert interval to seconds
        multipliers = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400
        }
        interval_seconds = interval * multipliers.get(unit, 1)

        # Create cache key from interval params and camera for uniqueness
        # This allows different cameras to have separate timers
        camera_suffix = f"_{ctx.camera_id}" if ctx.camera_id else ""
        cache_key = f"periodic_{interval}_{unit}{camera_suffix}"

        now = time.time()
        last_trigger = self._periodic_cache.get(cache_key, 0)
        elapsed = now - last_trigger

        if elapsed >= interval_seconds:
            # Enough time has passed - update cache and return True
            self._periodic_cache[cache_key] = now
            logger.info(f"[RuleEngine] Periodic interval TRIGGERED: {cache_key} (interval: {interval} {unit}, elapsed: {elapsed:.1f}s)")
            return True

        logger.debug(f"[RuleEngine] Periodic interval waiting: {cache_key} ({elapsed:.1f}s / {interval_seconds}s)")
        return False

    def _eval_emergency_active(self, params: Dict, ctx: RuleContext) -> bool:
        """Check if emergency mode is active."""
        # This would need to check backend state
        # For now, return based on context if available
        is_active = params.get("isActive", True)
        return ctx.attributes.get("emergency_active", False) == is_active

    def _eval_camera_status(self, params: Dict, ctx: RuleContext) -> bool:
        """Check camera status change."""
        camera_id = params.get("cameraId")
        required_status = params.get("status")

        if ctx.event_type != "camera_status":
            return False
        if camera_id and ctx.camera_id != camera_id:
            return False
        if ctx.attributes.get("status") != required_status:
            return False

        return True

    # =========================================================================
    # PIPELINE HANDLERS
    # =========================================================================

    async def _pipe_gemini_analysis(
        self, params: Dict, ctx: RuleContext, rule_id: str
    ) -> Any:
        """Run Gemini analysis."""
        if ctx.frame is None:
            return None

        try:
            # Import here to avoid circular dependency
            from services.gemini import get_gemini_analyzer

            gemini = get_gemini_analyzer()
            prompt_type = params.get("promptType", "scene_analysis")
            custom_prompt = params.get("customPrompt")

            if custom_prompt:
                result = await gemini.analyze_with_prompt(ctx.frame, custom_prompt)
            elif prompt_type == "person_description":
                result = await gemini.analyze_person(ctx.frame, ctx.bbox)
            elif prompt_type == "vehicle_identification":
                result = await gemini.analyze_vehicle(ctx.frame, ctx.bbox)
            elif prompt_type == "threat_assessment":
                result = await gemini.analyze_threat(ctx.frame)
            elif prompt_type == "weapon_verification":
                result = await gemini.verify_weapon(ctx.frame, ctx.bbox)
            else:
                result = await gemini.analyze_scene(ctx.frame)

            return result
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return None

    async def _pipe_filter(
        self, params: Dict, ctx: RuleContext, rule_id: str
    ) -> bool:
        """Filter/gate - return False to stop pipeline."""
        condition = params.get("condition", "true")

        # Create safe evaluation context
        eval_context = ctx.to_dict()
        eval_context.update(ctx.pipeline_results)

        try:
            # Simple expression evaluation (limited for safety)
            # Support basic comparisons: ==, !=, >, <, >=, <=, in, and, or, not
            result = self._safe_eval(condition, eval_context)
            return bool(result)
        except Exception as e:
            logger.error(f"Filter condition error: {e}")
            return False

    def _safe_eval(self, expression: str, context: Dict) -> Any:
        """
        Safely evaluate a simple expression.
        Only supports basic comparisons and logical operators.
        """
        # Replace context variables
        expr = expression
        for key, value in context.items():
            if isinstance(value, str):
                expr = expr.replace(f"context.{key}", f"'{value}'")
                expr = expr.replace(f"{key}", f"'{value}'")
            elif isinstance(value, bool):
                expr = expr.replace(f"context.{key}", str(value))
                expr = expr.replace(f"{key}", str(value))
            elif value is None:
                expr = expr.replace(f"context.{key}", "None")
            elif isinstance(value, (int, float)):
                expr = expr.replace(f"context.{key}", str(value))
                expr = expr.replace(f"{key}", str(value))

        # Use restricted eval
        allowed_names = {
            "True": True, "False": False, "None": None,
            "true": True, "false": False, "null": None
        }
        try:
            return eval(expr, {"__builtins__": {}}, allowed_names)
        except Exception:
            return False

    async def _pipe_delay(
        self, params: Dict, ctx: RuleContext, rule_id: str
    ) -> bool:
        """Add delay."""
        duration_ms = params.get("duration", 1000)
        await asyncio.sleep(duration_ms / 1000)
        return True

    async def _pipe_debounce(
        self, params: Dict, ctx: RuleContext, rule_id: str
    ) -> bool:
        """Debounce - prevent rapid re-triggering."""
        cooldown_ms = params.get("cooldownMs", 10000)
        key_template = params.get("key", "ruleId")

        # Build debounce key
        debounce_key = self._interpolate(key_template, ctx)
        debounce_key = f"{rule_id}_{debounce_key}"

        now = time.time() * 1000
        last_trigger = self._debounce_cache.get(debounce_key, 0)

        if now - last_trigger < cooldown_ms:
            logger.debug(f"Debounce active for key: {debounce_key}")
            return False  # Still in cooldown

        self._debounce_cache[debounce_key] = now

        # Clean old entries periodically
        if len(self._debounce_cache) > 1000:
            cutoff = now - 3600000  # 1 hour
            self._debounce_cache = {
                k: v for k, v in self._debounce_cache.items() if v > cutoff
            }

        return True

    async def _pipe_set_placeholder(
        self, params: Dict, ctx: RuleContext, rule_id: str
    ) -> bool:
        """
        Set a custom placeholder that can be used in subsequent pipeline steps and actions.
        Feature 4: Custom Placeholders in Event Pipeline.
        """
        name = params.get("name", "").strip()
        expression = params.get("expression", "")

        if not name:
            logger.warning("set_placeholder: No name provided")
            return True

        # Validate placeholder name (alphanumeric and underscore only)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            logger.warning(f"set_placeholder: Invalid name '{name}' (must be alphanumeric)")
            return True

        # Interpolate the expression with current context and existing placeholders
        value = self._interpolate_with_placeholders(expression, ctx)

        # Store the placeholder
        ctx.placeholders[name] = value
        logger.debug(f"set_placeholder: {name} = {value}")

        return True

    def _interpolate_with_placeholders(self, template: str, ctx: RuleContext) -> str:
        """
        Enhanced interpolation that includes custom placeholders and nested object access.
        Supports: {placeholder}, {object.field}, {camera.name}, {pipeline.result.field}
        """
        if not template:
            return ""

        try:
            data = ctx.to_dict()
            data.update(ctx.pipeline_results)
            data.update(ctx.placeholders)  # Custom placeholders take precedence

            result = template

            # Handle nested access patterns like {object.color} or {pipeline.result.field}
            nested_pattern = r'\{(\w+)\.(\w+)(?:\.(\w+))?\}'
            for match in re.finditer(nested_pattern, template):
                full_match = match.group(0)
                obj_name = match.group(1)
                field1 = match.group(2)
                field2 = match.group(3)

                # Get the object
                obj = data.get(obj_name)
                if isinstance(obj, dict):
                    value = obj.get(field1)
                    if field2 and isinstance(value, dict):
                        value = value.get(field2)
                    if value is not None:
                        result = result.replace(full_match, str(value))

            # Handle simple patterns like {placeholder}
            simple_pattern = r'\{(\w+)\}'
            for match in re.finditer(simple_pattern, result):
                key = match.group(1)
                if key in data:
                    value = data[key]
                    if value is not None:
                        result = result.replace(f"{{{key}}}", str(value))

            return result
        except Exception as e:
            logger.error(f"Error interpolating template: {e}")
            return template

    async def _pipe_transform(
        self, params: Dict, ctx: RuleContext, rule_id: str
    ) -> Dict:
        """Transform data."""
        mapping = params.get("mapping", {})
        result = {}
        for key, value_expr in mapping.items():
            result[key] = self._interpolate(str(value_expr), ctx)
        return result

    async def _pipe_enrich(
        self, params: Dict, ctx: RuleContext, rule_id: str
    ) -> Dict:
        """Enrich with additional data from backend."""
        source = params.get("source")
        fields = params.get("fields", [])

        try:
            async with httpx.AsyncClient() as client:
                if source == "tracked_objects" and ctx.track_id:
                    response = await client.get(
                        f"{BACKEND_URL}/api/tracked/{ctx.track_id}",
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if fields:
                            return {k: data.get(k) for k in fields if k in data}
                        return data

                elif source == "camera_info" and ctx.camera_id:
                    response = await client.get(
                        f"{BACKEND_URL}/api/cameras/{ctx.camera_id}",
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if fields:
                            return {k: data.get(k) for k in fields if k in data}
                        return data

        except Exception as e:
            logger.error(f"Enrich error: {e}")

        return {}

    async def _pipe_aggregate(
        self, params: Dict, ctx: RuleContext, rule_id: str
    ) -> bool:
        """Aggregate multiple events before continuing."""
        window_ms = params.get("windowMs", 5000)
        min_count = params.get("minCount", 2)
        group_by = params.get("groupBy", "cameraId")

        # Build aggregation key
        group_value = ctx.to_dict().get(group_by, "default")
        agg_key = f"{rule_id}_{group_value}"

        now = time.time() * 1000

        # Get or create event list for this key
        if agg_key not in self._aggregate_cache:
            self._aggregate_cache[agg_key] = []

        events = self._aggregate_cache[agg_key]

        # Remove old events outside window
        events = [e for e in events if now - e["timestamp"] < window_ms]
        events.append({"timestamp": now, "context": ctx.to_dict()})
        self._aggregate_cache[agg_key] = events

        # Check if we have enough events
        if len(events) >= min_count:
            # Clear the cache for this key
            self._aggregate_cache[agg_key] = []
            return True

        return False

    async def _pipe_custom_script(
        self, params: Dict, ctx: RuleContext, rule_id: str
    ) -> Any:
        """Run custom script (limited functionality for safety)."""
        # For security, custom scripts are not supported in production
        # This is a placeholder that logs a warning
        logger.warning("Custom scripts are disabled for security reasons")
        return None

    # =========================================================================
    # ACTION HANDLERS
    # =========================================================================

    async def _action_system_alert(self, params: Dict, ctx: RuleContext):
        """Send system alert to backend."""
        severity = params.get("severity", "info")
        title = self._interpolate(params.get("title", "Alert"), ctx)
        message = self._interpolate(params.get("message", ""), ctx)

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKEND_URL}/api/events",
                json={
                    "type": "alert" if severity == "critical" else "detection",
                    "severity": severity,
                    "title": title,
                    "description": message,
                    "cameraId": ctx.camera_id,
                    "metadata": {
                        "triggeredBy": "rule_engine",
                        "context": ctx.to_dict()
                    }
                },
                timeout=5.0
            )

        logger.info(f"System alert: [{severity}] {title}")

    async def _action_tts_radio(self, params: Dict, ctx: RuleContext):
        """Send TTS message to radio."""
        message = self._interpolate(params.get("message", ""), ctx)
        priority = params.get("priority", "normal")

        try:
            async with httpx.AsyncClient() as client:
                # Generate TTS
                tts_response = await client.post(
                    f"{BACKEND_URL.replace('3000', '8000')}/tts/generate",
                    json={"text": message, "language": "he"},
                    timeout=30.0
                )

                if tts_response.status_code == 200:
                    tts_data = tts_response.json()

                    # Transmit to radio
                    await client.post(
                        f"{BACKEND_URL.replace('3000', '8000')}/radio/transmit/audio",
                        json={
                            "audio_base64": tts_data["audio_base64"],
                            "sample_rate": tts_data.get("sample_rate", 16000),
                            "auto_ptt": True
                        },
                        timeout=30.0
                    )

                    logger.info(f"TTS radio: {message}")
        except Exception as e:
            logger.error(f"TTS radio error: {e}")

    async def _action_start_recording(self, params: Dict, ctx: RuleContext):
        """Start video recording with pre-buffer support.

        Uses the RecordingManager to capture video with frames from before
        the trigger event (pre-buffer) and continues recording for the
        specified duration.
        """
        duration = params.get("duration", 30)
        pre_buffer = params.get("preBuffer", 5)
        camera_id = params.get("cameraId") or ctx.camera_id

        if not camera_id:
            # For rules without camera context (like periodic rules), skip silently
            logger.debug("start_recording: No camera_id specified, skipping")
            return

        try:
            # Import recording manager
            from ..recording import get_recording_manager

            recording_manager = get_recording_manager()
            if not recording_manager:
                logger.warning("RecordingManager not initialized, recording skipped")
                return

            # Build trigger reason from context
            trigger_reason = f"Rule triggered for {ctx.event_type}"
            if ctx.object_type:
                trigger_reason += f" ({ctx.object_type})"

            # Build metadata from context
            metadata = {
                "event_type": ctx.event_type,
                "object_type": ctx.object_type,
                "person_count": ctx.person_count,
                "vehicle_count": ctx.vehicle_count,
                "track_id": ctx.track_id,
            }
            if ctx.attributes:
                metadata["attributes"] = ctx.attributes

            # Start recording (this is synchronous and fast)
            recording_id = recording_manager.start_recording(
                camera_id=camera_id,
                duration=duration,
                pre_buffer=pre_buffer,
                trigger_reason=trigger_reason,
                metadata=metadata
            )

            if recording_id:
                logger.info(f"Recording started: {recording_id} (camera={camera_id}, duration={duration}s, pre_buffer={pre_buffer}s)")

                # Emit event in background - don't block on HTTP response
                asyncio.create_task(self._emit_recording_started_event(
                    camera_id, recording_id, duration, pre_buffer
                ))
            else:
                logger.debug(f"Recording not started for camera {camera_id} (already recording)")

        except Exception as e:
            logger.error(f"start_recording error: {e}", exc_info=True)

    async def _emit_recording_started_event(
        self, camera_id: str, recording_id: str, duration: int, pre_buffer: int
    ):
        """Emit recording started event to backend (non-blocking background task)."""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BACKEND_URL}/api/events",
                    json={
                        "type": "system",
                        "severity": "info",
                        "title": "הקלטה התחילה",
                        "description": f"הקלטה של {duration} שניות במצלמה {camera_id}",
                        "cameraId": camera_id,
                        "details": {
                            "recording_id": recording_id,
                            "duration": duration,
                            "preBuffer": pre_buffer
                        }
                    },
                    timeout=10.0  # Increased timeout
                )
        except Exception as e:
            logger.debug(f"Failed to emit recording started event: {e}")

    async def _action_trigger_simulation(self, params: Dict, ctx: RuleContext):
        """Trigger simulation."""
        sim_type = params.get("simulationType")
        delay = params.get("delay", 0)

        if delay > 0:
            await asyncio.sleep(delay / 1000)

        # Map simulation types to Hebrew titles
        sim_titles = {
            "drone_dispatch": "רחפן הוקפץ",
            "phone_call": "חיוג למפקד תורן",
            "pa_announcement": "כריזה",
            "code_broadcast": "שידור קוד",
            "threat_neutralized": "איום נוטרל"
        }

        title = sim_titles.get(sim_type, sim_type)

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKEND_URL}/api/events",
                json={
                    "type": "simulation",
                    "severity": "warning",
                    "title": title,
                    "simulationType": sim_type,
                    "metadata": {"simulation": sim_type}
                },
                timeout=5.0
            )

        logger.info(f"Simulation triggered: {sim_type}")

    async def _action_emergency_mode(self, params: Dict, ctx: RuleContext):
        """Start or end emergency mode."""
        action = params.get("action", "start")

        endpoint = "alerts" if action == "start" else "alerts/end"

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKEND_URL}/api/{endpoint}",
                json={
                    "camera_id": ctx.camera_id,
                    "triggeredBy": "rule_engine",
                    "person_count": ctx.person_count,
                    "armed": ctx.attributes.get("armed", False)
                },
                timeout=5.0
            )

        logger.info(f"Emergency mode: {action}")

    async def _action_add_tag(self, params: Dict, ctx: RuleContext):
        """Add tag to track."""
        tag = params.get("tag")
        if ctx.track_id and tag:
            logger.info(f"Add tag '{tag}' to track {ctx.track_id}")
            # Implementation would depend on your tracking system

    async def _action_set_attribute(self, params: Dict, ctx: RuleContext):
        """Set track attribute."""
        key = params.get("key")
        value = self._interpolate(params.get("value", ""), ctx)
        if ctx.track_id and key:
            logger.info(f"Set attribute '{key}' = '{value}' on track {ctx.track_id}")
            # Implementation would depend on your tracking system

    async def _action_webhook(self, params: Dict, ctx: RuleContext):
        """Call webhook."""
        url = params.get("url")
        method = params.get("method", "POST")
        headers = params.get("headers", {})
        body_template = params.get("body", "{}")

        body = self._interpolate(body_template, ctx)

        try:
            body_json = json.loads(body) if body else {}
        except json.JSONDecodeError:
            body_json = {"raw": body}

        async with httpx.AsyncClient() as client:
            if method == "POST":
                await client.post(url, json=body_json, headers=headers, timeout=10.0)
            elif method == "GET":
                await client.get(url, headers=headers, timeout=10.0)
            elif method == "PUT":
                await client.put(url, json=body_json, headers=headers, timeout=10.0)
            elif method == "PATCH":
                await client.patch(url, json=body_json, headers=headers, timeout=10.0)

        logger.info(f"Webhook called: {method} {url}")

    async def _action_log_event(self, params: Dict, ctx: RuleContext):
        """Log event to the system (saves to database and shows in event log)."""
        message = self._interpolate(params.get("message", ""), ctx)
        level = params.get("level", "info")
        title = self._interpolate(params.get("title", "רישום ביומן"), ctx)

        # Map log level to severity
        severity_map = {
            "debug": "info",
            "info": "info",
            "warning": "warning",
            "error": "critical",
            "critical": "critical"
        }
        severity = severity_map.get(level, "info")

        # Log to Python logger
        log_func = getattr(logger, level, logger.info)
        log_func(f"[RuleEngine] {message}")

        # Also create an event in the database
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BACKEND_URL}/api/events",
                    json={
                        "type": "system",
                        "severity": severity,
                        "title": title,
                        "description": message,
                        "cameraId": ctx.camera_id,
                        "metadata": {
                            "source": "rule_engine",
                            "action": "log_event",
                            "level": level
                        }
                    },
                    timeout=5.0
                )
        except Exception as e:
            logger.debug(f"Failed to save log event: {e}")

    async def _action_play_sound(self, params: Dict, ctx: RuleContext):
        """Play sound (notify frontend via socket)."""
        sound = params.get("sound", "alert")
        volume = params.get("volume", 1)

        # Emit socket event for frontend to play sound
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKEND_URL}/api/system/play-sound",
                json={"sound": sound, "volume": volume},
                timeout=5.0
            )
        logger.info(f"[RuleEngine] Play sound: {sound} (volume: {volume})")

    async def _action_send_notification(self, params: Dict, ctx: RuleContext):
        """Send notification."""
        channel = params.get("channel", "ui")
        title = self._interpolate(params.get("title", ""), ctx)
        body = self._interpolate(params.get("body", ""), ctx)

        if channel == "ui":
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BACKEND_URL}/api/events",
                    json={
                        "type": "system",
                        "severity": "info",
                        "title": title,
                        "description": body,
                        "metadata": {"notification": True}
                    },
                    timeout=5.0
                )

        logger.info(f"Notification: {title}")

    async def _action_select_camera(self, params: Dict, ctx: RuleContext):
        """Select camera in UI."""
        camera_id = params.get("cameraId") or ctx.camera_id

        # Emit socket event via backend
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BACKEND_URL}/api/cameras/{camera_id}/select",
                    timeout=5.0
                )
        except Exception as e:
            logger.debug(f"Failed to select camera: {e}")

        logger.info(f"Select camera: {camera_id}")

    async def _action_auto_focus_camera(self, params: Dict, ctx: RuleContext):
        """
        Auto-focus camera with priority and optional return timeout.
        Feature 6: Automatic Camera Focus on Events.
        """
        camera_id = ctx.camera_id
        if not camera_id:
            logger.debug("auto_focus_camera: No camera_id in context")
            return

        priority = params.get("priority", "high")
        return_timeout = params.get("returnTimeout", 30)
        show_indicator = params.get("showIndicator", True)

        # Build reason from context
        reason = f"אירוע: {ctx.event_type}"
        if ctx.object_type:
            reason = f"זיהוי {ctx.object_type}"
        if ctx.attributes.get("armed"):
            reason = "זיהוי אדם חמוש"
        if ctx.attributes.get("stolen"):
            reason = "זיהוי רכב גנוב"

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BACKEND_URL}/api/cameras/auto-focus",
                    json={
                        "cameraId": camera_id,
                        "priority": priority,
                        "returnTimeout": return_timeout,
                        "showIndicator": show_indicator,
                        "reason": reason,
                        "eventType": ctx.event_type,
                        "severity": self._priority_to_severity(priority)
                    },
                    timeout=5.0
                )
                logger.info(f"Auto-focus camera: {camera_id} (priority={priority}, reason={reason})")
        except Exception as e:
            logger.debug(f"Failed to auto-focus camera: {e}")

    def _priority_to_severity(self, priority: str) -> str:
        """Convert priority to severity level."""
        mapping = {
            "critical": "critical",
            "high": "warning",
            "medium": "info",
            "low": "info"
        }
        return mapping.get(priority, "info")

    async def _action_create_event(self, params: Dict, ctx: RuleContext):
        """Create event in system."""
        event_type = params.get("type", "system")
        severity = params.get("severity", "info")
        title = self._interpolate(params.get("title", ""), ctx)
        description = self._interpolate(params.get("description", ""), ctx)

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKEND_URL}/api/events",
                json={
                    "type": event_type,
                    "severity": severity,
                    "title": title,
                    "description": description,
                    "cameraId": ctx.camera_id,
                    "metadata": ctx.to_dict()
                },
                timeout=5.0
            )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _interpolate(self, template: str, ctx: RuleContext) -> str:
        """Interpolate template string with context values and placeholders."""
        # Use the enhanced interpolation that supports placeholders
        return self._interpolate_with_placeholders(template, ctx)

    async def _record_trigger(self, rule_id: str):
        """Record rule trigger in backend using the /trigger endpoint."""
        if not rule_id:
            return
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BACKEND_URL}/api/event-rules/{rule_id}/trigger",
                    json={"context": {"source": "rule_engine"}},
                    timeout=5.0
                )
        except Exception as e:
            logger.debug(f"Failed to record trigger: {e}")


# Singleton instance
_engine: Optional[RuleEngine] = None


def get_rule_engine() -> RuleEngine:
    """Get or create rule engine instance."""
    global _engine
    if _engine is None:
        _engine = RuleEngine()
    return _engine


async def init_rule_engine() -> RuleEngine:
    """Initialize rule engine and load rules."""
    engine = get_rule_engine()
    await engine.load_rules()
    return engine
