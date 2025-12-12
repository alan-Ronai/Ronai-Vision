"""Command processor for Hebrew voice commands.

Detects keywords and commands in transcribed Hebrew text and executes
corresponding actions.
"""

import re
import logging
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommandMatch:
    """Represents a matched command."""

    command_id: str
    keyword: str
    text: str
    confidence: float
    timestamp: datetime
    parameters: Dict[str, Any]


class CommandProcessor:
    """Processes transcribed Hebrew text for commands and keywords.

    Detects keywords in Hebrew text and routes to appropriate handlers.
    Supports both exact and fuzzy matching.
    """

    def __init__(self):
        """Initialize command processor."""
        self._handlers: Dict[str, Callable] = {}
        self._keywords: Dict[str, List[str]] = {}
        self._command_history: List[CommandMatch] = []
        self._max_history = 100

        # Register default commands
        self._register_default_commands()

        logger.info("CommandProcessor initialized")

    def _register_default_commands(self):
        """Register default Hebrew commands."""

        # Tracking commands
        self.register_command(
            command_id="track_person",
            keywords=["עקוב", "עקוב אחרי", "תעקוב", "מעקב"],
            handler=self._handle_track_person,
        )

        self.register_command(
            command_id="track_car",
            keywords=["עקוב אחרי רכב", "תעקוב אחרי מכונית", "מעקב רכב"],
            handler=self._handle_track_car,
        )

        # PTZ commands
        self.register_command(
            command_id="zoom_in",
            keywords=["זום", "זום פנימה", "תקרב", "הקרב"],
            handler=self._handle_zoom_in,
        )

        self.register_command(
            command_id="zoom_out",
            keywords=["זום החוצה", "תרחק", "הרחק"],
            handler=self._handle_zoom_out,
        )

        self.register_command(
            command_id="pan_left",
            keywords=["שמאלה", "פנה שמאלה", "תזוז שמאלה"],
            handler=self._handle_pan_left,
        )

        self.register_command(
            command_id="pan_right",
            keywords=["ימינה", "פנה ימינה", "תזוז ימינה"],
            handler=self._handle_pan_right,
        )

        # Status commands
        self.register_command(
            command_id="status_report",
            keywords=["סטטוס", "דוח", "מה הסטטוס", "דווח"],
            handler=self._handle_status_report,
        )

        self.register_command(
            command_id="list_tracks",
            keywords=["רשימת מעקבים", "כמה עוקבים", "מה עוקבים"],
            handler=self._handle_list_tracks,
        )

        # System commands
        self.register_command(
            command_id="stop",
            keywords=["עצור", "תפסיק", "סטופ", "הפסק"],
            handler=self._handle_stop,
        )

        self.register_command(
            command_id="start",
            keywords=["התחל", "תתחיל", "סטארט"],
            handler=self._handle_start,
        )

        logger.info(f"Registered {len(self._keywords)} default commands")

    def register_command(self, command_id: str, keywords: List[str], handler: Callable):
        """Register a new command with keywords and handler.

        Args:
            command_id: Unique command identifier
            keywords: List of Hebrew keywords/phrases that trigger this command
            handler: Function to call when command is detected
        """
        self._keywords[command_id] = keywords
        self._handlers[command_id] = handler
        logger.debug(f"Registered command '{command_id}' with {len(keywords)} keywords")

    def process(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[CommandMatch]:
        """Process transcribed text for commands.

        Args:
            text: Transcribed Hebrew text
            context: Optional context dictionary (session_id, device_id, etc.)

        Returns:
            CommandMatch if command detected, None otherwise
        """
        if not text or not text.strip():
            return None

        text = text.strip()
        text_lower = text.lower()

        logger.debug(f"Processing text for commands: '{text}'")

        # Check each registered command
        for command_id, keywords in self._keywords.items():
            for keyword in keywords:
                # Case-insensitive substring match
                if keyword.lower() in text_lower:
                    logger.info(
                        f"Command detected: '{command_id}' (keyword: '{keyword}')"
                    )

                    # Extract parameters from text
                    parameters = self._extract_parameters(text, keyword, command_id)

                    # Add context
                    if context:
                        parameters.update(context)

                    # Create match
                    match = CommandMatch(
                        command_id=command_id,
                        keyword=keyword,
                        text=text,
                        confidence=1.0,  # TODO: implement fuzzy matching confidence
                        timestamp=datetime.now(),
                        parameters=parameters,
                    )

                    # Add to history
                    self._add_to_history(match)

                    # Execute handler
                    self._execute_handler(match)

                    return match

        logger.debug(f"No command detected in text: '{text}'")
        return None

    def _extract_parameters(
        self, text: str, keyword: str, command_id: str
    ) -> Dict[str, Any]:
        """Extract parameters from command text.

        Args:
            text: Full transcribed text
            keyword: Matched keyword
            command_id: Command identifier

        Returns:
            Dictionary of extracted parameters
        """
        params = {}

        # Extract numbers (for zoom levels, distances, etc.)
        numbers = re.findall(r"\d+", text)
        if numbers:
            params["numbers"] = [int(n) for n in numbers]

        # Command-specific parameter extraction
        if command_id in ("track_person", "track_car"):
            # Try to extract object identifier
            # e.g., "עקוב אחרי אדם 5" -> track person with ID 5
            if numbers:
                params["object_id"] = numbers[0]

        # Extract direction/degree for pan/tilt
        if "zoom" in command_id and numbers:
            params["zoom_level"] = numbers[0]

        return params

    def _execute_handler(self, match: CommandMatch):
        """Execute the command handler.

        Args:
            match: CommandMatch object
        """
        handler = self._handlers.get(match.command_id)
        if handler:
            try:
                handler(match)
            except Exception as e:
                logger.error(f"Handler error for command '{match.command_id}': {e}")
        else:
            logger.warning(f"No handler registered for command '{match.command_id}'")

    def _add_to_history(self, match: CommandMatch):
        """Add command to history.

        Args:
            match: CommandMatch object
        """
        self._command_history.append(match)

        # Trim history if too long
        if len(self._command_history) > self._max_history:
            self._command_history = self._command_history[-self._max_history :]

    def get_history(
        self, limit: int = 10, command_id: Optional[str] = None
    ) -> List[CommandMatch]:
        """Get command history.

        Args:
            limit: Maximum number of commands to return
            command_id: Filter by specific command ID (optional)

        Returns:
            List of CommandMatch objects
        """
        history = self._command_history

        if command_id:
            history = [m for m in history if m.command_id == command_id]

        return list(reversed(history[-limit:]))

    def clear_history(self):
        """Clear command history."""
        self._command_history.clear()
        logger.info("Command history cleared")

    # ========================================================================
    # DEFAULT COMMAND HANDLERS (Stubs)
    # ========================================================================

    def _handle_track_person(self, match: CommandMatch):
        """Handle track person command."""
        print(f"[COMMAND] Track person - not yet implemented")
        logger.info(f"Track person command received: {match.parameters}")
        # TODO: Implement person tracking
        pass

    def _handle_track_car(self, match: CommandMatch):
        """Handle track car command."""
        print(f"[COMMAND] Track car - not yet implemented")
        logger.info(f"Track car command received: {match.parameters}")
        # TODO: Implement car tracking
        pass

    def _handle_zoom_in(self, match: CommandMatch):
        """Handle zoom in command."""
        print(f"[COMMAND] Zoom in - not yet implemented")
        logger.info(f"Zoom in command received: {match.parameters}")
        # TODO: Implement PTZ zoom in
        pass

    def _handle_zoom_out(self, match: CommandMatch):
        """Handle zoom out command."""
        print(f"[COMMAND] Zoom out - not yet implemented")
        logger.info(f"Zoom out command received: {match.parameters}")
        # TODO: Implement PTZ zoom out
        pass

    def _handle_pan_left(self, match: CommandMatch):
        """Handle pan left command."""
        print(f"[COMMAND] Pan left - not yet implemented")
        logger.info(f"Pan left command received: {match.parameters}")
        # TODO: Implement PTZ pan left
        pass

    def _handle_pan_right(self, match: CommandMatch):
        """Handle pan right command."""
        print(f"[COMMAND] Pan right - not yet implemented")
        logger.info(f"Pan right command received: {match.parameters}")
        # TODO: Implement PTZ pan right
        pass

    def _handle_status_report(self, match: CommandMatch):
        """Handle status report command."""
        print(f"[COMMAND] Status report - not yet implemented")
        logger.info(f"Status report command received: {match.parameters}")
        # TODO: Implement status report
        pass

    def _handle_list_tracks(self, match: CommandMatch):
        """Handle list tracks command."""
        print(f"[COMMAND] List tracks - not yet implemented")
        logger.info(f"List tracks command received: {match.parameters}")
        # TODO: Implement list active tracks
        pass

    def _handle_stop(self, match: CommandMatch):
        """Handle stop command."""
        print(f"[COMMAND] Stop - not yet implemented")
        logger.info(f"Stop command received: {match.parameters}")
        # TODO: Implement stop operation
        pass

    def _handle_start(self, match: CommandMatch):
        """Handle start command."""
        print(f"[COMMAND] Start - not yet implemented")
        logger.info(f"Start command received: {match.parameters}")
        # TODO: Implement start operation
        pass
