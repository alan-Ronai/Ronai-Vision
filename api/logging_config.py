"""Logging configuration for FastAPI server.

Provides custom logging setup to reduce noise from repetitive stream publish logs.
"""

import logging
import time
from typing import Dict


class StreamPublishFilter(logging.Filter):
    """Filter to compress repetitive stream publish logs.

    Instead of logging every single stream publish request (which can be 30+ per second),
    this filter:
    - Logs the first occurrence immediately
    - Groups subsequent occurrences and logs a summary every N seconds
    - Logs changes in status code or errors immediately
    """

    def __init__(self, name="", interval: int = 10):
        super().__init__(name)
        self.interval = interval  # Log summary every N seconds
        self.counters: Dict[str, Dict] = {}  # Track counts per endpoint
        self.last_log_time: Dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records for stream publish endpoints.

        Returns:
            True to allow the log, False to suppress it
        """
        # Only filter INFO level access logs
        if record.levelno != logging.INFO:
            return True

        # Check if this is a stream publish log
        msg = record.getMessage()
        if "/api/stream/publish/" not in msg and "POST" not in msg:
            return True  # Not a stream publish log, allow it

        # Extract camera ID and status from log message
        # Expected format: "POST /api/stream/publish/cam1 HTTP/1.1" 200 OK
        try:
            parts = msg.split()
            if len(parts) < 4:
                return True  # Unexpected format, allow it

            # Find the endpoint and status
            endpoint = None
            status = None
            for i, part in enumerate(parts):
                if "/api/stream/publish/" in part:
                    endpoint = part
                    # Look for status code (200, 404, 500, etc.)
                    if i + 2 < len(parts) and parts[i + 2].isdigit():
                        status = parts[i + 2]
                    break

            if not endpoint:
                return True  # Not a stream publish log, allow it

            key = f"{endpoint}:{status}"
            current_time = time.time()

            # Initialize counter for this endpoint if needed
            if key not in self.counters:
                self.counters[key] = {
                    "count": 0,
                    "first_time": current_time,
                    "logged_first": False,
                }
                self.last_log_time[key] = current_time

            counter = self.counters[key]
            counter["count"] += 1

            # Always log first occurrence
            if not counter["logged_first"]:
                counter["logged_first"] = True
                return True

            # Check if it's time to log summary
            time_since_last_log = current_time - self.last_log_time[key]
            if time_since_last_log >= self.interval:
                # Create summary message
                count = counter["count"]
                duration = current_time - counter["first_time"]
                rate = count / duration if duration > 0 else 0

                # Print summary directly to stdout to avoid formatter complications
                # Format it to look like a uvicorn INFO log
                summary_msg = f"INFO:     [Stream Summary] {endpoint} - {count} requests in {duration:.1f}s ({rate:.1f} req/s, status {status})"
                print(summary_msg, flush=True)

                # Reset counters
                counter["count"] = 0
                counter["first_time"] = current_time
                self.last_log_time[key] = current_time

                # Suppress this log record (we already printed the summary)
                return False

            # Suppress this log (will be included in next summary)
            return False

        except Exception:
            # If parsing fails, allow the log through
            return True


class StreamAccessFormatter(logging.Formatter):
    """Custom formatter that handles both normal access logs and stream summaries.

    Wraps uvicorn's AccessFormatter but detects summary messages and formats them simply.
    """

    def __init__(self):
        super().__init__()
        # Import uvicorn's AccessFormatter
        try:
            from uvicorn.logging import AccessFormatter

            self.access_formatter = AccessFormatter(
                fmt='%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
            )
        except ImportError:
            self.access_formatter = None

    def format(self, record: logging.LogRecord) -> str:
        """Format log record, using simple format for summaries."""
        # Check if this is a summary message (marked by filter)
        if hasattr(record, "_skip_access_format") and record._skip_access_format:
            # Simple formatting for summary messages
            return f"INFO:     {record.getMessage()}"

        # Use AccessFormatter for normal access logs
        if self.access_formatter:
            return self.access_formatter.format(record)

        # Fallback to simple format
        return f"{record.levelname}:     {record.getMessage()}"


def configure_uvicorn_logging(log_level: str = "info") -> dict:
    """Configure uvicorn logging with stream publish filter.

    Args:
        log_level: Logging level (info, debug, warning, error)

    Returns:
        Logging configuration dict for uvicorn
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "stream_publish_filter": {
                "()": "api.logging_config.StreamPublishFilter",
                "interval": 10,  # Log summary every 10 seconds
            }
        },
        "formatters": {
            "default": {
                "format": "%(levelprefix)s %(message)s",
                "use_colors": True,
            },
            "access": {
                "format": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["stream_publish_filter"],  # Apply filter to access logs
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level.upper(),
            },
            "uvicorn.error": {
                "level": "INFO",
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


def get_simple_filter():
    """Get a simple stream publish filter for manual application.

    Use this if you're starting uvicorn without custom log config:

        import logging
        from api.logging_config import get_simple_filter

        logging.getLogger("uvicorn.access").addFilter(get_simple_filter())
    """
    return StreamPublishFilter(interval=10)
