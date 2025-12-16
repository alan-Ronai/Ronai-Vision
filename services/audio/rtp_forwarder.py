#!/usr/bin/env python3
"""Simple RTP forwarder that receives RTP packets and forwards them to another destination.

Listens for RTP packets on a UDP port and forwards them to a configured destination.
Useful for forwarding RTP streams from EC2 to a local development machine.
"""

import socket
import threading
import time
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class RTPForwarder:
    """Simple RTP packet forwarder.

    Receives RTP packets on a UDP port and forwards them to a destination address.
    """

    def __init__(
        self,
        listen_host: str = "0.0.0.0",
        listen_port: int = 5004,
        forward_host: str = "127.0.0.1",
        forward_port: int = 5004,
    ):
        """Initialize RTP forwarder.

        Args:
            listen_host: Host to bind to (default "0.0.0.0")
            listen_port: UDP port to listen on (default 5004)
            forward_host: Destination host to forward packets to
            forward_port: Destination port to forward packets to
        """
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.forward_host = forward_host
        self.forward_port = forward_port

        # Sockets
        self._listen_socket: Optional[socket.socket] = None
        self._forward_socket: Optional[socket.socket] = None

        # Threads
        self._forward_thread: Optional[threading.Thread] = None
        self._stats_thread: Optional[threading.Thread] = None

        # State
        self._running = False
        self._source_address = None

        # Statistics
        self._stats = {
            "packets_received": 0,
            "packets_forwarded": 0,
            "bytes_received": 0,
            "bytes_forwarded": 0,
            "forward_errors": 0,
            "start_time": None,
            "last_packet_time": None,
        }

        logger.info("RTPForwarder initialized")
        logger.info(f"  Listen: {listen_host}:{listen_port}")
        logger.info(f"  Forward to: {forward_host}:{forward_port}")

    def start(self):
        """Start listening and forwarding RTP packets."""
        if self._running:
            logger.warning("RTP forwarder already running")
            return

        logger.info(f"Starting RTP forwarder on {self.listen_host}:{self.listen_port}")

        try:
            # Create listening socket
            self._listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._listen_socket.bind((self.listen_host, self.listen_port))
            self._listen_socket.settimeout(1.0)

            # Create forwarding socket
            self._forward_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            logger.info(f"RTP forwarder listening on {self.listen_host}:{self.listen_port}")
            logger.info(f"Forwarding to {self.forward_host}:{self.forward_port}")

        except Exception as e:
            logger.error(f"Failed to bind sockets: {e}")
            raise

        # Start forwarding
        self._running = True
        self._stats["start_time"] = datetime.now()

        self._forward_thread = threading.Thread(
            target=self._forward_loop, daemon=True, name="RTP-Forward"
        )
        self._forward_thread.start()

        self._stats_thread = threading.Thread(
            target=self._stats_loop, daemon=True, name="RTP-Stats"
        )
        self._stats_thread.start()

        logger.info("RTP forwarder started")

    def stop(self):
        """Stop forwarding RTP packets."""
        if not self._running:
            return

        logger.info("Stopping RTP forwarder...")
        self._running = False

        # Close sockets
        if self._listen_socket:
            try:
                self._listen_socket.close()
            except Exception:
                pass

        if self._forward_socket:
            try:
                self._forward_socket.close()
            except Exception:
                pass

        # Wait for threads
        if self._forward_thread:
            self._forward_thread.join(timeout=2.0)
        if self._stats_thread:
            self._stats_thread.join(timeout=2.0)

        logger.info("RTP forwarder stopped")
        self._print_stats()

    def get_stats(self) -> dict:
        """Get forwarder statistics."""
        return {
            **self._stats,
            "running": self._running,
            "source_address": self._source_address,
        }

    def _forward_loop(self):
        """Main forwarding loop."""
        logger.info("Forward loop started")

        while self._running:
            try:
                # Check socket is valid
                if self._listen_socket is None:
                    logger.error("Listen socket is None in forward loop")
                    break

                # Receive packet
                data, addr = self._listen_socket.recvfrom(32000)

                # Track source address
                if self._source_address is None:
                    self._source_address = addr
                    logger.info(f"Detected RTP source: {addr}")

                # Update receive stats
                self._stats["packets_received"] += 1
                self._stats["bytes_received"] += len(data)
                self._stats["last_packet_time"] = datetime.now()

                # Forward packet to destination
                try:
                    if self._forward_socket:
                        self._forward_socket.sendto(
                            data, (self.forward_host, self.forward_port)
                        )
                        self._stats["packets_forwarded"] += 1
                        self._stats["bytes_forwarded"] += len(data)
                except Exception as e:
                    self._stats["forward_errors"] += 1
                    if self._stats["forward_errors"] % 100 == 1:  # Log first and every 100th error
                        logger.error(f"Forward error: {e}")

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Forward loop error: {e}", exc_info=True)

        logger.info("Forward loop stopped")

    def _stats_loop(self):
        """Print statistics periodically."""
        logger.info("Stats loop started")

        while self._running:
            try:
                time.sleep(10)  # Print stats every 10 seconds

                if self._stats["packets_received"] > 0:
                    elapsed = (
                        datetime.now() - self._stats["start_time"]
                    ).total_seconds()
                    pps = self._stats["packets_forwarded"] / elapsed if elapsed > 0 else 0
                    kbps = (
                        (self._stats["bytes_forwarded"] * 8 / 1024) / elapsed
                        if elapsed > 0
                        else 0
                    )

                    logger.info(
                        f"Stats: {self._stats['packets_forwarded']} packets forwarded "
                        f"({pps:.1f} pps, {kbps:.1f} kbps), "
                        f"{self._stats['forward_errors']} errors"
                    )

            except Exception as e:
                if self._running:
                    logger.error(f"Stats error: {e}", exc_info=True)

        logger.info("Stats loop stopped")

    def _print_stats(self):
        """Print final session statistics."""
        if self._stats["start_time"]:
            duration = (datetime.now() - self._stats["start_time"]).total_seconds()
            logger.info("Session statistics:")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(f"  Packets received: {self._stats['packets_received']}")
            logger.info(f"  Packets forwarded: {self._stats['packets_forwarded']}")
            logger.info(f"  Bytes received: {self._stats['bytes_received']}")
            logger.info(f"  Bytes forwarded: {self._stats['bytes_forwarded']}")
            logger.info(f"  Forward errors: {self._stats['forward_errors']}")

            if duration > 0:
                pps = self._stats['packets_forwarded'] / duration
                kbps = (self._stats['bytes_forwarded'] * 8 / 1024) / duration
                logger.info(f"  Average rate: {pps:.1f} pps, {kbps:.1f} kbps")


def main():
    """Run RTP forwarder."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RTP Forwarder - Forward RTP packets to another destination"
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="Host to listen on (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=5004,
        help="Port to listen on (default: 5004)",
    )
    parser.add_argument(
        "--forward-host",
        required=True,
        help="Destination host to forward packets to (your local IP)",
    )
    parser.add_argument(
        "--forward-port",
        type=int,
        default=5004,
        help="Destination port to forward packets to (default: 5004)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and start forwarder
    forwarder = RTPForwarder(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        forward_host=args.forward_host,
        forward_port=args.forward_port,
    )

    try:
        forwarder.start()
        logger.info("RTP Forwarder running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        forwarder.stop()


if __name__ == "__main__":
    main()
