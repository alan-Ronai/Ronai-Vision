#!/usr/bin/env python3
"""RTP to TCP relay server - runs on EC2.

Receives RTP packets via UDP and relays them to connected TCP clients.
Clients can connect from anywhere without port forwarding.
"""

import socket
import threading
import time
import logging
import struct
from datetime import datetime
from typing import Set, Optional

logger = logging.getLogger(__name__)


class RTPTCPServer:
    """Receives RTP via UDP and relays to TCP clients."""

    def __init__(
        self,
        rtp_listen_host: str = "0.0.0.0",
        rtp_listen_port: int = 5004,
        tcp_host: str = "0.0.0.0",
        tcp_port: int = 5005,
    ):
        """Initialize RTP to TCP relay server.

        Args:
            rtp_listen_host: Host to bind RTP UDP listener (default "0.0.0.0")
            rtp_listen_port: Port to listen for RTP packets (default 5004)
            tcp_host: Host to bind TCP server (default "0.0.0.0")
            tcp_port: Port for TCP clients to connect (default 5005)
        """
        self.rtp_listen_host = rtp_listen_host
        self.rtp_listen_port = rtp_listen_port
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port

        # Sockets
        self._rtp_socket: Optional[socket.socket] = None
        self._tcp_socket: Optional[socket.socket] = None

        # Connected TCP clients
        self._clients: Set[socket.socket] = set()
        self._clients_lock = threading.Lock()

        # Threads
        self._rtp_thread: Optional[threading.Thread] = None
        self._tcp_thread: Optional[threading.Thread] = None
        self._stats_thread: Optional[threading.Thread] = None

        # State
        self._running = False
        self._rtp_source = None

        # Statistics
        self._stats = {
            "packets_received": 0,
            "packets_relayed": 0,
            "bytes_received": 0,
            "bytes_relayed": 0,
            "clients_connected": 0,
            "start_time": None,
        }

        logger.info("RTPTCPServer initialized")
        logger.info(f"  RTP UDP listen: {rtp_listen_host}:{rtp_listen_port}")
        logger.info(f"  TCP server: {tcp_host}:{tcp_port}")

    def start(self):
        """Start the relay server."""
        if self._running:
            logger.warning("Server already running")
            return

        logger.info("Starting RTP to TCP relay server...")

        try:
            # Create RTP UDP listening socket
            self._rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._rtp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._rtp_socket.bind((self.rtp_listen_host, self.rtp_listen_port))
            self._rtp_socket.settimeout(1.0)
            logger.info(f"RTP listener bound to {self.rtp_listen_host}:{self.rtp_listen_port}")

            # Create TCP server socket
            self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._tcp_socket.bind((self.tcp_host, self.tcp_port))
            self._tcp_socket.listen(5)
            self._tcp_socket.settimeout(1.0)
            logger.info(f"TCP server listening on {self.tcp_host}:{self.tcp_port}")

        except Exception as e:
            logger.error(f"Failed to bind sockets: {e}")
            raise

        # Start threads
        self._running = True
        self._stats["start_time"] = datetime.now()

        self._rtp_thread = threading.Thread(
            target=self._rtp_receive_loop, daemon=True, name="RTP-Receive"
        )
        self._rtp_thread.start()

        self._tcp_thread = threading.Thread(
            target=self._tcp_accept_loop, daemon=True, name="TCP-Accept"
        )
        self._tcp_thread.start()

        self._stats_thread = threading.Thread(
            target=self._stats_loop, daemon=True, name="Stats"
        )
        self._stats_thread.start()

        logger.info("RTP to TCP relay server started")

    def stop(self):
        """Stop the relay server."""
        if not self._running:
            return

        logger.info("Stopping relay server...")
        self._running = False

        # Close all client connections
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()

        # Close sockets
        if self._rtp_socket:
            try:
                self._rtp_socket.close()
            except Exception:
                pass

        if self._tcp_socket:
            try:
                self._tcp_socket.close()
            except Exception:
                pass

        # Wait for threads
        if self._rtp_thread:
            self._rtp_thread.join(timeout=2.0)
        if self._tcp_thread:
            self._tcp_thread.join(timeout=2.0)
        if self._stats_thread:
            self._stats_thread.join(timeout=2.0)

        logger.info("Relay server stopped")
        self._print_stats()

    def _rtp_receive_loop(self):
        """Receive RTP packets and relay to all TCP clients."""
        logger.info("RTP receive loop started")

        while self._running:
            try:
                if self._rtp_socket is None:
                    break

                # Receive RTP packet
                data, addr = self._rtp_socket.recvfrom(32000)

                if self._rtp_source is None:
                    self._rtp_source = addr
                    logger.info(f"Detected RTP source: {addr}")

                self._stats["packets_received"] += 1
                self._stats["bytes_received"] += len(data)

                # Relay to all connected TCP clients
                self._relay_to_clients(data)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"RTP receive error: {e}", exc_info=True)

        logger.info("RTP receive loop stopped")

    def _tcp_accept_loop(self):
        """Accept incoming TCP client connections."""
        logger.info("TCP accept loop started")

        while self._running:
            try:
                if self._tcp_socket is None:
                    break

                # Accept new client
                client_socket, client_addr = self._tcp_socket.accept()
                logger.info(f"New TCP client connected: {client_addr}")

                with self._clients_lock:
                    self._clients.add(client_socket)
                    self._stats["clients_connected"] = len(self._clients)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"TCP accept error: {e}", exc_info=True)

        logger.info("TCP accept loop stopped")

    def _relay_to_clients(self, data: bytes):
        """Relay RTP packet to all connected TCP clients.

        Packet format: [4 bytes length][RTP packet data]
        """
        # Prepare packet with length prefix
        packet_length = len(data)
        packet = struct.pack("!I", packet_length) + data

        with self._clients_lock:
            dead_clients = set()

            for client in self._clients:
                try:
                    client.sendall(packet)
                    self._stats["packets_relayed"] += 1
                    self._stats["bytes_relayed"] += len(data)
                except Exception as e:
                    logger.warning(f"Failed to send to client: {e}")
                    dead_clients.add(client)

            # Remove dead clients
            for client in dead_clients:
                try:
                    client.close()
                except Exception:
                    pass
                self._clients.discard(client)
                logger.info("Removed disconnected client")

            self._stats["clients_connected"] = len(self._clients)

    def _stats_loop(self):
        """Print statistics periodically."""
        logger.info("Stats loop started")

        while self._running:
            try:
                time.sleep(10)

                if self._stats["packets_received"] > 0:
                    elapsed = (
                        datetime.now() - self._stats["start_time"]
                    ).total_seconds()
                    pps = self._stats["packets_received"] / elapsed if elapsed > 0 else 0
                    kbps = (
                        (self._stats["bytes_received"] * 8 / 1024) / elapsed
                        if elapsed > 0
                        else 0
                    )

                    logger.info(
                        f"Stats: {self._stats['packets_received']} RTP packets received, "
                        f"{self._stats['packets_relayed']} relayed to {self._stats['clients_connected']} clients "
                        f"({pps:.1f} pps, {kbps:.1f} kbps)"
                    )

            except Exception as e:
                if self._running:
                    logger.error(f"Stats error: {e}", exc_info=True)

        logger.info("Stats loop stopped")

    def _print_stats(self):
        """Print final statistics."""
        if self._stats["start_time"]:
            duration = (datetime.now() - self._stats["start_time"]).total_seconds()
            logger.info("Session statistics:")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(f"  RTP packets received: {self._stats['packets_received']}")
            logger.info(f"  Packets relayed: {self._stats['packets_relayed']}")
            logger.info(f"  Bytes received: {self._stats['bytes_received']}")
            logger.info(f"  Bytes relayed: {self._stats['bytes_relayed']}")


def main():
    """Run RTP to TCP relay server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RTP to TCP Relay Server - Receive RTP via UDP and relay to TCP clients"
    )
    parser.add_argument(
        "--rtp-port",
        type=int,
        default=5004,
        help="UDP port to receive RTP packets (default: 5004)",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=5005,
        help="TCP port for clients to connect (default: 5005)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = RTPTCPServer(
        rtp_listen_port=args.rtp_port,
        tcp_port=args.tcp_port,
    )

    try:
        server.start()
        logger.info("Server running. Press Ctrl+C to stop...")
        logger.info(f"Clients can connect to: <EC2_IP>:{args.tcp_port}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
