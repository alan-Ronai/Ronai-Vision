"""Pelco-D and Pelco-P PTZ controller over RS-485.

These protocols are commonly used for PTZ control over serial connections (RS-485).
Hikvision cameras support both protocols.
"""

import struct
import logging
from typing import Optional
import serial

logger = logging.getLogger(__name__)


class PelcoController:
    """Pelco-D/P PTZ controller for RS-485 serial communication.
    
    Supports both Pelco-D and Pelco-P protocols used by Hikvision and other PTZ cameras.
    """
    
    # Pelco-D command bytes
    CMD_STOP = 0x00
    CMD_RIGHT = 0x02
    CMD_LEFT = 0x04
    CMD_UP = 0x08
    CMD_DOWN = 0x10
    CMD_ZOOM_IN = 0x20
    CMD_ZOOM_OUT = 0x40
    CMD_FOCUS_NEAR = 0x01  # In second command byte
    CMD_FOCUS_FAR = 0x80   # In second command byte
    
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        address: int = 1,
        protocol: str = "pelco-d"
    ):
        """Initialize Pelco controller.
        
        Args:
            port: Serial port device (e.g., '/dev/ttyUSB0', 'COM3')
            baudrate: Baud rate (usually 2400, 4800, or 9600)
            address: Camera address (1-255)
            protocol: 'pelco-d' or 'pelco-p'
        """
        self.port = port
        self.baudrate = baudrate
        self.address = address
        self.protocol = protocol.lower()
        self.serial = None
        
        logger.info(f"Initializing {protocol.upper()} controller on {port} @ {baudrate} baud, address {address}")
        
    def connect(self):
        """Open serial connection."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            logger.info(f"Serial port {self.port} opened successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to open serial port: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Serial port closed")
    
    def _send_pelco_d(self, cmd1: int, cmd2: int, data1: int, data2: int):
        """Send Pelco-D command.
        
        Pelco-D format: 7 bytes
        Byte 1: Sync byte (0xFF)
        Byte 2: Address (0x01-0xFF)
        Byte 3: Command 1
        Byte 4: Command 2
        Byte 5: Data 1 (pan speed)
        Byte 6: Data 2 (tilt speed)
        Byte 7: Checksum (sum of bytes 2-6, modulo 256)
        """
        if not self.serial or not self.serial.is_open:
            logger.error("Serial port not open")
            return False
        
        # Build packet
        packet = bytearray([
            0xFF,  # Sync
            self.address,
            cmd1,
            cmd2,
            data1,
            data2,
            (self.address + cmd1 + cmd2 + data1 + data2) % 256  # Checksum
        ])
        
        try:
            self.serial.write(packet)
            logger.debug(f"Sent Pelco-D: {packet.hex()}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def _send_pelco_p(self, cmd1: int, cmd2: int, data1: int, data2: int):
        """Send Pelco-P command.
        
        Pelco-P format: 8 bytes
        Byte 1: Sync byte 1 (0xA0)
        Byte 2: Sync byte 2 (0x00)
        Byte 3: Address (0x00-0xFE)
        Byte 4: Command 1
        Byte 5: Command 2
        Byte 6: Data 1 (pan speed)
        Byte 7: Data 2 (tilt speed)
        Byte 8: Checksum (XOR of bytes 3-7)
        """
        if not self.serial or not self.serial.is_open:
            logger.error("Serial port not open")
            return False
        
        # Build packet
        checksum = self.address ^ cmd1 ^ cmd2 ^ data1 ^ data2
        packet = bytearray([
            0xA0,  # Sync 1
            0x00,  # Sync 2
            self.address,
            cmd1,
            cmd2,
            data1,
            data2,
            checksum
        ])
        
        try:
            self.serial.write(packet)
            logger.debug(f"Sent Pelco-P: {packet.hex()}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def _send_command(self, cmd1: int, cmd2: int, data1: int = 0, data2: int = 0):
        """Send command using configured protocol."""
        if self.protocol == "pelco-p":
            return self._send_pelco_p(cmd1, cmd2, data1, data2)
        else:
            return self._send_pelco_d(cmd1, cmd2, data1, data2)
    
    def pan_right(self, speed: int = 0x20):
        """Pan right at specified speed (0x00-0x3F)."""
        return self._send_command(0x00, self.CMD_RIGHT, speed, 0x00)
    
    def pan_left(self, speed: int = 0x20):
        """Pan left at specified speed (0x00-0x3F)."""
        return self._send_command(0x00, self.CMD_LEFT, speed, 0x00)
    
    def tilt_up(self, speed: int = 0x20):
        """Tilt up at specified speed (0x00-0x3F)."""
        return self._send_command(0x00, self.CMD_UP, 0x00, speed)
    
    def tilt_down(self, speed: int = 0x20):
        """Tilt down at specified speed (0x00-0x3F)."""
        return self._send_command(0x00, self.CMD_DOWN, 0x00, speed)
    
    def zoom_in(self, speed: int = 0x00):
        """Zoom in."""
        return self._send_command(0x00, self.CMD_ZOOM_IN, 0x00, 0x00)
    
    def zoom_out(self, speed: int = 0x00):
        """Zoom out."""
        return self._send_command(0x00, self.CMD_ZOOM_OUT, 0x00, 0x00)
    
    def focus_near(self):
        """Focus near."""
        return self._send_command(self.CMD_FOCUS_NEAR, 0x00, 0x00, 0x00)
    
    def focus_far(self):
        """Focus far."""
        return self._send_command(self.CMD_FOCUS_FAR, 0x00, 0x00, 0x00)
    
    def stop(self):
        """Stop all movement."""
        return self._send_command(0x00, 0x00, 0x00, 0x00)
    
    def go_to_preset(self, preset: int):
        """Go to preset position (1-255).
        
        Args:
            preset: Preset number (1-255)
        """
        return self._send_command(0x00, 0x07, 0x00, preset)
    
    def set_preset(self, preset: int):
        """Set current position as preset (1-255).
        
        Args:
            preset: Preset number (1-255)
        """
        return self._send_command(0x00, 0x03, 0x00, preset)
    
    def clear_preset(self, preset: int):
        """Clear preset position (1-255).
        
        Args:
            preset: Preset number (1-255)
        """
        return self._send_command(0x00, 0x05, 0x00, preset)


def test_pelco():
    """Test Pelco controller."""
    import time
    
    print("Pelco PTZ Controller Test")
    print("=" * 50)
    
    # Try to detect serial ports
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    
    if not ports:
        print("❌ No serial ports found")
        print("\nOn Linux, RS-485 adapters usually appear as:")
        print("  /dev/ttyUSB0, /dev/ttyUSB1, /dev/ttyAMA0")
        print("\nOn Windows:")
        print("  COM1, COM2, COM3, etc.")
        print("\nMake sure your RS-485 adapter is connected!")
        return
    
    print("\nAvailable serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i+1}. {port.device} - {port.description}")
    
    choice = input(f"\nSelect port (1-{len(ports)}) or enter custom path: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(ports):
        port = ports[int(choice)-1].device
    else:
        port = choice
    
    baudrate = input("Enter baud rate (default 9600): ").strip() or "9600"
    baudrate = int(baudrate)
    
    address = input("Enter camera address (default 1): ").strip() or "1"
    address = int(address)
    
    protocol = input("Enter protocol (pelco-d or pelco-p, default pelco-d): ").strip() or "pelco-d"
    
    print(f"\nConnecting to {port} @ {baudrate} baud...")
    print(f"Protocol: {protocol.upper()}, Address: {address}")
    
    controller = PelcoController(port, baudrate, address, protocol)
    
    if not controller.connect():
        print("❌ Failed to open serial port")
        return
    
    print("✅ Serial port opened")
    print("\nTesting PTZ commands...")
    print("(Note: You should see the camera move if wired correctly)\n")
    
    try:
        # Test pan right
        print("→ Pan right...")
        controller.pan_right(speed=0x20)
        time.sleep(2)
        controller.stop()
        time.sleep(1)
        
        # Test pan left
        print("← Pan left...")
        controller.pan_left(speed=0x20)
        time.sleep(2)
        controller.stop()
        time.sleep(1)
        
        # Test tilt up
        print("↑ Tilt up...")
        controller.tilt_up(speed=0x20)
        time.sleep(2)
        controller.stop()
        time.sleep(1)
        
        # Test tilt down
        print("↓ Tilt down...")
        controller.tilt_down(speed=0x20)
        time.sleep(2)
        controller.stop()
        time.sleep(1)
        
        print("\n✅ Test complete!")
        print("\nIf camera didn't move, check:")
        print("  1. RS-485 wiring (A to A, B to B)")
        print("  2. Camera address matches")
        print("  3. Baud rate matches camera settings")
        print("  4. Protocol matches (Pelco-D vs Pelco-P)")
        print("  5. Camera has RS-485 enabled")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    test_pelco()
