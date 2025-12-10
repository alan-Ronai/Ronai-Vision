#!/usr/bin/env python3
"""Test client for RTP/RTSP audio server.

Sends test audio to the RTP/RTSP server for connectivity testing.
"""

import socket
import struct
import time
import argparse
import audioop
import numpy as np


def send_rtsp_request(sock, request):
    """Send RTSP request and receive response."""
    sock.sendall(request.encode('utf-8'))
    response = sock.recv(4096).decode('utf-8', errors='ignore')
    print(f"Response:\n{response}")
    return response


def generate_tone(frequency=440, sample_rate=8000, duration=0.02):
    """Generate sine wave tone.

    Args:
        frequency: Tone frequency in Hz (default 440 Hz = A4)
        sample_rate: Sample rate in Hz
        duration: Duration in seconds (default 20ms for G.711)

    Returns:
        PCM samples as int16 numpy array
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, False)
    tone = np.sin(frequency * 2 * np.pi * t)

    # Convert to int16
    pcm = (tone * 32767).astype(np.int16)
    return pcm


def encode_g711_ulaw(pcm_samples):
    """Encode PCM to G.711 μ-law.

    Args:
        pcm_samples: int16 numpy array

    Returns:
        G.711 encoded bytes
    """
    pcm_bytes = pcm_samples.tobytes()
    return audioop.lin2ulaw(pcm_bytes, 2)


def create_rtp_packet(payload, sequence_number, timestamp, payload_type=0, ssrc=0x12345678):
    """Create RTP packet.

    Args:
        payload: RTP payload bytes
        sequence_number: Sequence number (0-65535)
        timestamp: RTP timestamp
        payload_type: Payload type (0 for G.711 μ-law)
        ssrc: Synchronization source identifier

    Returns:
        Complete RTP packet bytes
    """
    # RTP header (12 bytes)
    # V(2), P(1), X(1), CC(4), M(1), PT(7)
    version = 2
    padding = 0
    extension = 0
    csrc_count = 0
    marker = 0

    byte0 = (version << 6) | (padding << 5) | (extension << 4) | csrc_count
    byte1 = (marker << 7) | payload_type

    header = struct.pack('!BBHII',
                         byte0,
                         byte1,
                         sequence_number & 0xFFFF,
                         timestamp & 0xFFFFFFFF,
                         ssrc)

    return header + payload


def test_rtsp_rtcp_audio(server_host="127.0.0.1", rtsp_port=8554, duration=10):
    """Test RTSP/RTP audio transmission.

    Args:
        server_host: RTSP server host
        rtsp_port: RTSP server port
        duration: Test duration in seconds
    """
    print(f"[Test] Connecting to RTSP server at {server_host}:{rtsp_port}")

    # Connect to RTSP server
    rtsp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rtsp_sock.connect((server_host, rtsp_port))

    cseq = 1
    session_id = None

    try:
        # 1. OPTIONS
        print("\n[Test] Sending OPTIONS request...")
        request = (
            f"OPTIONS rtsp://{server_host}:{rtsp_port}/audio RTSP/1.0\r\n"
            f"CSeq: {cseq}\r\n"
            f"\r\n"
        )
        send_rtsp_request(rtsp_sock, request)
        cseq += 1

        # 2. SETUP
        print("\n[Test] Sending SETUP request...")
        client_rtp_port = 6000
        client_rtcp_port = 6001

        request = (
            f"SETUP rtsp://{server_host}:{rtsp_port}/audio/track1 RTSP/1.0\r\n"
            f"CSeq: {cseq}\r\n"
            f"Transport: RTP/AVP/UDP;unicast;client_port={client_rtp_port}-{client_rtcp_port}\r\n"
            f"\r\n"
        )
        response = send_rtsp_request(rtsp_sock, request)
        cseq += 1

        # Parse session ID
        for line in response.split('\r\n'):
            if line.startswith('Session:'):
                session_id = line.split(':')[1].strip()
                print(f"[Test] Got session ID: {session_id}")
                break

        # Parse server RTP port
        server_rtp_port = 5004  # Default
        for line in response.split('\r\n'):
            if 'server_port=' in line:
                ports = line.split('server_port=')[1].split(';')[0]
                server_rtp_port = int(ports.split('-')[0])
                print(f"[Test] Server RTP port: {server_rtp_port}")
                break

        # 3. PLAY
        print("\n[Test] Sending PLAY request...")
        request = (
            f"PLAY rtsp://{server_host}:{rtsp_port}/audio RTSP/1.0\r\n"
            f"CSeq: {cseq}\r\n"
            f"Session: {session_id}\r\n"
            f"\r\n"
        )
        send_rtsp_request(rtsp_sock, request)
        cseq += 1

        # 4. Send RTP packets
        print(f"\n[Test] Sending RTP audio for {duration} seconds...")
        print(f"[Test] Target: {server_host}:{server_rtp_port}")

        # Create UDP socket for RTP
        rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Audio parameters (G.711 μ-law)
        sample_rate = 8000
        packet_duration = 0.02  # 20ms packets
        samples_per_packet = int(sample_rate * packet_duration)

        # Timing
        start_time = time.time()
        sequence_number = 0
        timestamp = 0
        packet_count = 0

        # Generate test tone (440 Hz = A4)
        print("[Test] Generating 440 Hz test tone...")

        while time.time() - start_time < duration:
            # Generate tone for this packet
            pcm_samples = generate_tone(frequency=440, sample_rate=sample_rate, duration=packet_duration)

            # Encode to G.711 μ-law
            g711_payload = encode_g711_ulaw(pcm_samples)

            # Create RTP packet
            rtp_packet = create_rtp_packet(
                payload=g711_payload,
                sequence_number=sequence_number,
                timestamp=timestamp,
                payload_type=0  # G.711 μ-law
            )

            # Send packet
            rtp_sock.sendto(rtp_packet, (server_host, server_rtp_port))

            # Update counters
            sequence_number = (sequence_number + 1) & 0xFFFF
            timestamp += samples_per_packet
            packet_count += 1

            # Progress indicator
            if packet_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[Test] Sent {packet_count} packets ({elapsed:.1f}s elapsed)")

            # Wait for next packet (maintain timing)
            time.sleep(packet_duration)

        print(f"\n[Test] Sent total {packet_count} RTP packets")

        # 5. TEARDOWN
        print("\n[Test] Sending TEARDOWN request...")
        request = (
            f"TEARDOWN rtsp://{server_host}:{rtsp_port}/audio RTSP/1.0\r\n"
            f"CSeq: {cseq}\r\n"
            f"Session: {session_id}\r\n"
            f"\r\n"
        )
        send_rtsp_request(rtsp_sock, request)

        # Close sockets
        rtp_sock.close()
        rtsp_sock.close()

        print("\n[Test] Test completed successfully!")
        print(f"[Test] Check audio_storage/recordings/ for the recorded file")

    except Exception as e:
        print(f"\n[Test] Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            rtsp_sock.close()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RTP/RTSP audio client")
    parser.add_argument("--host", default="127.0.0.1", help="RTSP server host")
    parser.add_argument("--port", type=int, default=8554, help="RTSP server port")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")

    args = parser.parse_args()

    print("=" * 60)
    print("RTP/RTSP Audio Test Client")
    print("=" * 60)

    test_rtsp_rtcp_audio(
        server_host=args.host,
        rtsp_port=args.port,
        duration=args.duration
    )
