/**
 * WebRTCPlayer Component for HAMAL-AI
 * Plays video streams via WebRTC using go2rtc
 *
 * Features:
 * - WebRTC playback (lowest latency)
 * - MSE WebSocket fallback
 * - Auto-reconnection
 * - FPS counter
 */

import { useEffect, useRef, useState, useCallback } from 'react';

// Use proxy to avoid mixed content issues with HTTPS
const GO2RTC_PROXY = '/go2rtc';

export default function WebRTCPlayer({
  streamId,
  className = '',
  onFpsUpdate,
  onConnectionChange,
  autoPlay = true,
  muted = true,
  fallbackToMSE = true
}) {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const fpsIntervalRef = useRef(null);
  const frameCountRef = useRef(0);

  const [connectionState, setConnectionState] = useState('disconnected');
  const [error, setError] = useState(null);
  const [usingFallback, setUsingFallback] = useState(false);

  // FPS calculation using video frame callback (if available)
  const startFpsCounter = useCallback(() => {
    if (fpsIntervalRef.current) return;

    let lastTime = performance.now();
    let frames = 0;

    // Use requestVideoFrameCallback if available (more accurate)
    if (videoRef.current && 'requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      const countFrame = () => {
        frames++;
        if (videoRef.current) {
          videoRef.current.requestVideoFrameCallback(countFrame);
        }
      };
      videoRef.current.requestVideoFrameCallback(countFrame);
    }

    // Report FPS every second
    fpsIntervalRef.current = setInterval(() => {
      const now = performance.now();
      const elapsed = (now - lastTime) / 1000;
      const fps = Math.round(frames / elapsed);

      if (onFpsUpdate) {
        onFpsUpdate(fps);
      }

      frames = 0;
      lastTime = now;
    }, 1000);
  }, [onFpsUpdate]);

  const stopFpsCounter = useCallback(() => {
    if (fpsIntervalRef.current) {
      clearInterval(fpsIntervalRef.current);
      fpsIntervalRef.current = null;
    }
  }, []);

  // WebRTC connection using WHEP (WebRTC HTTP Egress Protocol)
  const connectWebRTC = useCallback(async () => {
    if (!streamId) return;

    try {
      setConnectionState('connecting');
      setError(null);

      // Create peer connection
      const pc = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' }
        ]
      });
      pcRef.current = pc;

      // Add transceiver for receiving video
      pc.addTransceiver('video', { direction: 'recvonly' });
      pc.addTransceiver('audio', { direction: 'recvonly' });

      // Handle incoming tracks
      pc.ontrack = (event) => {
        console.log('[WebRTC] Received track:', event.track.kind);
        if (videoRef.current && event.streams[0]) {
          videoRef.current.srcObject = event.streams[0];
          startFpsCounter();
        }
      };

      // Connection state changes
      pc.onconnectionstatechange = () => {
        console.log('[WebRTC] Connection state:', pc.connectionState);
        setConnectionState(pc.connectionState);

        if (onConnectionChange) {
          onConnectionChange(pc.connectionState);
        }

        if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
          scheduleReconnect();
        }
      };

      pc.oniceconnectionstatechange = () => {
        console.log('[WebRTC] ICE state:', pc.iceConnectionState);
      };

      // Create offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE gathering to complete (or timeout)
      await new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
        } else {
          const checkState = () => {
            if (pc.iceGatheringState === 'complete') {
              pc.removeEventListener('icegatheringstatechange', checkState);
              resolve();
            }
          };
          pc.addEventListener('icegatheringstatechange', checkState);
          // Timeout after 2 seconds
          setTimeout(resolve, 2000);
        }
      });

      // Send offer to go2rtc WHEP endpoint (via proxy)
      const response = await fetch(`${GO2RTC_PROXY}/api/webrtc?src=${streamId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/sdp'
        },
        body: pc.localDescription.sdp
      });

      if (!response.ok) {
        throw new Error(`go2rtc responded with ${response.status}`);
      }

      const answerSdp = await response.text();
      await pc.setRemoteDescription({
        type: 'answer',
        sdp: answerSdp
      });

      console.log('[WebRTC] Connected to stream:', streamId);
      setConnectionState('connected');

    } catch (err) {
      console.error('[WebRTC] Connection error:', err);
      setError(err.message);
      setConnectionState('failed');

      // Try MSE fallback if enabled
      if (fallbackToMSE) {
        console.log('[WebRTC] Falling back to MSE...');
        connectMSE();
      } else {
        scheduleReconnect();
      }
    }
  }, [streamId, startFpsCounter, onConnectionChange, fallbackToMSE]);

  // MSE WebSocket fallback connection
  const connectMSE = useCallback(() => {
    if (!streamId) return;

    try {
      setUsingFallback(true);
      setConnectionState('connecting');
      setError(null);

      // Construct WebSocket URL using current page's protocol (ws:// or wss://)
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}${GO2RTC_PROXY}/api/ws?src=${streamId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.binaryType = 'arraybuffer';

      let mediaSource = null;
      let sourceBuffer = null;
      const queue = [];

      ws.onopen = () => {
        console.log('[MSE] WebSocket connected');
        setConnectionState('connected');
      };

      ws.onmessage = async (event) => {
        if (typeof event.data === 'string') {
          // Handle JSON messages (codec info, etc.)
          const msg = JSON.parse(event.data);
          console.log('[MSE] Message:', msg);

          if (msg.type === 'mse') {
            // Initialize MediaSource
            mediaSource = new MediaSource();
            videoRef.current.src = URL.createObjectURL(mediaSource);

            mediaSource.addEventListener('sourceopen', () => {
              try {
                sourceBuffer = mediaSource.addSourceBuffer(msg.codecs);
                sourceBuffer.mode = 'segments';

                sourceBuffer.addEventListener('updateend', () => {
                  if (queue.length > 0 && !sourceBuffer.updating) {
                    sourceBuffer.appendBuffer(queue.shift());
                  }
                });

                startFpsCounter();
              } catch (e) {
                console.error('[MSE] SourceBuffer error:', e);
              }
            });
          }
        } else {
          // Binary data (video segments)
          if (sourceBuffer && !sourceBuffer.updating) {
            try {
              sourceBuffer.appendBuffer(event.data);
            } catch (e) {
              // Buffer full, add to queue
              queue.push(event.data);
            }
          } else {
            queue.push(event.data);
          }
        }
      };

      ws.onerror = (err) => {
        console.error('[MSE] WebSocket error:', err);
        setError('MSE connection error');
      };

      ws.onclose = () => {
        console.log('[MSE] WebSocket closed');
        setConnectionState('disconnected');
        scheduleReconnect();
      };

    } catch (err) {
      console.error('[MSE] Connection error:', err);
      setError(err.message);
      scheduleReconnect();
    }
  }, [streamId, startFpsCounter]);

  // Schedule reconnection
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) return;

    console.log('[WebRTC] Scheduling reconnect in 3s...');
    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectTimeoutRef.current = null;
      disconnect();

      if (usingFallback) {
        connectMSE();
      } else {
        connectWebRTC();
      }
    }, 3000);
  }, [connectWebRTC, connectMSE, usingFallback]);

  // Disconnect and cleanup
  const disconnect = useCallback(() => {
    stopFpsCounter();

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = '';
    }

    setConnectionState('disconnected');
  }, [stopFpsCounter]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    if (streamId && autoPlay) {
      connectWebRTC();
    }

    return () => {
      disconnect();
    };
  }, [streamId, autoPlay, connectWebRTC, disconnect]);

  // Retry connection
  const retry = useCallback(() => {
    disconnect();
    setUsingFallback(false);
    connectWebRTC();
  }, [disconnect, connectWebRTC]);

  return (
    <div className={`relative ${className}`}>
      <video
        ref={videoRef}
        autoPlay={autoPlay}
        muted={muted}
        playsInline
        className="w-full h-full object-contain bg-black"
      />

      {/* Connection status overlay */}
      {connectionState !== 'connected' && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70">
          <div className="text-center text-white">
            {connectionState === 'connecting' && (
              <>
                <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full mx-auto mb-2" />
                <p>Connecting{usingFallback ? ' (MSE fallback)' : ' via WebRTC'}...</p>
              </>
            )}
            {connectionState === 'failed' && (
              <>
                <p className="text-red-400 mb-2">Connection failed</p>
                {error && <p className="text-sm text-gray-400 mb-2">{error}</p>}
                <button
                  onClick={retry}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                >
                  Retry
                </button>
              </>
            )}
            {connectionState === 'disconnected' && (
              <p className="text-gray-400">Disconnected</p>
            )}
          </div>
        </div>
      )}

      {/* Fallback indicator */}
      {usingFallback && connectionState === 'connected' && (
        <div className="absolute top-2 right-2 px-2 py-1 bg-yellow-600/80 text-white text-xs rounded">
          MSE
        </div>
      )}
    </div>
  );
}
