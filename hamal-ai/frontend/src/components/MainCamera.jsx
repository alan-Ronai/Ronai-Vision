import { useApp } from '../context/AppContext';
import { useEffect, useRef, useState, useCallback } from 'react';

// Service URLs
const AI_SERVICE_URL = import.meta.env.VITE_AI_SERVICE_URL || 'http://localhost:8000';
const AI_SERVICE_WS = AI_SERVICE_URL.replace('http://', 'ws://').replace('https://', 'wss://');
const GO2RTC_URL = import.meta.env.VITE_GO2RTC_URL || 'http://localhost:1984';

export default function MainCamera() {
  const { selectedCamera, cameras, isEmergency } = useApp();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const websocketRef = useRef(null);
  const animationFrameRef = useRef(null);

  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [streamMode, setStreamMode] = useState('ai'); // 'ai' (with overlay) or 'raw'
  const [detections, setDetections] = useState([]);
  const [detectionCount, setDetectionCount] = useState(0);
  const [backendFrameSize, setBackendFrameSize] = useState({ width: 0, height: 0 });

  const camera = cameras.find(c => c.cameraId === selectedCamera);

  // Color scheme matching backend BBoxDrawer
  const getBoxColor = (detection) => {
    const isArmed = detection.metadata?.analysis?.armed || detection.metadata?.armed || false;
    const isPredicted = detection.is_predicted || detection.consecutive_misses > 0;
    const label = detection.class || detection.label || 'object';

    if (isArmed) return '#FF0000';      // Red - armed person
    if (isPredicted) return '#808080';   // Gray - predicted
    if (label === 'person') return '#00FF00';  // Green - person
    // Vehicles - cyan
    if (['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle'].includes(label)) {
      return '#00BFFF';
    }
    return '#00BFFF'; // Default cyan
  };

  // Setup WebRTC connection to go2rtc
  const setupWebRTC = useCallback(async (cameraId) => {
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
    }

    try {
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });
      peerConnectionRef.current = pc;

      pc.ontrack = (event) => {
        if (videoRef.current && event.streams[0]) {
          videoRef.current.srcObject = event.streams[0];
          setIsConnected(true);
          setError(null);
        }
      };

      pc.oniceconnectionstatechange = () => {
        if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
          setIsConnected(false);
          setError('WebRTC connection lost');
        }
      };

      // Create offer
      pc.addTransceiver('video', { direction: 'recvonly' });
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Send offer to go2rtc
      const response = await fetch(`${GO2RTC_URL}/api/webrtc?src=${encodeURIComponent(cameraId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/sdp' },
        body: offer.sdp
      });

      if (!response.ok) {
        throw new Error(`go2rtc returned ${response.status}`);
      }

      const answerSdp = await response.text();
      await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });

    } catch (err) {
      console.error('WebRTC setup failed:', err);
      setError(`WebRTC failed: ${err.message}`);
      setIsConnected(false);
    }
  }, []);

  // Setup WebSocket connection for detections
  const setupWebSocket = useCallback((cameraId) => {
    if (websocketRef.current) {
      websocketRef.current.close();
    }

    const wsUrl = `${AI_SERVICE_WS}/api/detections/${encodeURIComponent(cameraId)}/ws`;
    console.log('Connecting to detection WebSocket:', wsUrl);

    const ws = new WebSocket(wsUrl);
    websocketRef.current = ws;

    ws.onopen = () => {
      console.log('Detection WebSocket connected');
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Ignore ping messages
        if (data.type === 'ping') return;

        if (data.detections !== undefined) {
          setDetections(data.detections);
          setDetectionCount(data.count || data.detections.length);
          // Store frame dimensions from backend for coordinate scaling
          if (data.frame_width && data.frame_height) {
            setBackendFrameSize({ width: data.frame_width, height: data.frame_height });
          }
        }
      } catch (e) {
        console.error('Error parsing detection message:', e);
      }
    };

    ws.onclose = () => {
      console.log('Detection WebSocket closed');
      setWsConnected(false);
      // Don't auto-reconnect here - let the effect handle it
    };

    ws.onerror = (err) => {
      console.error('Detection WebSocket error:', err);
      setWsConnected(false);
    };

    return ws;
  }, []);

  // Draw overlay on canvas - called every animation frame
  const drawOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      animationFrameRef.current = requestAnimationFrame(drawOverlay);
      return;
    }

    const ctx = canvas.getContext('2d');

    // Use canvas's parent element dimensions (the video container div)
    const parent = canvas.parentElement;
    const containerWidth = parent?.clientWidth || 800;
    const containerHeight = parent?.clientHeight || 600;

    // Native video resolution from WebRTC stream
    const videoNativeWidth = video.videoWidth;
    const videoNativeHeight = video.videoHeight;

    // Backend detection frame dimensions (may differ from WebRTC video!)
    // Use backend dimensions if available, otherwise fall back to video dimensions
    const detectionWidth = backendFrameSize.width || videoNativeWidth;
    const detectionHeight = backendFrameSize.height || videoNativeHeight;

    // Set canvas resolution to match container (CSS stretches to fill)
    if (canvas.width !== containerWidth || canvas.height !== containerHeight) {
      canvas.width = containerWidth;
      canvas.height = containerHeight;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate how object-contain positions the video within container
    // object-contain maintains aspect ratio and fits within container
    const videoAspect = videoNativeWidth / videoNativeHeight;
    const containerAspect = containerWidth / containerHeight;

    let renderWidth, renderHeight, offsetX, offsetY;

    if (videoAspect > containerAspect) {
      // Video is wider - fits width, letterbox top/bottom
      renderWidth = containerWidth;
      renderHeight = containerWidth / videoAspect;
      offsetX = 0;
      offsetY = (containerHeight - renderHeight) / 2;
    } else {
      // Video is taller - fits height, letterbox left/right
      renderHeight = containerHeight;
      renderWidth = containerHeight * videoAspect;
      offsetX = (containerWidth - renderWidth) / 2;
      offsetY = 0;
    }

    // Scale factors: convert detection coords (backend frame) to canvas coords
    // CRITICAL: Use backend detection frame size, not WebRTC video size
    const scaleX = renderWidth / detectionWidth;
    const scaleY = renderHeight / detectionHeight;

    // Draw each detection
    detections.forEach(det => {
      const bbox = det.bbox || det.box || [];
      if (bbox.length < 4) return;

      const [x1, y1, x2, y2] = bbox;
      const trackId = det.track_id || det.id || '?';
      const label = det.class || det.label || 'object';
      const confidence = det.confidence || det.conf || 0;
      const isArmed = det.metadata?.analysis?.armed || det.metadata?.armed || false;

      // Transform coordinates from detection frame to canvas
      const sx1 = x1 * scaleX + offsetX;
      const sy1 = y1 * scaleY + offsetY;
      const sx2 = x2 * scaleX + offsetX;
      const sy2 = y2 * scaleY + offsetY;
      const boxWidth = sx2 - sx1;
      const boxHeight = sy2 - sy1;

      // Get color
      const color = getBoxColor(det);

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = isArmed ? 4 : 2;
      ctx.strokeRect(sx1, sy1, boxWidth, boxHeight);

      // Build label - clean format: "ID:X class"
      // Handle track IDs like "v_0_1" or "p_1_5" - extract just the last number
      let idDisplay = String(trackId);
      const parts = idDisplay.split('_');
      if (parts.length >= 3) {
        // New format: v_<session>_<id> -> show just the id
        idDisplay = parts[parts.length - 1];
      } else if (parts.length === 2) {
        // Old format: v_<id> -> show just the id
        idDisplay = parts[1];
      }

      let labelText = `ID:${idDisplay} ${label}`;
      if (isArmed) {
        labelText += ' ARMED';
      } else if (confidence > 0 && confidence < 0.85) {
        labelText += ` ${Math.round(confidence * 100)}%`;
      }

      // Draw label with bigger font
      ctx.font = 'bold 16px monospace';
      ctx.textBaseline = 'top';  // Use top baseline for easier positioning
      ctx.textAlign = 'left';
      const textMetrics = ctx.measureText(labelText);
      const padding = 6;
      const labelWidth = textMetrics.width + padding * 2;
      const labelHeight = 22 + padding;  // Font size + padding

      // Position label ABOVE the bbox
      const labelX = sx1;
      const labelY = Math.max(0, sy1 - labelHeight - 2);  // 2px gap above bbox

      // Draw label background FIRST
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
      ctx.fillRect(labelX, labelY, labelWidth, labelHeight);

      // Draw label text INSIDE the background
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(labelText, labelX + padding, labelY + padding);
    });

    // Continue animation loop
    if (streamMode === 'ai') {
      animationFrameRef.current = requestAnimationFrame(drawOverlay);
    }
  }, [detections, streamMode, backendFrameSize]);

  // Main effect: setup WebRTC and WebSocket
  useEffect(() => {
    if (!selectedCamera) return;

    setError(null);
    setIsConnected(false);
    setWsConnected(false);
    setDetections([]);
    setDetectionCount(0);

    // Setup WebRTC for video
    setupWebRTC(selectedCamera);

    // Setup WebSocket for detections (if AI mode)
    if (streamMode === 'ai') {
      setupWebSocket(selectedCamera);
    }

    return () => {
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
        peerConnectionRef.current = null;
      }
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [selectedCamera, streamMode, setupWebRTC, setupWebSocket]);

  // Start overlay animation when video plays
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handlePlay = () => {
      if (streamMode === 'ai' && !animationFrameRef.current) {
        animationFrameRef.current = requestAnimationFrame(drawOverlay);
      }
    };

    const handlePause = () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };

    video.addEventListener('playing', handlePlay);
    video.addEventListener('pause', handlePause);

    // Start if already playing
    if (!video.paused && streamMode === 'ai') {
      handlePlay();
    }

    return () => {
      video.removeEventListener('playing', handlePlay);
      video.removeEventListener('pause', handlePause);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [streamMode, drawOverlay]);

  // Fullscreen handling
  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => {
        setIsFullscreen(true);
      }).catch(err => console.error('Fullscreen error:', err));
    } else {
      document.exitFullscreen().then(() => {
        setIsFullscreen(false);
      });
    }
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  return (
    <div
      ref={containerRef}
      className={`
        h-full bg-gray-800 rounded-lg overflow-hidden flex flex-col
        ${isEmergency ? 'border-2 border-red-500 animate-border-pulse' : 'border border-gray-700'}
        ${isFullscreen ? 'fixed inset-0 z-50 rounded-none' : ''}
      `}
    >
      {/* Header */}
      <div className="bg-gray-700 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-lg">📹</span>
          <span className="font-bold">{camera?.name || 'בחר מצלמה'}</span>
          {camera?.location && (
            <span className="text-sm text-gray-400">| {camera.location}</span>
          )}
        </div>

        <div className="flex items-center gap-4">
          {/* Stream mode toggle */}
          <div className="flex gap-1 bg-gray-600 rounded p-1">
            <button
              onClick={() => setStreamMode('ai')}
              className={`px-2 py-1 text-xs rounded ${streamMode === 'ai' ? 'bg-blue-600' : ''}`}
              title="עם זיהוי AI"
            >
              🤖 AI
            </button>
            <button
              onClick={() => setStreamMode('raw')}
              className={`px-2 py-1 text-xs rounded ${streamMode === 'raw' ? 'bg-blue-600' : ''}`}
              title="וידאו רגיל"
            >
              📹 Raw
            </button>
          </div>

          {/* Connection status */}
          <div className={`
            flex items-center gap-1 px-2 py-1 rounded text-sm
            ${isConnected ? 'bg-green-600' : 'bg-yellow-600'}
          `}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-300 animate-pulse' : 'bg-yellow-300'}`}></div>
            {isConnected ? 'WebRTC' : 'מתחבר...'}
          </div>

          {/* WebSocket status (AI mode only) */}
          {streamMode === 'ai' && (
            <div className={`
              flex items-center gap-1 px-2 py-1 rounded text-sm
              ${wsConnected ? 'bg-purple-600' : 'bg-gray-600'}
            `}>
              <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-purple-300' : 'bg-gray-400'}`}></div>
              WS
            </div>
          )}

          {/* Detection count */}
          {streamMode === 'ai' && detectionCount > 0 && (
            <span className="text-sm px-2 py-1 bg-blue-600 rounded">
              {detectionCount} זיהויים
            </span>
          )}

          {/* Fullscreen button */}
          <button
            onClick={toggleFullscreen}
            className="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm transition-colors"
            title={isFullscreen ? 'יציאה ממסך מלא' : 'מסך מלא'}
          >
            {isFullscreen ? '⛶' : '⛶'}
          </button>
        </div>
      </div>

      {/* Video display with overlay */}
      <div className="flex-1 relative bg-black min-h-0">
        {!camera ? (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <div className="text-6xl mb-4">📷</div>
              <p>בחר מצלמה מהרשימה</p>
            </div>
          </div>
        ) : error ? (
          <div className="absolute inset-0 flex items-center justify-center text-red-500">
            <div className="text-center">
              <div className="text-6xl mb-4">⚠️</div>
              <p>{error}</p>
              <button
                onClick={() => setupWebRTC(selectedCamera)}
                className="mt-2 px-3 py-1 bg-gray-700 rounded text-sm hover:bg-gray-600"
              >
                נסה שוב
              </button>
            </div>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-contain"
            />
            {/* Canvas overlay - positioned exactly over the video */}
            {streamMode === 'ai' && (
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />
            )}
          </>
        )}

        {/* Emergency overlay */}
        {isEmergency && (
          <div className="absolute inset-0 border-4 border-red-500 pointer-events-none">
            <div className="absolute top-2 right-2 bg-red-600 text-white px-3 py-1 rounded font-bold animate-pulse">
              🚨 אירוע חירום
            </div>
          </div>
        )}

        {/* AI detection indicator */}
        {camera?.aiEnabled && streamMode === 'ai' && (
          <div className="absolute bottom-2 right-2 bg-blue-600/80 text-white px-2 py-1 rounded text-sm flex items-center gap-1">
            <span>🤖</span>
            <span>AI פעיל</span>
          </div>
        )}
      </div>

      {/* PTZ Controls (if supported) */}
      {camera?.ptzEnabled && (
        <div className="bg-gray-700 px-4 py-2 flex items-center justify-center gap-2 flex-shrink-0">
          <button className="bg-gray-600 hover:bg-gray-500 p-2 rounded">⬆️</button>
          <div className="flex flex-col gap-1">
            <button className="bg-gray-600 hover:bg-gray-500 p-2 rounded">⬅️</button>
            <button className="bg-gray-600 hover:bg-gray-500 p-2 rounded">➡️</button>
          </div>
          <button className="bg-gray-600 hover:bg-gray-500 p-2 rounded">⬇️</button>
          <div className="mr-4 flex gap-2">
            <button className="bg-gray-600 hover:bg-gray-500 px-3 py-2 rounded text-sm">➕</button>
            <button className="bg-gray-600 hover:bg-gray-500 px-3 py-2 rounded text-sm">➖</button>
          </div>
        </div>
      )}
    </div>
  );
}
