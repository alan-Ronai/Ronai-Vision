import { useState, useRef, useCallback, useEffect } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';
const GO2RTC_URL = import.meta.env.VITE_GO2RTC_URL || 'http://localhost:1984';

/**
 * WebcamShare Component
 * Allows users to share their browser webcam with the HAMAL-AI system
 * Uses WebRTC WHIP (WebRTC-HTTP Ingestion Protocol) to stream to go2rtc
 */
export default function WebcamShare({ onStreamStarted, onStreamStopped }) {
  const [isSharing, setIsSharing] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState(null);
  const [streamInfo, setStreamInfo] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const peerConnectionRef = useRef(null);

  // Fetch available video devices
  useEffect(() => {
    async function getDevices() {
      try {
        // Request permission first to get device labels
        const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
        tempStream.getTracks().forEach(track => track.stop());

        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = allDevices.filter(d => d.kind === 'videoinput');
        setDevices(videoDevices);
        if (videoDevices.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.log('Could not enumerate devices:', err);
      }
    }
    getDevices();
  }, []);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopSharing();
    };
  }, []);

  const startSharing = useCallback(async () => {
    setError(null);
    setIsConnecting(true);

    try {
      // Step 1: Get user's webcam
      const constraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: 1280, height: 720 }
          : { width: 1280, height: 720 },
        audio: false // Audio not needed for security monitoring
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      // Show local preview
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Step 2: Register with backend and get WHIP URL
      const clientId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const regResponse = await fetch(`${API_URL}/api/cameras/browser-webcam/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: `המצלמה שלי`,
          clientId
        })
      });

      if (!regResponse.ok) {
        throw new Error('Failed to register webcam with server');
      }

      const regData = await regResponse.json();
      setStreamInfo(regData);

      // Step 3: Create WebRTC peer connection
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });
      peerConnectionRef.current = pc;

      // Add tracks to peer connection
      stream.getTracks().forEach(track => {
        pc.addTrack(track, stream);
      });

      // Create offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE gathering to complete
      await new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
        } else {
          pc.addEventListener('icegatheringstatechange', () => {
            if (pc.iceGatheringState === 'complete') {
              resolve();
            }
          });
          // Timeout after 5 seconds
          setTimeout(resolve, 5000);
        }
      });

      // Step 4: Send offer to go2rtc WHIP endpoint
      const whipUrl = regData.whip;
      const whipResponse = await fetch(whipUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/sdp' },
        body: pc.localDescription.sdp
      });

      if (!whipResponse.ok) {
        throw new Error(`WHIP connection failed: ${whipResponse.status}`);
      }

      // Get answer from go2rtc
      const answerSdp = await whipResponse.text();
      await pc.setRemoteDescription({
        type: 'answer',
        sdp: answerSdp
      });

      // Step 5: Notify backend that connection is established
      await fetch(`${API_URL}/api/cameras/browser-webcam/${regData.camera.cameraId}/connected`, {
        method: 'POST'
      });

      setIsSharing(true);
      setIsConnecting(false);

      if (onStreamStarted) {
        onStreamStarted(regData.camera);
      }

      // Handle connection state changes
      pc.addEventListener('connectionstatechange', () => {
        if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
          stopSharing();
        }
      });

    } catch (err) {
      console.error('Failed to start webcam sharing:', err);
      setError(err.message);
      setIsConnecting(false);
      stopSharing();
    }
  }, [selectedDeviceId, onStreamStarted]);

  const stopSharing = useCallback(async () => {
    // Stop local stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // Close peer connection
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }

    // Clear video preview
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Notify backend
    if (streamInfo?.camera?.cameraId) {
      try {
        await fetch(`${API_URL}/api/cameras/browser-webcam/${streamInfo.camera.cameraId}/disconnected`, {
          method: 'POST'
        });
      } catch (e) {
        // Ignore
      }
    }

    setIsSharing(false);
    setIsConnecting(false);
    setStreamInfo(null);

    if (onStreamStopped) {
      onStreamStopped();
    }
  }, [streamInfo, onStreamStopped]);

  return (
    <div className="bg-gray-700 rounded-lg p-4">
      <h3 className="font-bold mb-3 flex items-center gap-2">
        <span>📷</span>
        <span>שתף את המצלמה שלך</span>
      </h3>

      {/* Device selector */}
      {!isSharing && devices.length > 1 && (
        <div className="mb-3">
          <label className="block text-sm text-gray-400 mb-1">בחר מצלמה</label>
          <select
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            className="w-full bg-gray-600 rounded px-3 py-2 text-white text-sm"
          >
            {devices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `מצלמה ${devices.indexOf(device) + 1}`}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Preview */}
      <div className="relative mb-3 bg-gray-800 rounded-lg overflow-hidden aspect-video">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        {!isSharing && !isConnecting && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <span className="text-4xl block mb-2">📹</span>
              <span className="text-sm">לחץ להתחלת שיתוף</span>
            </div>
          </div>
        )}
        {isConnecting && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <div className="text-center">
              <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
              <span className="text-sm">מתחבר...</span>
            </div>
          </div>
        )}
        {isSharing && (
          <div className="absolute top-2 right-2 flex items-center gap-1 bg-red-600 px-2 py-1 rounded text-xs">
            <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
            <span>משדר</span>
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="mb-3 text-red-400 text-sm bg-red-900/30 rounded px-3 py-2">
          שגיאה: {error}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2">
        {!isSharing ? (
          <button
            onClick={startSharing}
            disabled={isConnecting}
            className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded-lg flex items-center justify-center gap-2"
          >
            {isConnecting ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>מתחבר...</span>
              </>
            ) : (
              <>
                <span>▶️</span>
                <span>התחל שיתוף</span>
              </>
            )}
          </button>
        ) : (
          <button
            onClick={stopSharing}
            className="flex-1 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg flex items-center justify-center gap-2"
          >
            <span>⏹️</span>
            <span>עצור שיתוף</span>
          </button>
        )}
      </div>

      {/* Stream info */}
      {isSharing && streamInfo && (
        <div className="mt-3 text-xs text-gray-400 bg-gray-800 rounded p-2">
          <div>מזהה: {streamInfo.camera?.cameraId}</div>
          <div>המצלמה שלך משודרת כעת למערכת HAMAL-AI</div>
        </div>
      )}
    </div>
  );
}
