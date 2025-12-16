import { useApp } from '../context/AppContext';
import { useEffect, useRef, useState, useCallback } from 'react';

// AI Service URL for direct access
const AI_SERVICE_URL = import.meta.env.VITE_AI_SERVICE_URL || 'http://localhost:8000';

export default function MainCamera() {
  const { selectedCamera, cameras, isEmergency } = useApp();
  const imgRef = useRef(null);
  const [error, setError] = useState(null);
  const [frameData, setFrameData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [streamMode, setStreamMode] = useState('annotated'); // 'annotated' or 'raw'

  const camera = cameras.find(c => c.cameraId === selectedCamera);

  // SSE-based streaming - single connection, server controls frame rate
  useEffect(() => {
    if (!selectedCamera) return;

    setError(null);
    setIsConnected(false);
    setFrameCount(0);

    // Use annotated stream (with bboxes) or raw stream
    const endpoint = streamMode === 'annotated'
      ? `${AI_SERVICE_URL}/api/stream/annotated/${encodeURIComponent(selectedCamera)}?fps=5`
      : `${AI_SERVICE_URL}/api/stream/sse/${encodeURIComponent(selectedCamera)}?fps=5`;

    console.log('Connecting to stream:', endpoint);

    const eventSource = new EventSource(endpoint);

    eventSource.onopen = () => {
      console.log('SSE connection opened for', selectedCamera);
      setIsConnected(true);
      setError(null);
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.frame) {
          setFrameData(`data:image/jpeg;base64,${data.frame}`);
          setFrameCount(prev => prev + 1);
        } else if (data.error) {
          console.error('Stream error:', data.error);
        }
      } catch (e) {
        console.error('Error parsing frame:', e);
      }
    };

    eventSource.onerror = (err) => {
      console.error('SSE error:', err);
      setIsConnected(false);
      if (eventSource.readyState === EventSource.CLOSED) {
        setError('החיבור נסגר - מנסה להתחבר מחדש...');
      }
    };

    return () => {
      console.log('Closing SSE connection for', selectedCamera);
      eventSource.close();
    };
  }, [selectedCamera, streamMode]);

  const handleRetry = useCallback(() => {
    setError(null);
    setFrameData(null);
    setFrameCount(0);
  }, []);

  return (
    <div className={`
      h-full bg-gray-800 rounded-lg overflow-hidden flex flex-col
      ${isEmergency ? 'border-2 border-red-500 animate-border-pulse' : 'border border-gray-700'}
    `}>
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
              onClick={() => setStreamMode('annotated')}
              className={`px-2 py-1 text-xs rounded ${streamMode === 'annotated' ? 'bg-blue-600' : ''}`}
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
            {isConnected ? 'משדר' : 'מתחבר...'}
          </div>

          {/* Frame counter */}
          {isConnected && (
            <span className="text-xs text-gray-400">
              {frameCount} פריימים
            </span>
          )}
        </div>
      </div>

      {/* Video/Image display */}
      <div className="flex-1 relative bg-black flex items-center justify-center min-h-0">
        {!camera ? (
          <div className="text-gray-500 text-center">
            <div className="text-6xl mb-4">📷</div>
            <p>בחר מצלמה מהרשימה</p>
          </div>
        ) : error ? (
          <div className="text-red-500 text-center">
            <div className="text-6xl mb-4">⚠️</div>
            <p>{error}</p>
            <button
              onClick={handleRetry}
              className="mt-2 px-3 py-1 bg-gray-700 rounded text-sm hover:bg-gray-600"
            >
              נסה שוב
            </button>
          </div>
        ) : frameData ? (
          <img
            ref={imgRef}
            src={frameData}
            alt={camera.name}
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="text-gray-500 text-center">
            <div className="text-6xl mb-4">📡</div>
            <p>מתחבר לזרם...</p>
            <div className="mt-2 animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
          </div>
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
        {camera?.aiEnabled && (
          <div className="absolute bottom-2 right-2 bg-blue-600/80 text-white px-2 py-1 rounded text-sm flex items-center gap-1">
            <span>🤖</span>
            <span>AI פעיל</span>
          </div>
        )}

        {/* Recording indicator */}
        {camera?.recordEnabled && (
          <div className="absolute bottom-2 left-2 bg-red-600/80 text-white px-2 py-1 rounded text-sm flex items-center gap-1">
            <div className="w-2 h-2 bg-red-300 rounded-full animate-pulse"></div>
            <span>מקליט</span>
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
