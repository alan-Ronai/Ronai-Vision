/**
 * NewCameraDialog - Dialog to connect a new camera
 *
 * Shows during the NEW_CAMERA stage to add a new RTSP stream.
 */

import { useState } from 'react';
import { useScenario } from '../../context/ScenarioContext';

export default function NewCameraDialog() {
  const { newCameraDialog, connectNewCamera, config } = useScenario();
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!newCameraDialog) {
    return null;
  }

  const handleConnect = async () => {
    if (!url.trim()) {
      setError('נא להזין כתובת');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const success = await connectNewCamera(url);
      if (!success) {
        setError('התחברות נכשלה');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 flex items-center justify-center">
      <div className="bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
        {/* Title */}
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <span>&#128247;</span>
          {config?.ui?.newCameraTitle || 'התחבר למצלמה חדשה'}
        </h2>

        {/* Description */}
        <p className="text-gray-400 mb-4 text-right">
          הזן כתובת RTSP לחיבור מצלמה חדשה למערכת
        </p>

        {/* URL input */}
        <div className="mb-4">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder={config?.ui?.newCameraPlaceholder || 'rtsp://...'}
            className="w-full bg-gray-900 border border-gray-600 rounded px-4 py-3 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            dir="ltr"
          />
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-4 p-3 bg-red-900/50 border border-red-500 rounded text-red-300 text-sm text-right">
            {error}
          </div>
        )}

        {/* Preset URLs for demo */}
        <div className="mb-4">
          <div className="text-sm text-gray-500 mb-2">כתובות לדוגמה:</div>
          <div className="space-y-1">
            {[
              'rtsp://admin:admin@192.168.1.100:554/stream1',
              'rtsp://viewer:viewer@10.0.0.50:554/live',
            ].map((preset, i) => (
              <button
                key={i}
                onClick={() => setUrl(preset)}
                className="block w-full text-left text-xs text-blue-400 hover:text-blue-300 truncate"
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={handleConnect}
            disabled={loading}
            className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed text-white font-semibold py-3 rounded flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <span className="animate-spin">&#9696;</span>
                מתחבר...
              </>
            ) : (
              <>
                {config?.ui?.connectButton || 'התחבר'}
              </>
            )}
          </button>
        </div>

        {/* Note */}
        <div className="mt-4 text-xs text-gray-500 text-center">
          המצלמה תתווסף לרשימת המצלמות ותוצג כמצלמה ראשית
        </div>
      </div>
    </div>
  );
}
