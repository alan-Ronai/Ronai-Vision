import { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import CameraManager from './CameraManager';

const AI_SERVICE_URL = import.meta.env.VITE_AI_SERVICE_URL || 'http://localhost:8000';

export default function CameraGrid() {
  const { cameras, selectedCamera, selectCamera, isEmergency, API_URL } = useApp();
  const [showCameraManager, setShowCameraManager] = useState(false);

  return (
    <div className={`
      h-full bg-gray-800 rounded-lg overflow-hidden flex flex-col
      ${isEmergency ? 'border-2 border-red-500' : 'border border-gray-700'}
    `}>
      {/* Header */}
      <div className="bg-gray-700 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <span>📷</span>
          <span className="font-bold">מצלמות</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowCameraManager(true)}
            className="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-xs flex items-center gap-1 transition-colors"
            title="ניהול מצלמות"
          >
            <span>⚙️</span>
            <span>ניהול</span>
          </button>
          <span className="text-sm text-gray-400">{cameras.length} מצלמות</span>
        </div>
      </div>

      {/* Camera grid */}
      <div className="flex-1 p-2 overflow-y-auto min-h-0">
        <div className="grid grid-cols-3 gap-2">
          {cameras.map((camera) => (
            <CameraThumbnail
              key={camera.cameraId}
              camera={camera}
              isSelected={selectedCamera === camera.cameraId}
              onSelect={() => selectCamera(camera.cameraId)}
              aiServiceUrl={AI_SERVICE_URL}
            />
          ))}

          {cameras.length === 0 && (
            <div className="col-span-3 text-center text-gray-500 py-8">
              <div className="text-4xl mb-2">📷</div>
              <p>אין מצלמות מוגדרות</p>
            </div>
          )}
        </div>
      </div>

      {/* Camera Manager Modal */}
      <CameraManager
        isOpen={showCameraManager}
        onClose={() => setShowCameraManager(false)}
      />
    </div>
  );
}

function CameraThumbnail({ camera, isSelected, onSelect, aiServiceUrl }) {
  const [imgError, setImgError] = useState(false);
  const [imgLoaded, setImgLoaded] = useState(false);

  const statusColors = {
    online: 'bg-green-500',
    offline: 'bg-gray-500',
    error: 'bg-red-500',
    connecting: 'bg-yellow-500 animate-pulse'
  };

  // Use AI service URL for a single frame snapshot instead of MJPEG stream
  // This is more reliable for thumbnails and doesn't keep connections open
  const snapshotUrl = `${aiServiceUrl}/api/stream/snapshot/${camera.cameraId}?t=${Date.now()}`;

  // Reset error state when camera changes or comes online
  useEffect(() => {
    if (camera.status === 'online') {
      setImgError(false);
      setImgLoaded(false);
    }
  }, [camera.cameraId, camera.status]);

  return (
    <button
      onClick={onSelect}
      className={`
        relative aspect-video bg-gray-900 rounded overflow-hidden
        transition-all duration-200
        ${isSelected
          ? 'ring-2 ring-blue-500 scale-105'
          : 'hover:ring-1 hover:ring-gray-500'
        }
      `}
    >
      {/* Thumbnail image or placeholder */}
      {camera.status === 'online' && !imgError ? (
        <>
          <img
            src={snapshotUrl}
            alt={camera.name}
            className={`w-full h-full object-cover transition-opacity ${imgLoaded ? 'opacity-100' : 'opacity-0'}`}
            onLoad={() => setImgLoaded(true)}
            onError={() => setImgError(true)}
          />
          {!imgLoaded && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-600">
              <span className="text-2xl animate-pulse">📷</span>
            </div>
          )}
        </>
      ) : (
        <div className="w-full h-full flex items-center justify-center text-gray-600">
          <span className="text-2xl">📷</span>
        </div>
      )}

      {/* Status indicator */}
      <div className={`
        absolute top-1 left-1 w-2 h-2 rounded-full
        ${statusColors[camera.status] || 'bg-gray-500'}
      `}></div>

      {/* Camera name */}
      <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-1">
        <p className="text-xs text-white truncate">{camera.name}</p>
      </div>

      {/* AI indicator */}
      {camera.aiEnabled && (
        <div className="absolute top-1 right-1 text-xs">🤖</div>
      )}

      {/* Selected overlay */}
      {isSelected && (
        <div className="absolute inset-0 border-2 border-blue-500 pointer-events-none"></div>
      )}
    </button>
  );
}
