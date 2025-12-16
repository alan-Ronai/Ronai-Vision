import { useApp } from '../context/AppContext';

export default function CameraGrid() {
  const { cameras, selectedCamera, selectCamera, isEmergency, API_URL } = useApp();

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
        <span className="text-sm text-gray-400">{cameras.length} מצלמות</span>
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
              apiUrl={API_URL}
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
    </div>
  );
}

function CameraThumbnail({ camera, isSelected, onSelect, apiUrl }) {
  const statusColors = {
    online: 'bg-green-500',
    offline: 'bg-gray-500',
    error: 'bg-red-500',
    connecting: 'bg-yellow-500'
  };

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
      {camera.thumbnail ? (
        <img
          src={`${apiUrl}${camera.thumbnail}`}
          alt={camera.name}
          className="w-full h-full object-cover"
        />
      ) : camera.status === 'online' ? (
        <img
          src={`${apiUrl}/api/stream/mjpeg/${camera.cameraId}`}
          alt={camera.name}
          className="w-full h-full object-cover"
          loading="lazy"
        />
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
