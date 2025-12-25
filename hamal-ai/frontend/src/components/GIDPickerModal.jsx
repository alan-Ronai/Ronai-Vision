import { useState, useEffect, useCallback } from 'react';
import { useApp } from '../context/AppContext';

// Use relative URLs to leverage Vite proxy (avoids mixed content issues with HTTPS)
const BACKEND_URL = '';

export default function GIDPickerModal({ isOpen, onClose, onSelect, type }) {
  const [objects, setObjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedGid, setSelectedGid] = useState(null);

  // Fetch tracked objects
  const fetchObjects = useCallback(async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (type) params.append('type', type);
      params.append('isActive', 'true');
      params.append('limit', '50');

      const response = await fetch(`${BACKEND_URL}/api/tracked?${params}`);
      const data = await response.json();
      setObjects(data.objects || []);
    } catch (error) {
      console.error('Failed to fetch tracked objects:', error);
    } finally {
      setLoading(false);
    }
  }, [type]);

  useEffect(() => {
    if (isOpen) {
      fetchObjects();
      setSelectedGid(null);
    }
  }, [isOpen, fetchObjects]);

  if (!isOpen) return null;

  const handleSelect = () => {
    if (selectedGid) {
      const selected = objects.find(o => o.gid === selectedGid);
      onSelect(selected);
      onClose();
    }
  };

  const handleSelectRandom = () => {
    onSelect(null); // null means use random
    onClose();
  };

  const typeLabel = type === 'vehicle' ? 'רכב' : type === 'person' ? 'אדם' : 'אובייקט';
  const typeIcon = type === 'vehicle' ? '🚗' : type === 'person' ? '👤' : '📦';

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg shadow-2xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gray-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span className="text-2xl">{typeIcon}</span>
            בחר {typeLabel} לדמו
          </h2>
          <button
            onClick={onClose}
            className="text-gray-300 hover:text-white text-2xl"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto max-h-[50vh]">
          {loading ? (
            <div className="text-center py-8 text-gray-400">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-blue-500 mx-auto mb-2"></div>
              טוען נתונים...
            </div>
          ) : objects.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <span className="text-4xl block mb-2">📭</span>
              אין {type === 'vehicle' ? 'רכבים' : 'אנשים'} פעילים במערכת
              <p className="text-sm mt-2">נא לוודא שיש מצלמות פעילות עם זיהויים</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {objects.map((obj) => (
                <div
                  key={obj.gid}
                  onClick={() => setSelectedGid(obj.gid)}
                  className={`
                    p-3 rounded-lg cursor-pointer transition-all
                    ${selectedGid === obj.gid
                      ? 'bg-blue-600 ring-2 ring-blue-400'
                      : 'bg-gray-700 hover:bg-gray-600'
                    }
                  `}
                >
                  {/* Cutout image */}
                  {obj.cutout ? (
                    <img
                      src={`data:image/jpeg;base64,${obj.cutout}`}
                      alt={`GID ${obj.gid}`}
                      className="w-full h-24 object-contain bg-gray-900 rounded mb-2"
                    />
                  ) : (
                    <div className="w-full h-24 bg-gray-900 rounded mb-2 flex items-center justify-center">
                      <span className="text-4xl opacity-30">{typeIcon}</span>
                    </div>
                  )}

                  {/* Info */}
                  <div className="text-sm">
                    <div className="font-bold text-white">GID: {obj.gid}</div>
                    {type === 'vehicle' && obj.analysis && (
                      <>
                        {obj.analysis.licensePlate && (
                          <div className="text-yellow-400 text-xs">
                            {obj.analysis.licensePlate}
                          </div>
                        )}
                        {obj.analysis.color && (
                          <div className="text-gray-400 text-xs">
                            {obj.analysis.color}
                          </div>
                        )}
                      </>
                    )}
                    {type === 'person' && obj.analysis && (
                      <>
                        {obj.analysis.isArmed && (
                          <div className="text-red-400 text-xs font-bold">
                            🔫 חמוש
                          </div>
                        )}
                        {obj.analysis.shirtColor && (
                          <div className="text-gray-400 text-xs">
                            חולצה: {obj.analysis.shirtColor}
                          </div>
                        )}
                      </>
                    )}
                    <div className="text-gray-500 text-xs">
                      {obj.cameraId}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-900 px-6 py-3 flex justify-between items-center">
          <button
            onClick={handleSelectRandom}
            className="bg-gray-600 hover:bg-gray-500 px-4 py-2 rounded font-bold transition-colors"
          >
            🎲 בחר אקראי
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="bg-gray-600 hover:bg-gray-500 px-4 py-2 rounded font-bold transition-colors"
            >
              ביטול
            </button>
            <button
              onClick={handleSelect}
              disabled={!selectedGid}
              className={`px-4 py-2 rounded font-bold transition-colors ${
                selectedGid
                  ? 'bg-blue-600 hover:bg-blue-500'
                  : 'bg-gray-600 opacity-50 cursor-not-allowed'
              }`}
            >
              ✓ בחר
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
