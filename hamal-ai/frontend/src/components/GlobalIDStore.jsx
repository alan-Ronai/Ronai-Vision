import { useState, useEffect, useCallback } from 'react';
import { useApp } from '../context/AppContext';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3000';

// Helper to check if a value is a valid analysis result (not a placeholder)
const isValidValue = (value) => {
  if (!value) return false;
  const placeholders = ['לא זוהה', 'לא נראה', 'לא נראה בבירור', 'לא ידוע', 'null', 'undefined'];
  return !placeholders.includes(String(value).toLowerCase().trim());
};

export default function GlobalIDStore({ isOpen, onClose }) {
  const { socket } = useApp();
  const [objects, setObjects] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [storageType, setStorageType] = useState(null); // 'mongodb', 'in-memory', or 'in-memory-fallback'
  const [filter, setFilter] = useState({
    type: 'all',
    isActive: true,
    isArmed: false,
  });
  const [selectedObject, setSelectedObject] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch tracked objects
  const fetchObjects = useCallback(async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (filter.type !== 'all') params.append('type', filter.type);
      params.append('isActive', filter.isActive);
      if (filter.isArmed) params.append('isArmed', 'true');
      params.append('limit', '100');

      const response = await fetch(`${BACKEND_URL}/api/tracked?${params}`);
      const data = await response.json();
      setObjects(data.objects || []);
      if (data.storage) {
        setStorageType(data.storage);
      }
    } catch (error) {
      console.error('Failed to fetch tracked objects:', error);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/tracked/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  }, []);

  // Initial fetch and refresh interval
  useEffect(() => {
    if (isOpen) {
      fetchObjects();
      fetchStats();

      // Refresh every 10 seconds
      const interval = setInterval(() => {
        fetchObjects();
        fetchStats();
      }, 10000);

      return () => clearInterval(interval);
    }
  }, [isOpen, fetchObjects, fetchStats]);

  // Socket.IO real-time updates
  useEffect(() => {
    if (socket && isOpen) {
      const handleUpdate = (object) => {
        setObjects(prev => {
          const idx = prev.findIndex(o => o.gid === object.gid || o._id === object._id);
          if (idx >= 0) {
            const updated = [...prev];
            updated[idx] = object;
            return updated;
          }
          return [object, ...prev];
        });
        // Update selected if it's the same object
        if (selectedObject && (selectedObject.gid === object.gid || selectedObject._id === object._id)) {
          setSelectedObject(object);
        }
      };

      const handleDeactivated = ({ gid }) => {
        setObjects(prev => prev.filter(o => o.gid !== gid));
        if (selectedObject?.gid === gid) {
          setSelectedObject(null);
        }
      };

      socket.on('tracked:update', handleUpdate);
      socket.on('tracked:appearance', ({ gid }) => {
        // Refetch the updated object
        fetchObjects();
      });
      socket.on('tracked:deactivated', handleDeactivated);

      return () => {
        socket.off('tracked:update', handleUpdate);
        socket.off('tracked:appearance');
        socket.off('tracked:deactivated', handleDeactivated);
      };
    }
  }, [socket, isOpen, selectedObject, fetchObjects]);

  // Filter objects by search query
  const filteredObjects = objects.filter(obj => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      (obj.gid && obj.gid.toString().includes(query)) ||
      (obj.trackId && obj.trackId.toLowerCase().includes(query)) ||
      (obj.type && obj.type.includes(query)) ||
      (obj.analysis?.licensePlate && obj.analysis.licensePlate.toLowerCase().includes(query)) ||
      (obj.analysis?.color && obj.analysis.color.toLowerCase().includes(query)) ||
      (obj.analysis?.description && obj.analysis.description.toLowerCase().includes(query)) ||
      (obj.analysis?.clothingColor && obj.analysis.clothingColor.toLowerCase().includes(query))
    );
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg w-[90vw] max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gray-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span className="text-2xl">🆔</span>
            <span>מאגר זיהויים גלובלי (Global ID Store)</span>
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl leading-none"
          >
            &times;
          </button>
        </div>

        {/* Stats Bar */}
        {stats && (
          <div className="bg-gray-700/50 px-6 py-3 flex gap-6 text-sm border-b border-gray-600 flex-wrap items-center">
            <div className="flex items-center gap-2">
              <span className="text-green-400">👤</span>
              <span>אנשים: {stats.persons?.active || 0} פעילים / {stats.persons?.total || 0} סה"כ</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-blue-400">🚗</span>
              <span>רכבים: {stats.vehicles?.active || 0} פעילים / {stats.vehicles?.total || 0} סה"כ</span>
            </div>
            {(stats.persons?.armed || 0) > 0 && (
              <div className="flex items-center gap-2 text-red-400">
                <span>⚠️</span>
                <span>חמושים: {stats.persons?.armed}</span>
              </div>
            )}
            <div className="flex items-center gap-2 text-yellow-400">
              <span>📍</span>
              <span>פעילים אחרונה: {stats.recentlyActive || 0}</span>
            </div>
            {(stats.threats || 0) > 0 && (
              <div className="flex items-center gap-2 text-orange-400">
                <span>🚨</span>
                <span>איומים: {stats.threats}</span>
              </div>
            )}
            {/* Storage indicator */}
            <div className="flex-1"></div>
            <div className={`flex items-center gap-1 text-xs px-2 py-1 rounded ${
              storageType === 'mongodb' ? 'bg-green-900/50 text-green-400' :
              storageType === 'in-memory' ? 'bg-yellow-900/50 text-yellow-400' :
              'bg-orange-900/50 text-orange-400'
            }`}>
              <span>{storageType === 'mongodb' ? '💾' : '🧠'}</span>
              <span>{
                storageType === 'mongodb' ? 'MongoDB' :
                storageType === 'in-memory' ? 'זיכרון (ללא DB)' :
                storageType === 'in-memory-fallback' ? 'זיכרון (fallback)' :
                'טוען...'
              }</span>
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="px-6 py-3 bg-gray-700/30 flex gap-4 items-center border-b border-gray-600 flex-wrap">
          {/* Search */}
          <div className="flex-1 min-w-[200px]">
            <input
              type="text"
              placeholder="חיפוש לפי GID, לוחית רישוי, צבע..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Type filter */}
          <select
            value={filter.type}
            onChange={(e) => setFilter({ ...filter, type: e.target.value })}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
          >
            <option value="all">כל הסוגים</option>
            <option value="person">אנשים</option>
            <option value="vehicle">רכבים</option>
          </select>

          {/* Active filter */}
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={filter.isActive}
              onChange={(e) => setFilter({ ...filter, isActive: e.target.checked })}
              className="rounded"
            />
            פעילים בלבד
          </label>

          {/* Armed filter */}
          <label className="flex items-center gap-2 text-sm text-red-400 cursor-pointer">
            <input
              type="checkbox"
              checked={filter.isArmed}
              onChange={(e) => setFilter({ ...filter, isArmed: e.target.checked })}
              className="rounded"
            />
            חמושים בלבד
          </label>

          {/* Refresh */}
          <button
            onClick={() => { fetchObjects(); fetchStats(); }}
            className="bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm transition-colors"
          >
            🔄 רענן
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex">
          {/* Object List */}
          <div className="w-1/2 overflow-y-auto border-l border-gray-600">
            {loading ? (
              <div className="p-8 text-center text-gray-400">טוען...</div>
            ) : filteredObjects.length === 0 ? (
              <div className="p-8 text-center text-gray-400">לא נמצאו אובייקטים</div>
            ) : (
              <div className="divide-y divide-gray-700">
                {filteredObjects.map((obj) => (
                  <div
                    key={obj._id || obj.gid}
                    onClick={() => setSelectedObject(obj)}
                    className={`p-4 cursor-pointer hover:bg-gray-700/50 transition-colors ${
                      selectedObject?._id === obj._id || selectedObject?.gid === obj.gid
                        ? 'bg-blue-900/30'
                        : ''
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {/* Cutout thumbnail or fallback icon */}
                        {obj.analysis?.cutout_image ? (
                          <img
                            src={`data:image/jpeg;base64,${obj.analysis.cutout_image}`}
                            alt=""
                            className="w-10 h-10 object-cover rounded border border-gray-600"
                          />
                        ) : (
                          <span className="text-2xl">
                            {obj.type === 'person'
                              ? (obj.isArmed || obj.analysis?.armed ? '🔫' : '👤')
                              : '🚗'}
                          </span>
                        )}
                        <div>
                          <div className="font-bold">
                            GID #{obj.gid}
                            {(obj.isArmed || obj.analysis?.armed) && (
                              <span className="mr-2 text-red-400 text-sm">⚠️ חמוש</span>
                            )}
                          </div>
                          <div className="text-sm text-gray-400">
                            {obj.type === 'person' ? 'אדם' : 'רכב'}
                            {obj.analysis?.color && ` • ${obj.analysis.color}`}
                            {obj.analysis?.licensePlate && ` • ${obj.analysis.licensePlate}`}
                            {obj.analysis?.clothingColor && ` • ${obj.analysis.clothingColor}`}
                          </div>
                        </div>
                      </div>
                      <div className="text-left text-sm text-gray-400">
                        <div>📍 {obj.appearances?.length || 0} הופעות</div>
                        <div>🕐 {obj.lastSeen ? new Date(obj.lastSeen).toLocaleTimeString('he-IL') : '-'}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Detail Panel */}
          <div className="w-1/2 overflow-y-auto p-6 bg-gray-900/50">
            {selectedObject ? (
              <ObjectDetail object={selectedObject} onRefreshAnalysis={() => fetchObjects()} />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                בחר אובייקט לצפייה בפרטים
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const AI_SERVICE_URL = import.meta.env.VITE_AI_SERVICE_URL || 'http://localhost:8000';

// Detail component for selected object
function ObjectDetail({ object, onRefreshAnalysis }) {
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState(null);

  const formatDate = (date) => {
    if (!date) return '-';
    return new Date(date).toLocaleString('he-IL');
  };

  const isArmed = object.isArmed || object.analysis?.armed;

  const handleRefreshAnalysis = async () => {
    setRefreshing(true);
    setRefreshError(null);

    try {
      const trackId = object.trackId || `${object.type === 'vehicle' ? 'v' : 't'}_${object.gid}`;
      const cameraId = object.cameraId || '';

      const response = await fetch(
        `${AI_SERVICE_URL}/tracker/refresh-analysis/${trackId}?camera_id=${cameraId}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Refresh failed');
      }

      const result = await response.json();
      // Call parent callback to refresh the object list
      if (onRefreshAnalysis) {
        onRefreshAnalysis(result);
      }
    } catch (error) {
      console.error('Refresh analysis error:', error);
      setRefreshError(error.message);
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header with Cutout Image */}
      <div className="flex items-start gap-4">
        {/* Cutout image or fallback icon */}
        {object.analysis?.cutout_image ? (
          <img
            src={`data:image/jpeg;base64,${object.analysis.cutout_image}`}
            alt="Object cutout"
            className="w-24 h-24 object-cover rounded-lg border-2 border-gray-600"
          />
        ) : (
          <div className="w-24 h-24 flex items-center justify-center bg-gray-700 rounded-lg border-2 border-gray-600">
            <span className="text-5xl">
              {object.type === 'person' ? (isArmed ? '🔫' : '👤') : '🚗'}
            </span>
          </div>
        )}
        <div className="flex-1">
          <h3 className="text-2xl font-bold">GID #{object.gid}</h3>
          <p className="text-gray-400">
            {object.type === 'person' ? 'אדם' : 'רכב'}
            {object.subType && ` (${object.subType})`}
            {object.class && ` - ${object.class}`}
          </p>
          {isArmed && (
            <span className="inline-block mt-2 bg-red-600 px-3 py-1 rounded-full text-sm">
              ⚠️ מזוהה כחמוש
            </span>
          )}
        </div>
      </div>

      {/* Status */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-800 rounded p-3">
          <div className="text-sm text-gray-400">סטטוס</div>
          <div className={object.isActive && object.status === 'active' ? 'text-green-400' : 'text-gray-500'}>
            {object.isActive && object.status === 'active' ? '● פעיל' : '○ לא פעיל'}
          </div>
        </div>
        <div className="bg-gray-800 rounded p-3">
          <div className="text-sm text-gray-400">רמת איום</div>
          <div className={
            object.threatLevel === 'critical' ? 'text-red-400' :
            object.threatLevel === 'high' ? 'text-orange-400' :
            object.threatLevel === 'medium' ? 'text-yellow-400' :
            'text-green-400'
          }>
            {object.threatLevel || 'ללא'}
          </div>
        </div>
      </div>

      {/* Camera */}
      {object.cameraId && (
        <div className="bg-gray-800 rounded p-3">
          <div className="text-sm text-gray-400">מצלמה אחרונה</div>
          <div>{object.cameraName || object.cameraId}</div>
        </div>
      )}

      {/* Timestamps */}
      <div className="bg-gray-800 rounded p-4">
        <h4 className="font-bold mb-2">זמנים</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-gray-400">נראה לראשונה:</span>
            <div>{formatDate(object.firstSeen)}</div>
          </div>
          <div>
            <span className="text-gray-400">נראה לאחרונה:</span>
            <div>{formatDate(object.lastSeen)}</div>
          </div>
        </div>
      </div>

      {/* Analysis */}
      <div className="bg-gray-800 rounded p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-bold">ניתוח Gemini</h4>
          <button
            onClick={handleRefreshAnalysis}
            disabled={refreshing || !object.isActive}
            className={`px-3 py-1 rounded text-sm flex items-center gap-1 transition-colors ${
              refreshing
                ? 'bg-gray-600 cursor-wait'
                : object.isActive
                  ? 'bg-blue-600 hover:bg-blue-700'
                  : 'bg-gray-600 cursor-not-allowed'
            }`}
            title={object.isActive ? 'רענן ניתוח' : 'האובייקט לא פעיל בסצנה'}
          >
            {refreshing ? (
              <>
                <span className="animate-spin">⟳</span>
                <span>מנתח...</span>
              </>
            ) : (
              <>
                <span>🔄</span>
                <span>רענן ניתוח</span>
              </>
            )}
          </button>
        </div>
        {refreshError && (
          <div className="text-red-400 text-sm mb-2 bg-red-900/30 p-2 rounded">
            שגיאה: {refreshError}
          </div>
        )}

        {/* Analysis content with image */}
        <div className="flex gap-4">
          {/* Cutout image in analysis section */}
          {object.analysis?.cutout_image && (
            <div className="flex-shrink-0">
              <img
                src={`data:image/jpeg;base64,${object.analysis.cutout_image}`}
                alt="Analysis cutout"
                className="w-32 h-32 object-cover rounded-lg border-2 border-gray-600"
              />
            </div>
          )}

          {/* Analysis details */}
          <div className="flex-1 grid grid-cols-2 gap-2 text-sm">
          {object.type === 'person' && (
            <>
              <div>
                <span className="text-gray-400">לבוש:</span>{' '}
                <span className={!object.analysis?.clothing ? 'text-gray-500 italic' : ''}>
                  {object.analysis?.clothing || 'לא זוהה'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">צבע לבוש:</span>{' '}
                <span className={!object.analysis?.clothingColor ? 'text-gray-500 italic' : ''}>
                  {object.analysis?.clothingColor || 'לא זוהה'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">מגדר:</span>{' '}
                <span className={!object.analysis?.gender ? 'text-gray-500 italic' : ''}>
                  {object.analysis?.gender || 'לא זוהה'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">טווח גילאים:</span>{' '}
                <span className={!object.analysis?.ageRange ? 'text-gray-500 italic' : ''}>
                  {object.analysis?.ageRange || 'לא זוהה'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">חמוש:</span>{' '}
                <span className={object.analysis?.armed ? 'text-red-400 font-bold' : ''}>
                  {object.analysis?.armed ? 'כן' : 'לא'}
                </span>
              </div>
              {object.analysis?.weaponType && (
                <div className="text-red-400">
                  <span className="text-gray-400">סוג נשק:</span> {object.analysis.weaponType}
                </div>
              )}
              <div>
                <span className="text-gray-400">חשוד:</span>{' '}
                <span className={object.analysis?.suspicious ? 'text-orange-400' : ''}>
                  {object.analysis?.suspicious ? 'כן' : 'לא'}
                </span>
              </div>
              {object.analysis?.suspiciousReason && (
                <div className="text-orange-400 col-span-2">
                  <span className="text-gray-400">סיבה לחשד:</span> {object.analysis.suspiciousReason}
                </div>
              )}
            </>
          )}
          {object.type === 'vehicle' && (
            <>
              <div>
                <span className="text-gray-400">צבע:</span>{' '}
                <span className={!isValidValue(object.analysis?.color) ? 'text-gray-500 italic' : ''}>
                  {isValidValue(object.analysis?.color) ? object.analysis.color : 'לא זוהה'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">יצרן:</span>{' '}
                <span className={!isValidValue(object.analysis?.manufacturer) ? 'text-gray-500 italic' : ''}>
                  {isValidValue(object.analysis?.manufacturer) ? object.analysis.manufacturer : 'לא זוהה'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">דגם:</span>{' '}
                <span className={!isValidValue(object.analysis?.model) ? 'text-gray-500 italic' : ''}>
                  {isValidValue(object.analysis?.model) ? object.analysis.model : 'לא זוהה'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">סוג רכב:</span>{' '}
                <span className={!isValidValue(object.analysis?.vehicleType) ? 'text-gray-500 italic' : ''}>
                  {isValidValue(object.analysis?.vehicleType) ? object.analysis.vehicleType : 'לא זוהה'}
                </span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-400">לוחית רישוי:</span>{' '}
                {isValidValue(object.analysis?.licensePlate) ? (
                  <span className="text-lg font-mono bg-yellow-100 text-black px-2 py-1 rounded mr-2">
                    {object.analysis.licensePlate}
                  </span>
                ) : (
                  <span className="text-gray-500 italic">לא זוהה</span>
                )}
              </div>
            </>
          )}
          {object.analysis?.description && (
            <div className="col-span-2 mt-2">
              <span className="text-gray-400">תיאור:</span>
              <p className="mt-1">{object.analysis.description}</p>
            </div>
          )}
          {!object.analysis && (
            <div className="col-span-2 text-gray-500 italic text-center py-4">
              לא בוצע ניתוח עדיין. לחץ על "רענן ניתוח" כדי לנתח את האובייקט.
            </div>
          )}
          </div>
        </div>
      </div>

      {/* Appearances */}
      {object.appearances && object.appearances.length > 0 && (
        <div className="bg-gray-800 rounded p-4">
          <h4 className="font-bold mb-2">היסטוריית הופעות ({object.appearances.length})</h4>
          <div className="max-h-48 overflow-y-auto space-y-2">
            {object.appearances.slice().reverse().slice(0, 20).map((app, idx) => (
              <div key={idx} className="text-sm bg-gray-700/50 rounded p-2">
                <div className="flex justify-between">
                  <span>📹 {app.cameraName || app.cameraId}</span>
                  <span className="text-gray-400">
                    {app.timestamp ? new Date(app.timestamp).toLocaleTimeString('he-IL') : '-'}
                  </span>
                </div>
                {app.confidence && (
                  <div className="text-gray-400">ביטחון: {(app.confidence * 100).toFixed(1)}%</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metadata */}
      {object.metadata && Object.keys(object.metadata).length > 0 && (
        <div className="bg-gray-800 rounded p-4">
          <h4 className="font-bold mb-2">מטא-דאטא נוסף</h4>
          <pre className="text-xs bg-gray-900 p-2 rounded overflow-auto max-h-32">
            {JSON.stringify(object.metadata, null, 2)}
          </pre>
        </div>
      )}

      {/* Notes */}
      {object.notes && (
        <div className="bg-gray-800 rounded p-4">
          <h4 className="font-bold mb-2">הערות</h4>
          <p className="text-sm">{object.notes}</p>
        </div>
      )}

      {/* Tags */}
      {object.tags && object.tags.length > 0 && (
        <div className="flex gap-2 flex-wrap">
          {object.tags.map((tag, idx) => (
            <span key={idx} className="bg-blue-600/30 text-blue-300 px-2 py-1 rounded text-sm">
              #{tag}
            </span>
          ))}
        </div>
      )}

      {/* Debug info */}
      <div className="text-xs text-gray-500 border-t border-gray-700 pt-4">
        <div>MongoDB ID: {object._id}</div>
        <div>Track ID: {object.trackId}</div>
        <div>Created: {formatDate(object.createdAt)}</div>
        <div>Updated: {formatDate(object.updatedAt)}</div>
      </div>
    </div>
  );
}
