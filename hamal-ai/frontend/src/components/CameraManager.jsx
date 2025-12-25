import { useState, useEffect } from 'react';
import WebcamShare from './WebcamShare';

// Use relative URLs to leverage Vite proxy (avoids mixed content issues with HTTPS)
const API_URL = '';
const AI_API_PROXY = '/ai-api';  // Proxied to AI service /api

export default function CameraManager({ isOpen, onClose }) {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingCamera, setEditingCamera] = useState(null);
  const [formData, setFormData] = useState({
    cameraId: '',
    name: '',
    location: '',
    rtspUrl: '',
    username: '',
    password: '',
    type: 'rtsp',
    aiEnabled: true,
    deviceIndex: '0'
  });
  const [availableWebcams, setAvailableWebcams] = useState([]);

  useEffect(() => {
    if (isOpen) {
      fetchCameras();
      fetchAvailableWebcams();
    }
  }, [isOpen]);

  const fetchAvailableWebcams = async () => {
    try {
      const res = await fetch(`/detection/webcams`);
      const data = await res.json();
      if (data.devices) {
        setAvailableWebcams(data.devices);
      }
    } catch (error) {
      console.log('Failed to fetch webcams (AI service may be offline):', error);
    }
  };

  const fetchCameras = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/cameras`);
      const data = await res.json();
      setCameras(data);
    } catch (error) {
      console.error('Failed to fetch cameras:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const url = editingCamera
        ? `${API_URL}/api/cameras/${editingCamera.cameraId}`
        : `${API_URL}/api/cameras`;

      // For webcam type, use deviceIndex as rtspUrl
      const submitData = { ...formData };
      if (submitData.type === 'webcam') {
        submitData.rtspUrl = submitData.deviceIndex || '0';
      }

      const res = await fetch(url, {
        method: editingCamera ? 'PUT' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(submitData)
      });

      if (res.ok) {
        await fetchCameras();
        resetForm();
      } else {
        const error = await res.json();
        alert(`שגיאה: ${error.error}`);
      }
    } catch (error) {
      console.error('Failed to save camera:', error);
      alert('שגיאה בשמירת המצלמה');
    }
  };

  const handleDelete = async (camera) => {
    if (!confirm(`למחוק את המצלמה "${camera.name}"?`)) return;

    try {
      await fetch(`${API_URL}/api/cameras/${camera.cameraId}`, { method: 'DELETE' });
      await fetchCameras();
    } catch (error) {
      console.error('Failed to delete camera:', error);
    }
  };

  const handleTest = async (camera) => {
    try {
      await fetch(`${API_URL}/api/cameras/${camera.cameraId}/test`, { method: 'POST' });
      // Status update will come via socket
    } catch (error) {
      console.error('Failed to test camera:', error);
    }
  };

  const handleEdit = (camera) => {
    setEditingCamera(camera);
    setFormData({
      cameraId: camera.cameraId || '',
      name: camera.name || '',
      location: camera.location || '',
      rtspUrl: camera.rtspUrl || '',
      username: camera.username || '',
      password: '',
      type: camera.type || 'rtsp',
      aiEnabled: camera.aiEnabled !== false,
      deviceIndex: camera.type === 'webcam' ? (camera.rtspUrl || '0') : '0'
    });
    setShowForm(true);
  };

  const resetForm = () => {
    setEditingCamera(null);
    setFormData({
      cameraId: '',
      name: '',
      location: '',
      rtspUrl: '',
      username: '',
      password: '',
      type: 'rtsp',
      aiEnabled: true,
      deviceIndex: '0'
    });
    setShowForm(false);
  };

  const getStatusInfo = (status) => {
    const map = {
      online: { color: 'bg-green-500', text: 'מחובר' },
      connecting: { color: 'bg-yellow-500 animate-pulse', text: 'מתחבר...' },
      error: { color: 'bg-red-500', text: 'שגיאה' },
      offline: { color: 'bg-gray-500', text: 'מנותק' }
    };
    return map[status] || map.offline;
  };

  const getTypeLabel = (type) => {
    const labels = {
      rtsp: 'RTSP',
      simulator: 'סימולטור',
      file: 'קובץ',
      webcam: 'מצלמת רשת',
      'browser-webcam': 'מצלמת דפדפן'
    };
    return labels[type] || type;
  };

  // Camera preview component
  const CameraPreview = ({ cameraId, status }) => {
    const [previewUrl, setPreviewUrl] = useState(null);
    const [previewError, setPreviewError] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
      // Generate a new URL with timestamp to avoid caching
      const url = `${AI_API_PROXY}/stream/snapshot/${cameraId}?t=${Date.now()}`;
      setPreviewUrl(url);
      setPreviewError(false);
      setLoading(true);
    }, [cameraId, status]);

    return (
      <div className="w-24 h-16 bg-gray-800 rounded overflow-hidden flex-shrink-0 relative">
        {loading && !previewError && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-4 h-4 border-2 border-gray-500 border-t-blue-500 rounded-full animate-spin" />
          </div>
        )}
        {previewUrl && !previewError && (
          <img
            src={previewUrl}
            alt={`Preview ${cameraId}`}
            className={`w-full h-full object-cover ${loading ? 'opacity-0' : 'opacity-100'} transition-opacity`}
            onLoad={() => setLoading(false)}
            onError={() => {
              setPreviewError(true);
              setLoading(false);
            }}
          />
        )}
        {previewError && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500">
            <span className="text-2xl">📹</span>
          </div>
        )}
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" dir="rtl">
      <div className="bg-gray-800 rounded-xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gray-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span>📹</span>
            <span>ניהול מצלמות</span>
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Browser Webcam Share Section */}
          <div className="mb-6">
            <WebcamShare
              onStreamStarted={() => fetchCameras()}
              onStreamStopped={() => fetchCameras()}
            />
          </div>

          {/* Add Camera Button */}
          {!showForm && (
            <button
              onClick={() => setShowForm(true)}
              className="mb-4 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg flex items-center gap-2"
            >
              ➕ הוסף מצלמה חדשה
            </button>
          )}

          {/* Camera Form */}
          {showForm && (
            <form onSubmit={handleSubmit} className="mb-6 bg-gray-700 rounded-lg p-4">
              <h3 className="font-bold mb-4">
                {editingCamera ? 'ערוך מצלמה' : 'מצלמה חדשה'}
              </h3>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">מזהה מצלמה *</label>
                  <input
                    type="text"
                    value={formData.cameraId}
                    onChange={(e) => setFormData({ ...formData, cameraId: e.target.value })}
                    className="w-full bg-gray-600 rounded px-3 py-2 text-white"
                    placeholder="cam-1"
                    required
                    disabled={!!editingCamera}
                    dir="ltr"
                  />
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">שם המצלמה *</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full bg-gray-600 rounded px-3 py-2 text-white"
                    placeholder="שער ראשי"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">מיקום</label>
                  <input
                    type="text"
                    value={formData.location}
                    onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                    className="w-full bg-gray-600 rounded px-3 py-2 text-white"
                    placeholder="כניסה ראשית"
                  />
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">סוג מצלמה</label>
                  <select
                    value={formData.type}
                    onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                    className="w-full bg-gray-600 rounded px-3 py-2 text-white"
                  >
                    <option value="rtsp">RTSP</option>
                    <option value="simulator">סימולטור</option>
                    <option value="file">קובץ וידאו</option>
                    <option value="webcam">מצלמת רשת</option>
                  </select>
                </div>

                {formData.type === 'webcam' ? (
                  <div className="col-span-2">
                    <label className="block text-sm text-gray-400 mb-1">מצלמת רשת</label>
                    <select
                      value={formData.deviceIndex}
                      onChange={(e) => setFormData({ ...formData, deviceIndex: e.target.value })}
                      className="w-full bg-gray-600 rounded px-3 py-2 text-white"
                    >
                      {availableWebcams.length > 0 ? (
                        availableWebcams.map((cam) => (
                          <option key={cam.index} value={cam.index.toString()}>
                            {cam.name} {cam.available ? '(זמין)' : '(לא זמין)'}
                          </option>
                        ))
                      ) : (
                        <>
                          <option value="0">Webcam 0 (ברירת מחדל)</option>
                          <option value="1">Webcam 1</option>
                          <option value="2">Webcam 2</option>
                        </>
                      )}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      בחר את מצלמת הרשת המחוברת למחשב
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="col-span-2">
                      <label className="block text-sm text-gray-400 mb-1">כתובת RTSP</label>
                      <input
                        type="text"
                        value={formData.rtspUrl}
                        onChange={(e) => setFormData({ ...formData, rtspUrl: e.target.value })}
                        className="w-full bg-gray-600 rounded px-3 py-2 font-mono text-sm text-white"
                        placeholder={formData.type === 'file' ? 'assets/video.mp4' : 'rtsp://192.168.1.100:554/stream1'}
                        dir="ltr"
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-400 mb-1">שם משתמש</label>
                      <input
                        type="text"
                        value={formData.username}
                        onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                        className="w-full bg-gray-600 rounded px-3 py-2 text-white"
                        placeholder="admin"
                        dir="ltr"
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-400 mb-1">סיסמה</label>
                      <input
                        type="password"
                        value={formData.password}
                        onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                        className="w-full bg-gray-600 rounded px-3 py-2 text-white"
                        placeholder={editingCamera ? '(ללא שינוי)' : ''}
                        dir="ltr"
                      />
                    </div>
                  </>
                )}

                <div className="col-span-2 flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="aiEnabled"
                    checked={formData.aiEnabled}
                    onChange={(e) => setFormData({ ...formData, aiEnabled: e.target.checked })}
                    className="w-4 h-4"
                  />
                  <label htmlFor="aiEnabled" className="text-white">הפעל זיהוי AI</label>
                </div>
              </div>

              <div className="flex gap-2 mt-4">
                <button
                  type="submit"
                  className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
                >
                  {editingCamera ? 'שמור שינויים' : 'הוסף מצלמה'}
                </button>
                <button
                  type="button"
                  onClick={resetForm}
                  className="bg-gray-600 hover:bg-gray-500 px-4 py-2 rounded"
                >
                  ביטול
                </button>
              </div>
            </form>
          )}

          {/* Camera List */}
          {loading ? (
            <div className="text-center py-8 text-gray-400">טוען...</div>
          ) : cameras.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <div className="text-4xl mb-2">📹</div>
              <p>אין מצלמות מוגדרות</p>
              <p className="text-sm">הוסף מצלמה חדשה להתחיל</p>
            </div>
          ) : (
            <div className="space-y-3">
              {cameras.map((camera) => {
                const statusInfo = getStatusInfo(camera.status);
                return (
                  <div
                    key={camera._id || camera.cameraId}
                    className="bg-gray-700 rounded-lg p-4 flex items-center gap-4"
                  >
                    {/* Camera Preview */}
                    <CameraPreview cameraId={camera.cameraId} status={camera.status} />

                    {/* Status Indicator */}
                    <div className={`w-3 h-3 rounded-full flex-shrink-0 ${statusInfo.color}`} />

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="font-bold text-white flex items-center gap-2">
                        {camera.name}
                        {camera.isMainCamera && (
                          <span className="text-xs bg-yellow-600 px-2 py-0.5 rounded">ראשית</span>
                        )}
                      </div>
                      <div className="text-sm text-gray-400 flex items-center gap-2 flex-wrap">
                        {camera.location && <span>{camera.location}</span>}
                        <span>|</span>
                        <span>{statusInfo.text}</span>
                        <span className="px-2 py-0.5 bg-gray-600 rounded text-xs">
                          {getTypeLabel(camera.type)}
                        </span>
                        {camera.aiEnabled && (
                          <span className="px-2 py-0.5 bg-blue-600 rounded text-xs">AI</span>
                        )}
                      </div>
                      {camera.rtspUrl && (
                        <div className="text-xs text-gray-500 font-mono mt-1 truncate" dir="ltr">
                          {camera.rtspUrl.replace(/:[^:@]+@/, ':***@')}
                        </div>
                      )}
                      {camera.lastError && camera.status === 'error' && (
                        <div className="text-xs text-red-400 mt-1">
                          שגיאה: {camera.lastError}
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex gap-2 flex-shrink-0">
                      <button
                        onClick={() => handleTest(camera)}
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm disabled:opacity-50"
                        disabled={camera.status === 'connecting'}
                      >
                        {camera.status === 'connecting' ? '...' : 'בדוק'}
                      </button>
                      <button
                        onClick={() => handleEdit(camera)}
                        className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm"
                      >
                        ערוך
                      </button>
                      <button
                        onClick={() => handleDelete(camera)}
                        className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
                      >
                        מחק
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-700 px-6 py-3 flex justify-between items-center text-sm text-gray-400">
          <span>{cameras.length} מצלמות</span>
          <span>{cameras.filter(c => c.status === 'online').length} מחוברות</span>
        </div>
      </div>
    </div>
  );
}
