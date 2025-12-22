import { useState, useEffect } from 'react';

const AI_SERVICE_URL = import.meta.env.VITE_AI_SERVICE_URL || 'http://localhost:8000';

export default function DetectionSettings({ isOpen, onClose }) {
  const [config, setConfig] = useState({
    detection_fps: 15,
    stream_fps: 15,
    reader_fps: 25,
    recording_fps: 15,
    use_reid_recovery: false,
    yolo_confidence: 0.35,
    weapon_confidence: 0.40,
    recovery_confidence: 0.20,
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (isOpen) {
      fetchConfig();
    }
  }, [isOpen]);

  const fetchConfig = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${AI_SERVICE_URL}/detection/config`);
      const data = await res.json();
      setConfig({
        detection_fps: data.detection_fps || 15,
        stream_fps: data.stream_fps || 15,
        reader_fps: data.reader_fps || 25,
        recording_fps: data.recording_fps || 15,
        use_reid_recovery: data.use_reid_recovery || false,
        yolo_confidence: data.yolo_confidence || 0.35,
        weapon_confidence: data.weapon_confidence || 0.40,
        recovery_confidence: data.recovery_confidence || 0.20,
      });
    } catch (error) {
      console.error('Failed to fetch detection config:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateFPS = async () => {
    try {
      setSaving(true);
      const params = new URLSearchParams({
        detection_fps: config.detection_fps,
        stream_fps: config.stream_fps,
        reader_fps: config.reader_fps,
        recording_fps: config.recording_fps,
      });

      const res = await fetch(`${AI_SERVICE_URL}/detection/config/fps?${params}`, {
        method: 'POST',
      });

      if (res.ok) {
        alert('✅ FPS עודכן בהצלחה!\n\nהזרמים יתחברו מחדש אוטומטית.');
        // Reload page to reconnect streams with new FPS
        setTimeout(() => window.location.reload(), 1000);
      }
    } catch (error) {
      console.error('Failed to update FPS:', error);
      alert('שגיאה בעדכון FPS');
    } finally {
      setSaving(false);
    }
  };

  const updateConfidence = async () => {
    try {
      setSaving(true);
      const params = new URLSearchParams({
        yolo_confidence: config.yolo_confidence,
        weapon_confidence: config.weapon_confidence,
        recovery_confidence: config.recovery_confidence,
      });

      const res = await fetch(`${AI_SERVICE_URL}/detection/config/confidence?${params}`, {
        method: 'POST',
      });

      if (res.ok) {
        alert('✅ רמות הביטחון עודכנו בהצלחה!\n\nהשינויים ייכנסו לתוקף מיידית.');
      }
    } catch (error) {
      console.error('Failed to update confidence:', error);
      alert('שגיאה בעדכון רמות הביטחון');
    } finally {
      setSaving(false);
    }
  };

  const toggleReidRecovery = async (enabled) => {
    try {
      setSaving(true);
      const res = await fetch(
        `${AI_SERVICE_URL}/detection/config/reid-recovery?enabled=${enabled}`,
        { method: 'POST' }
      );

      if (res.ok) {
        setConfig({ ...config, use_reid_recovery: enabled });
        alert(`✅ ReID Recovery ${enabled ? 'הופעל' : 'הושבת'}`);
      }
    } catch (error) {
      console.error('Failed to toggle ReID recovery:', error);
      alert('שגיאה בשינוי הגדרות ReID Recovery');
    } finally {
      setSaving(false);
    }
  };

  const applyFPSPreset = (preset) => {
    const presets = {
      high: { detection: 25, stream: 25, reader: 30, recording: 25 },
      balanced: { detection: 15, stream: 15, reader: 20, recording: 15 },
      low: { detection: 10, stream: 10, reader: 15, recording: 10 },
      power: { detection: 5, stream: 5, reader: 10, recording: 10 },
    };
    const { detection, stream, reader, recording } = presets[preset];
    setConfig({
      ...config,
      detection_fps: detection,
      stream_fps: stream,
      reader_fps: reader,
      recording_fps: recording,
    });
  };

  const applyConfidencePreset = (preset) => {
    const presets = {
      sensitive: { yolo: 0.25, weapon: 0.30, recovery: 0.15 },
      balanced: { yolo: 0.35, weapon: 0.40, recovery: 0.20 },
      precise: { yolo: 0.50, weapon: 0.60, recovery: 0.30 },
      security: { yolo: 0.30, weapon: 0.35, recovery: 0.18 },
    };
    const { yolo, weapon, recovery } = presets[preset];
    setConfig({
      ...config,
      yolo_confidence: yolo,
      weapon_confidence: weapon,
      recovery_confidence: recovery,
    });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" dir="rtl">
      <div className="bg-gray-800 rounded-xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gray-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span>⚙️</span>
            <span>הגדרות זיהוי AI</span>
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="text-center py-8 text-gray-400">טוען...</div>
          ) : (
            <div className="space-y-6">
              {/* FPS Settings */}
              <div className="bg-gray-700 rounded-lg p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <span>🎥</span>
                  <span>קצב פריימים (FPS)</span>
                </h3>

                {/* FPS Presets */}
                <div className="mb-4 flex gap-2 flex-wrap">
                  <button
                    onClick={() => applyFPSPreset('high')}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
                  >
                    ⚡ גבוה (30 FPS)
                  </button>
                  <button
                    onClick={() => applyFPSPreset('balanced')}
                    className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm"
                  >
                    ⚖️ מאוזן (15 FPS)
                  </button>
                  <button
                    onClick={() => applyFPSPreset('low')}
                    className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-sm"
                  >
                    🔋 נמוך (10 FPS)
                  </button>
                  <button
                    onClick={() => applyFPSPreset('power')}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                  >
                    💾 חיסכון (5 FPS)
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  {/* Detection FPS */}
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      <span className="text-yellow-400">🤖</span> FPS זיהוי (AI): {config.detection_fps}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      value={config.detection_fps}
                      onChange={(e) =>
                        setConfig({ ...config, detection_fps: Number(e.target.value) })
                      }
                      className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className="text-xs text-gray-400 mt-1">
                      כמה פעמים להריץ YOLO בשנייה (משפיע על CPU/GPU)
                    </p>
                  </div>

                  {/* Stream FPS */}
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      <span className="text-blue-400">📺</span> FPS הזרמה (וידאו): {config.stream_fps}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      value={config.stream_fps}
                      onChange={(e) =>
                        setConfig({ ...config, stream_fps: Number(e.target.value) })
                      }
                      className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className="text-xs text-gray-400 mt-1">
                      כמה פריימים להציג בממשק (חלקות הוידאו)
                    </p>
                  </div>

                  {/* Reader FPS */}
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      <span className="text-green-400">📡</span> FPS קריאה (RTSP): {config.reader_fps}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      value={config.reader_fps}
                      onChange={(e) =>
                        setConfig({ ...config, reader_fps: Number(e.target.value) })
                      }
                      className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className="text-xs text-gray-400 mt-1">
                      כמה פריימים לקרוא מהמצלמה (משפיע על רשת/דקודינג)
                    </p>
                  </div>

                  {/* Recording FPS */}
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      <span className="text-red-400">🔴</span> FPS הקלטה: {config.recording_fps}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      value={config.recording_fps}
                      onChange={(e) =>
                        setConfig({ ...config, recording_fps: Number(e.target.value) })
                      }
                      className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className="text-xs text-gray-400 mt-1">
                      כמה פריימים לשמור בהקלטות (גודל קבצים)
                    </p>
                  </div>
                </div>

                <button
                  onClick={updateFPS}
                  disabled={saving}
                  className="mt-4 bg-green-600 hover:bg-green-700 px-6 py-2 rounded disabled:opacity-50"
                >
                  {saving ? '...מעדכן' : '✅ עדכן FPS'}
                </button>
              </div>

              {/* Confidence Settings */}
              <div className="bg-gray-700 rounded-lg p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <span>🎯</span>
                  <span>רמות ביטחון זיהוי</span>
                </h3>

                {/* Confidence Presets */}
                <div className="mb-4 flex gap-2 flex-wrap">
                  <button
                    onClick={() => applyConfidencePreset('sensitive')}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
                  >
                    🔴 רגיש (יותר זיהויים)
                  </button>
                  <button
                    onClick={() => applyConfidencePreset('balanced')}
                    className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm"
                  >
                    ⚖️ מאוזן (ברירת מחדל)
                  </button>
                  <button
                    onClick={() => applyConfidencePreset('precise')}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                  >
                    🔵 מדויק (פחות זיהויים)
                  </button>
                  <button
                    onClick={() => applyConfidencePreset('security')}
                    className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-sm"
                  >
                    🛡️ ביטחוני (נשק רגיש)
                  </button>
                </div>

                <div className="space-y-4">
                  {/* YOLO Confidence */}
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      זיהוי כללי (YOLO): {(config.yolo_confidence * 100).toFixed(0)}%
                    </label>
                    <input
                      type="range"
                      min="0.10"
                      max="0.80"
                      step="0.05"
                      value={config.yolo_confidence}
                      onChange={(e) =>
                        setConfig({ ...config, yolo_confidence: Number(e.target.value) })
                      }
                      className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className="text-xs text-gray-400 mt-1">
                      נמוך = יותר זיהויים (כולל שגויים) | גבוה = פחות זיהויים (מדויקים יותר)
                    </p>
                  </div>

                  {/* Weapon Confidence */}
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      זיהוי נשק: {(config.weapon_confidence * 100).toFixed(0)}%
                    </label>
                    <input
                      type="range"
                      min="0.20"
                      max="0.80"
                      step="0.05"
                      value={config.weapon_confidence}
                      onChange={(e) =>
                        setConfig({ ...config, weapon_confidence: Number(e.target.value) })
                      }
                      className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className="text-xs text-gray-400 mt-1">
                      נמוך = זיהוי רגיש (יותר התרעות) | גבוה = שמרני (פחות התרעות שווא)
                    </p>
                  </div>

                  {/* Recovery Confidence */}
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      שחזור זיהוי (Recovery): {(config.recovery_confidence * 100).toFixed(0)}%
                    </label>
                    <input
                      type="range"
                      min="0.10"
                      max="0.40"
                      step="0.05"
                      value={config.recovery_confidence}
                      onChange={(e) =>
                        setConfig({ ...config, recovery_confidence: Number(e.target.value) })
                      }
                      className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                      disabled={!config.use_reid_recovery}
                    />
                    <p className="text-xs text-gray-400 mt-1">
                      פעיל רק כאשר ReID Recovery מופעל
                    </p>
                  </div>
                </div>

                <button
                  onClick={updateConfidence}
                  disabled={saving}
                  className="mt-4 bg-green-600 hover:bg-green-700 px-6 py-2 rounded disabled:opacity-50"
                >
                  {saving ? '...מעדכן' : '✅ עדכן רמות ביטחון'}
                </button>
              </div>

              {/* ReID Recovery Toggle */}
              <div className="bg-gray-700 rounded-lg p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <span>🔄</span>
                  <span>ReID Recovery</span>
                </h3>

                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    id="reidRecovery"
                    checked={config.use_reid_recovery}
                    onChange={(e) => toggleReidRecovery(e.target.checked)}
                    className="mt-1 w-5 h-5"
                    disabled={saving}
                  />
                  <div>
                    <label htmlFor="reidRecovery" className="text-white font-medium cursor-pointer">
                      הפעל שחזור מעקב מתקדם (ReID Recovery)
                    </label>
                    <p className="text-sm text-gray-400 mt-1">
                      עוזר לשמור על מעקב כאשר אובייקטים מוסתרים זמנית או כשיש תנאי תאורה קשים.
                      <br />
                      <span className="text-yellow-400">⚠️ עלול להשפיע על ביצועים (~10-20% CPU)</span>
                    </p>
                  </div>
                </div>
              </div>

              {/* Info Box */}
              <div className="bg-blue-900/30 border border-blue-500 rounded-lg p-4">
                <h4 className="font-bold mb-2 flex items-center gap-2">
                  <span>💡</span>
                  <span>טיפים</span>
                </h4>
                <ul className="text-sm space-y-1 text-gray-300">
                  <li>• FPS גבוה = חלק יותר אבל דורש יותר משאבים</li>
                  <li>• רמת ביטחון נמוכה = יותר זיהויים (כולל שגויים)</li>
                  <li>• רמת ביטחון גבוהה = פחות זיהויים (רק בטוחים)</li>
                  <li>• הגדרות מאוזנות מומלצות למרבית המקרים</li>
                  <li>• שינויי FPS דורשים התחברות מחדש של הזרמים</li>
                  <li>• שינויי רמות ביטחון נכנסים לתוקף מיידית</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-700 px-6 py-3 text-sm text-gray-400">
          <span>✨ השינויים חלים על כל המצלמות במערכת</span>
        </div>
      </div>
    </div>
  );
}
