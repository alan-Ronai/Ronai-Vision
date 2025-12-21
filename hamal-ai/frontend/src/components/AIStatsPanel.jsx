import { useState, useEffect, useCallback, useRef } from 'react';

const AI_SERVICE_URL = import.meta.env.VITE_AI_SERVICE_URL || 'http://localhost:8000';

// Progress bar component for timing stats
function TimingBar({ label, value = 0, maxValue = 100, unit = 'ms', color = 'blue' }) {
  const safeValue = typeof value === 'number' ? value : 0;
  const percentage = Math.min((safeValue / maxValue) * 100, 100);
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500',
    purple: 'bg-purple-500',
    cyan: 'bg-cyan-500',
  };

  // Determine color based on percentage
  let barColor = colorClasses[color] || 'bg-blue-500';
  if (percentage > 80) barColor = 'bg-red-500';
  else if (percentage > 60) barColor = 'bg-yellow-500';

  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-300">{label}</span>
        <span className="text-white font-mono">{safeValue.toFixed(1)} {unit}</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// Counter component
function Counter({ label, value = 0, icon, trend }) {
  const safeValue = typeof value === 'number' ? value : 0;
  return (
    <div className="bg-gray-700 rounded-lg p-3 text-center">
      <div className="text-2xl mb-1">{icon}</div>
      <div className="text-xl font-bold text-white">{safeValue.toLocaleString()}</div>
      <div className="text-xs text-gray-400">{label}</div>
      {trend !== undefined && trend > 0 && (
        <div className="text-xs mt-1 text-green-400">
          +{trend}/s
        </div>
      )}
    </div>
  );
}

// Status indicator
function StatusIndicator({ label, active, count }) {
  return (
    <div className="flex items-center gap-2 bg-gray-700 rounded px-3 py-2">
      <div className={`w-2 h-2 rounded-full ${active ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
      <span className="text-sm text-gray-300">{label}</span>
      {count !== undefined && (
        <span className="text-sm font-mono text-white ml-auto">{count}</span>
      )}
    </div>
  );
}

export default function AIStatsPanel({ isOpen, onClose }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [rates, setRates] = useState({ frames: 0, detections: 0, events: 0 });
  const prevCountersRef = useRef({});

  const fetchStats = useCallback(async () => {
    try {
      console.log('Fetching stats from:', `${AI_SERVICE_URL}/api/stats/realtime`);
      const res = await fetch(`${AI_SERVICE_URL}/api/stats/realtime`);
      if (!res.ok) throw new Error('Failed to fetch stats');
      const data = await res.json();
      console.log('Stats received:', data);

      // Ensure counters exist
      const counters = data.counters || {};

      // Calculate rates (per second)
      const prev = prevCountersRef.current;
      if (prev.timestamp && data.timestamp) {
        const elapsed = data.timestamp - prev.timestamp;
        if (elapsed > 0 && elapsed < 5) { // Sanity check
          setRates({
            frames: Math.round(((counters.frames_processed || 0) - (prev.frames_processed || 0)) / elapsed),
            detections: Math.round(((counters.detections || 0) - (prev.detections || 0)) / elapsed),
            events: Math.round(((counters.events_sent || 0) - (prev.events_sent || 0)) / elapsed),
          });
        }
      }

      prevCountersRef.current = {
        timestamp: data.timestamp,
        frames_processed: counters.frames_processed || 0,
        detections: counters.detections || 0,
        events_sent: counters.events_sent || 0,
      };

      setStats(data);
      setError(null);
      setLoading(false);
    } catch (err) {
      console.error('Stats fetch error:', err);
      setError(err.message);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    console.log('AIStatsPanel isOpen:', isOpen);
    if (!isOpen) return;

    // Reset state when opening
    setLoading(true);
    setError(null);
    prevCountersRef.current = {};

    fetchStats();
    const interval = setInterval(fetchStats, 1000);

    return () => clearInterval(interval);
  }, [isOpen, fetchStats]);

  console.log('AIStatsPanel rendering, isOpen:', isOpen, 'loading:', loading, 'stats:', stats, 'error:', error);
  if (!isOpen) return null;

  // Debug: Show raw data if something seems off
  if (stats && !stats.performance) {
    console.warn('Stats received but no performance data:', stats);
  }

  // Safe extraction with defaults
  const perf = stats?.performance || {};
  const counters = stats?.counters || {};
  const tracker = stats?.tracker || {};
  const pressure = stats?.pressure || {};
  const config = stats?.config || {};

  // Calculate FPS efficiency
  const targetFps = perf.target_fps || 15;
  const actualFps = perf.actual_fps || 0;
  const fpsEfficiency = targetFps > 0 ? Math.round((actualFps / targetFps) * 100) : 0;

  // Format uptime
  const uptimeSeconds = stats?.uptime_seconds || 0;
  const uptimeMin = Math.floor(uptimeSeconds / 60);
  const uptimeSec = Math.floor(uptimeSeconds % 60);

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" dir="rtl">
      <div className="bg-gray-800 rounded-xl w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gray-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span>📊</span>
            <span>סטטיסטיקות AI בזמן אמת</span>
            {stats && (
              <span className="text-sm font-normal text-gray-400 mr-4">
                זמן פעילות: {uptimeMin}:{String(uptimeSec).padStart(2, '0')}
              </span>
            )}
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading && !stats ? (
            <div className="text-center py-8 text-gray-400">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
              טוען...
            </div>
          ) : error ? (
            <div className="text-center py-8">
              <div className="text-red-400 mb-4">{error}</div>
              <button
                onClick={fetchStats}
                className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
              >
                נסה שוב
              </button>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Performance Timing Section */}
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <span>⏱️</span>
                  <span>זמני עיבוד (מילישניות)</span>
                  <span className="text-sm font-normal text-gray-400 mr-auto">
                    סה"כ: {(perf.total_frame_ms || 0).toFixed(1)} ms/frame
                  </span>
                </h3>

                <div className="grid grid-cols-2 gap-x-8 gap-y-2">
                  <TimingBar label="YOLO Detection" value={perf.yolo_ms} maxValue={50} color="yellow" />
                  <TimingBar label="ReID Extraction" value={perf.reid_ms} maxValue={30} color="purple" />
                  <TimingBar label="BoT-SORT Tracker" value={perf.tracker_ms} maxValue={20} color="cyan" />
                  <TimingBar label="Recovery" value={perf.recovery_ms} maxValue={30} color="green" />
                  <TimingBar label="Drawing" value={perf.drawing_ms} maxValue={15} color="blue" />
                  <div className="flex items-center gap-4">
                    <div className={`text-2xl font-bold ${fpsEfficiency >= 90 ? 'text-green-400' : fpsEfficiency >= 70 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {actualFps} FPS
                    </div>
                    <div className="text-sm text-gray-400">
                      יעד: {targetFps} FPS ({fpsEfficiency}%)
                    </div>
                  </div>
                </div>
              </div>

              {/* Counters Grid */}
              <div className="grid grid-cols-4 gap-4">
                <Counter
                  label="פריימים מעובדים"
                  value={counters.frames_processed}
                  icon="🎬"
                  trend={rates.frames}
                />
                <Counter
                  label="זיהויים"
                  value={counters.detections}
                  icon="👁️"
                  trend={rates.detections}
                />
                <Counter
                  label="ReID Extractions"
                  value={counters.reid_extractions}
                  icon="🔍"
                />
                <Counter
                  label="ReID Recoveries"
                  value={counters.reid_recoveries}
                  icon="🔄"
                />
                <Counter
                  label="אירועים נשלחו"
                  value={counters.events_sent}
                  icon="📤"
                  trend={rates.events}
                />
                <Counter
                  label="Gemini Calls"
                  value={counters.gemini_calls}
                  icon="🤖"
                />
                <Counter
                  label="פריימים נזרקו"
                  value={counters.frames_dropped}
                  icon="🗑️"
                />
                <Counter
                  label="מצלמות פעילות"
                  value={pressure.active_cameras?.length}
                  icon="📹"
                />
              </div>

              {/* Tracker Stats */}
              <div className="grid grid-cols-2 gap-4">
                {/* BoT-SORT */}
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
                    <span>🎯</span>
                    <span>BoT-SORT Tracker</span>
                  </h3>
                  <div className="grid grid-cols-2 gap-2">
                    <StatusIndicator
                      label="אנשים"
                      active={(tracker.bot_sort?.persons || 0) > 0}
                      count={tracker.bot_sort?.persons || 0}
                    />
                    <StatusIndicator
                      label="רכבים"
                      active={(tracker.bot_sort?.vehicles || 0) > 0}
                      count={tracker.bot_sort?.vehicles || 0}
                    />
                    <StatusIndicator
                      label="סה״כ Tracks"
                      active={(tracker.bot_sort?.total_tracks || 0) > 0}
                      count={tracker.bot_sort?.total_tracks || 0}
                    />
                    <StatusIndicator
                      label="Tracks נמחקו"
                      active={false}
                      count={tracker.bot_sort?.deleted_tracks || 0}
                    />
                  </div>
                </div>

                {/* ReID Tracker */}
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
                    <span>🔍</span>
                    <span>ReID Tracker</span>
                  </h3>
                  <div className="grid grid-cols-2 gap-2">
                    <StatusIndicator
                      label="סה״כ Tracked"
                      active={(tracker.reid?.total_tracked || 0) > 0}
                      count={tracker.reid?.total_tracked || 0}
                    />
                    <StatusIndicator
                      label="אנשים חמושים"
                      active={(tracker.reid?.armed_persons || 0) > 0}
                      count={tracker.reid?.armed_persons || 0}
                    />
                    <StatusIndicator
                      label="BoT-SORT"
                      active={tracker.bot_sort?.active !== false}
                    />
                    <StatusIndicator
                      label="Stable Tracker"
                      active={(tracker.stable?.total_objects || 0) > 0}
                      count={tracker.stable?.total_objects || 0}
                    />
                  </div>
                </div>
              </div>

              {/* Pressure / Queue Stats */}
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
                  <span>📊</span>
                  <span>עומס מערכת</span>
                </h3>
                <div className="grid grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className={`text-2xl font-bold ${(pressure.pending_frames || 0) > 5 ? 'text-red-400' : 'text-green-400'}`}>
                      {pressure.pending_frames || 0}
                    </div>
                    <div className="text-xs text-gray-400">פריימים בהמתנה</div>
                  </div>
                  <div className="text-center">
                    <div className={`text-2xl font-bold ${(pressure.result_queue_size || 0) > 10 ? 'text-red-400' : 'text-green-400'}`}>
                      {pressure.result_queue_size || 0}
                    </div>
                    <div className="text-xs text-gray-400">Result Queue</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">
                      {typeof stats?.recording?.active_recordings === 'object'
                        ? Object.keys(stats?.recording?.active_recordings || {}).length
                        : (stats?.recording?.active_recordings || 0)}
                    </div>
                    <div className="text-xs text-gray-400">הקלטות פעילות</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-400">
                      {stats?.recording?.completed_recordings || 0}
                    </div>
                    <div className="text-xs text-gray-400">הקלטות הושלמו</div>
                  </div>
                </div>
              </div>

              {/* Configuration Display */}
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
                  <span>⚙️</span>
                  <span>תצורה נוכחית</span>
                </h3>
                <div className="grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Detection FPS:</span>
                    <span className="text-white mr-2">{config.detection_fps || '-'}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Stream FPS:</span>
                    <span className="text-white mr-2">{config.stream_fps || '-'}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">YOLO Confidence:</span>
                    <span className="text-white mr-2">{config.yolo_confidence ? `${(config.yolo_confidence * 100).toFixed(0)}%` : '-'}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">ReID Recovery:</span>
                    <span className={`mr-2 ${config.use_reid_recovery ? 'text-green-400' : 'text-gray-500'}`}>
                      {config.use_reid_recovery ? 'פעיל' : 'כבוי'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-700 px-6 py-3 text-sm text-gray-400 flex justify-between">
          <span>🔄 מתעדכן בזמן אמת (כל שנייה)</span>
          <span>Endpoint: /api/stats/realtime</span>
        </div>
      </div>
    </div>
  );
}
