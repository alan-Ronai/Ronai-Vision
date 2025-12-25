import { useApp } from '../context/AppContext';
import { useScenario } from '../context/ScenarioContext';

export default function StatusBar({ onOpenAIStats, onOpenActiveScenarios }) {
  const { connected, cameras, events, isEmergency } = useApp();
  const { soundVolume, setSoundVolume, soundMuted, setSoundMuted, scenario } = useScenario();

  const onlineCameras = cameras.filter(c => c.status === 'online').length;
  const criticalEvents = events.filter(e => e.severity === 'critical' && !e.acknowledged).length;
  const warningEvents = events.filter(e => e.severity === 'warning' && !e.acknowledged).length;

  // Check if dev mode is enabled (via VITE_DEV_MODE env variable)
  const isDevMode = import.meta.env.VITE_DEV_MODE === 'true';

  const currentTime = new Date().toLocaleTimeString('he-IL', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });

  const currentDate = new Date().toLocaleDateString('he-IL', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  return (
    <div className={`
      flex items-center justify-between px-4 py-2
      ${isEmergency ? 'bg-red-900/50' : 'bg-gray-800'}
      border-b border-gray-700 flex-shrink-0
    `}>
      {/* Logo and title */}
      <div className="flex items-center gap-3">
        <div className="text-2xl">🛡️</div>
        <div>
          <h1 className="text-lg font-bold">חמ"ל AI</h1>
          <p className="text-xs text-gray-400">מרכז שליטה ובקרה</p>
        </div>
      </div>

      {/* Status indicators */}
      <div className="flex items-center gap-6">
        {/* Connection status */}
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-sm">{connected ? 'מחובר' : 'מנותק'}</span>
        </div>

        {/* Cameras online */}
        <div className="flex items-center gap-2">
          <span className="text-gray-400">📹</span>
          <span className="text-sm">{onlineCameras}/{cameras.length} מצלמות</span>
        </div>

        {/* Alert counts - clickable to show active scenarios */}
        {(criticalEvents > 0 || scenario?.active) && (
          <button
            onClick={onOpenActiveScenarios}
            className="flex items-center gap-2 bg-red-600 hover:bg-red-500 px-2 py-1 rounded animate-pulse cursor-pointer transition-colors"
            title="לחץ להצגת אירועים פעילים"
          >
            <span>🚨</span>
            <span className="text-sm font-bold">
              {criticalEvents > 0 ? `${criticalEvents} קריטי` : 'תרחיש פעיל'}
            </span>
          </button>
        )}

        {warningEvents > 0 && (
          <div className="flex items-center gap-2 bg-yellow-600 px-2 py-1 rounded">
            <span>⚠️</span>
            <span className="text-sm">{warningEvents} אזהרות</span>
          </div>
        )}

        {/* AI Stats button - dev mode only */}
        {isDevMode && onOpenAIStats && (
          <button
            onClick={onOpenAIStats}
            className="flex items-center gap-1 bg-green-700 hover:bg-green-600 px-2 py-1 rounded text-sm transition-colors"
            title="סטטיסטיקות AI (Dev Mode)"
          >
            <span>📊</span>
            <span>AI Stats</span>
          </button>
        )}

        {/* Volume controls */}
        <div className="flex items-center gap-2 bg-gray-700 px-3 py-1 rounded">
          <button
            onClick={() => setSoundMuted(!soundMuted)}
            className={`text-lg hover:opacity-80 transition-opacity ${soundMuted ? 'text-red-400' : 'text-white'}`}
            title={soundMuted ? 'בטל השתקה' : 'השתק'}
          >
            {soundMuted || soundVolume === 0 ? '🔇' : soundVolume < 0.5 ? '🔉' : '🔊'}
          </button>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={soundMuted ? 0 : soundVolume}
            onChange={(e) => {
              const newVolume = parseFloat(e.target.value);
              setSoundVolume(newVolume);
              if (soundMuted && newVolume > 0) {
                setSoundMuted(false);
              }
            }}
            className="w-20 h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
            title={`עוצמה: ${Math.round(soundVolume * 100)}%`}
          />
          <span className="text-xs text-gray-400 w-8">
            {soundMuted ? 'OFF' : `${Math.round(soundVolume * 100)}%`}
          </span>
        </div>
      </div>

      {/* Date and time */}
      <div className="text-left">
        <div className="text-xl font-mono">{currentTime}</div>
        <div className="text-xs text-gray-400">{currentDate}</div>
      </div>
    </div>
  );
}
