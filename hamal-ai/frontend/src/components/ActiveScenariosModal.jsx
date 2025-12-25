import { useScenario } from '../context/ScenarioContext';
import { useApp } from '../context/AppContext';

export default function ActiveScenariosModal({ isOpen, onClose }) {
  const { scenario } = useScenario();
  const { events, acknowledgeEvent, API_URL } = useApp();

  if (!isOpen) return null;

  // Get unacknowledged critical events
  const criticalEvents = events.filter(e => e.severity === 'critical' && !e.acknowledged);

  const endScenario = async (reason) => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason })
      });
      if (!response.ok) {
        console.error('Failed to end scenario:', await response.text());
      }
    } catch (error) {
      console.error('Error ending scenario:', error);
    }
  };

  const handleDismissAsfalseAlarm = async () => {
    if (scenario?.active) {
      await endScenario('false_alarm');
    }
    onClose();
  };

  const handleNeutralized = async () => {
    if (scenario?.active) {
      await endScenario('neutralized');
    }
    onClose();
  };

  const handleAcknowledgeEvent = (eventId) => {
    if (acknowledgeEvent) {
      acknowledgeEvent(eventId);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg shadow-2xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="bg-red-900 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span className="text-2xl">🚨</span>
            אירועים פעילים
          </h2>
          <button
            onClick={onClose}
            className="text-gray-300 hover:text-white text-2xl"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          {/* Active Scenario */}
          {scenario?.active && (
            <div className="bg-gray-700/50 rounded-lg p-4 mb-4 border-l-4 border-red-500">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-bold text-red-400">תרחיש פעיל</h3>
                <span className="px-2 py-1 bg-red-600 rounded text-xs">
                  {scenario.variant === 'stolen-vehicle' ? 'רכב גנוב' : 'אדם חמוש'}
                </span>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">שלב נוכחי:</span>
                  <span className="text-white">{scenario.stage || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">מזהה:</span>
                  <span className="text-white font-mono text-xs">{scenario.scenarioId || 'N/A'}</span>
                </div>
                {scenario.vehicle && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">רכב:</span>
                    <span className="text-white">{scenario.vehicle.licensePlate}</span>
                  </div>
                )}
                {scenario.armedCount > 0 && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">חמושים:</span>
                    <span className="text-red-400 font-bold">{scenario.armedCount}</span>
                  </div>
                )}
              </div>

              {/* Action buttons */}
              <div className="flex gap-3 mt-4">
                <button
                  onClick={handleDismissAsfalseAlarm}
                  className="flex-1 bg-yellow-600 hover:bg-yellow-500 px-4 py-2 rounded font-bold transition-colors"
                >
                  ✋ אזעקת שווא
                </button>
                <button
                  onClick={handleNeutralized}
                  className="flex-1 bg-green-600 hover:bg-green-500 px-4 py-2 rounded font-bold transition-colors"
                >
                  ✅ נוטרל
                </button>
              </div>
            </div>
          )}

          {/* Critical Events */}
          {criticalEvents.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-lg font-bold text-yellow-400 mb-2">אירועים קריטיים</h3>
              {criticalEvents.map((event, idx) => (
                <div
                  key={event.id || idx}
                  className="bg-gray-700/30 rounded-lg p-3 border-l-4 border-yellow-500"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="font-bold text-white">{event.title || event.type}</p>
                      <p className="text-sm text-gray-400">{event.description}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        {new Date(event.timestamp).toLocaleTimeString('he-IL')}
                      </p>
                    </div>
                    <button
                      onClick={() => handleAcknowledgeEvent(event.id)}
                      className="text-xs bg-gray-600 hover:bg-gray-500 px-2 py-1 rounded"
                    >
                      סמן כנקרא
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* No events message */}
          {!scenario?.active && criticalEvents.length === 0 && (
            <div className="text-center py-8 text-gray-400">
              <span className="text-4xl block mb-2">✅</span>
              אין אירועים פעילים כרגע
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-900 px-6 py-3 flex justify-end">
          <button
            onClick={onClose}
            className="bg-gray-600 hover:bg-gray-500 px-6 py-2 rounded font-bold transition-colors"
          >
            סגור
          </button>
        </div>
      </div>
    </div>
  );
}
