/**
 * EmergencyModal - Full-screen emergency alert with action buttons
 *
 * Shows when armed persons threshold is reached.
 * User must acknowledge or declare false alarm.
 */

import { useScenario } from '../../context/ScenarioContext';

export default function EmergencyModal() {
  const {
    emergencyModal,
    acknowledgeEmergency,
    declareFalseAlarm,
    config,
  } = useScenario();

  if (!emergencyModal) {
    return null;
  }

  const handleAcknowledge = async () => {
    await acknowledgeEmergency();
  };

  const handleFalseAlarm = async () => {
    await declareFalseAlarm();
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90">
      {/* Main alert box */}
      <div
        className="bg-red-900 border-4 border-red-500 rounded-lg p-8 max-w-2xl w-full mx-4 text-center"
        style={{ animation: 'pulse 1.5s ease-in-out infinite' }}
      >
        {/* Title */}
        <h1 className="text-5xl font-bold text-white mb-4 flex items-center justify-center gap-4">
          <span className="animate-bounce">&#128680;</span>
          <span>{emergencyModal.title || config?.ui?.emergencyTitle || 'חדירה ודאית!'}</span>
          <span className="animate-bounce">&#128680;</span>
        </h1>

        {/* Subtitle */}
        <h2 className="text-2xl text-red-200 mb-6">
          {emergencyModal.subtitle}
        </h2>

        {/* Vehicle info */}
        {emergencyModal.vehicle && (
          <div className="bg-black/50 rounded p-4 mb-4 text-right">
            <h3 className="text-lg font-bold text-yellow-400 mb-2">רכב חשוד:</h3>
            <div className="text-white space-y-1">
              <p><strong>לוחית רישוי:</strong> {emergencyModal.vehicle.licensePlate}</p>
              {emergencyModal.vehicle.color && (
                <p><strong>צבע:</strong> {emergencyModal.vehicle.color}</p>
              )}
              {emergencyModal.vehicle.make && (
                <p><strong>יצרן:</strong> {emergencyModal.vehicle.make} {emergencyModal.vehicle.model || ''}</p>
              )}
            </div>
          </div>
        )}

        {/* Persons info */}
        {emergencyModal.persons && emergencyModal.persons.length > 0 && (
          <div className="bg-black/50 rounded p-4 mb-6 text-right">
            <h3 className="text-lg font-bold text-red-400 mb-2">חמושים שזוהו:</h3>
            <div className="space-y-2">
              {emergencyModal.persons.map((person, i) => (
                <div key={person.trackId || i} className="text-white flex items-center gap-2">
                  <span className="text-red-400">#{i + 1}</span>
                  {person.clothing && <span>לבוש: {person.clothing}</span>}
                  {person.weaponType && <span className="text-red-300">| נשק: {person.weaponType}</span>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Auto-triggered responses notice */}
        <div className="text-sm text-gray-300 mb-6">
          &#128222; חיוג למפקד תורן | &#128681; הקפצת רחפן | &#128266; כריזה למגורים
        </div>

        {/* Action buttons */}
        <div className="flex gap-4 justify-center">
          <button
            onClick={handleAcknowledge}
            className="bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-8 rounded-lg text-xl transition-colors"
          >
            {config?.ui?.handlingIt || 'מטפל בזה'}
          </button>
          <button
            onClick={handleFalseAlarm}
            className="bg-gray-600 hover:bg-gray-700 text-white font-bold py-4 px-8 rounded-lg text-xl transition-colors"
          >
            {config?.ui?.falseAlarm || 'אזעקת שווא'}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
          }
          50% {
            transform: scale(1.02);
            box-shadow: 0 0 30px 10px rgba(239, 68, 68, 0.5);
          }
        }
      `}</style>
    </div>
  );
}
