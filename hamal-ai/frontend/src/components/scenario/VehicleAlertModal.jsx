/**
 * VehicleAlertModal - Full-screen alert for stolen vehicle detection
 *
 * Shows a prominent full-screen modal when a stolen vehicle is detected,
 * similar to the emergency modal but for the initial vehicle alert stage.
 */

import { useScenario } from '../../context/ScenarioContext';

export default function VehicleAlertModal() {
  const { alertPopup, dismissAlert, config } = useScenario();

  // Only show for vehicle-type alerts
  if (!alertPopup || alertPopup.alertType !== 'vehicle') {
    return null;
  }

  const vehicle = alertPopup.vehicle || {};

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90">
      {/* Main alert box */}
      <div
        className="bg-yellow-900 border-4 border-yellow-500 rounded-lg p-8 max-w-2xl w-full mx-4 text-center"
        style={{ animation: 'pulse-yellow 1.5s ease-in-out infinite' }}
      >
        {/* Title */}
        <h1 className="text-5xl font-bold text-white mb-4 flex items-center justify-center gap-4">
          <span className="animate-bounce">🚨</span>
          <span>{alertPopup.title || 'רכב גנוב זוהה!'}</span>
          <span className="animate-bounce">🚨</span>
        </h1>

        {/* Subtitle */}
        <h2 className="text-2xl text-yellow-200 mb-6">
          נדרשת תשומת לב מיידית
        </h2>

        {/* Vehicle info */}
        <div className="bg-black/50 rounded p-4 mb-6 text-right">
          <h3 className="text-lg font-bold text-yellow-400 mb-3">פרטי הרכב:</h3>
          <div className="text-white space-y-2 text-lg">
            <p className="flex justify-between">
              <span className="text-gray-400">לוחית רישוי:</span>
              <span className="font-bold text-yellow-300">{vehicle.licensePlate || 'לא ידוע'}</span>
            </p>
            {vehicle.color && (
              <p className="flex justify-between">
                <span className="text-gray-400">צבע:</span>
                <span>{vehicle.color}</span>
              </p>
            )}
            {(vehicle.make || vehicle.model) && (
              <p className="flex justify-between">
                <span className="text-gray-400">יצרן/דגם:</span>
                <span>{vehicle.make} {vehicle.model}</span>
              </p>
            )}
            {vehicle.cameraId && (
              <p className="flex justify-between">
                <span className="text-gray-400">מצלמה:</span>
                <span>{vehicle.cameraId}</span>
              </p>
            )}
          </div>
        </div>

        {/* Status indicator */}
        <div className="bg-yellow-800/50 rounded p-3 mb-6">
          <div className="text-yellow-200 flex items-center justify-center gap-2">
            <span className="w-3 h-3 bg-yellow-400 rounded-full animate-pulse" />
            <span>ממתין לזיהוי חמושים...</span>
          </div>
        </div>

        {/* Info text */}
        <div className="text-sm text-gray-300 mb-6">
          המערכת עוקבת אחר הרכב ומחפשת איומים נוספים
        </div>

        {/* Action button */}
        <button
          onClick={dismissAlert}
          className="bg-yellow-600 hover:bg-yellow-700 text-white font-bold py-4 px-8 rounded-lg text-xl transition-colors"
        >
          {config?.ui?.understood || 'הבנתי'}
        </button>
      </div>

      <style>{`
        @keyframes pulse-yellow {
          0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(234, 179, 8, 0.7);
          }
          50% {
            transform: scale(1.02);
            box-shadow: 0 0 30px 10px rgba(234, 179, 8, 0.5);
          }
        }
      `}</style>
    </div>
  );
}
