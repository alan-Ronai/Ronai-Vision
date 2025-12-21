/**
 * ScenarioAlertPopup - Alert popup for scenario events
 *
 * Shows alerts like stolen vehicle detection.
 */

import { useScenario } from '../../context/ScenarioContext';

export default function ScenarioAlertPopup() {
  const { alertPopup, dismissAlert } = useScenario();

  // Don't show for vehicle alerts (VehicleAlertModal handles those)
  if (!alertPopup || alertPopup.alertType === 'vehicle') {
    return null;
  }

  const getTypeStyles = () => {
    switch (alertPopup.type) {
      case 'critical':
        return 'bg-red-900 border-red-500';
      case 'warning':
        return 'bg-yellow-900 border-yellow-500';
      case 'info':
        return 'bg-blue-900 border-blue-500';
      default:
        return 'bg-gray-900 border-gray-500';
    }
  };

  const getIcon = () => {
    switch (alertPopup.type) {
      case 'critical':
        return '&#128680;'; // Alarm
      case 'warning':
        return '&#9888;'; // Warning
      case 'info':
        return '&#8505;'; // Info
      default:
        return '&#128276;'; // Bell
    }
  };

  return (
    <div className="fixed top-20 right-4 z-[90] max-w-md animate-slide-in-right">
      <div className={`${getTypeStyles()} border-2 rounded-lg shadow-xl p-4`}>
        {/* Header */}
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <span dangerouslySetInnerHTML={{ __html: getIcon() }} />
            {alertPopup.title}
          </h3>
          <button
            onClick={dismissAlert}
            className="text-gray-400 hover:text-white text-xl"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="text-white whitespace-pre-line text-right">
          {alertPopup.content}
        </div>

        {/* Timestamp */}
        <div className="text-xs text-gray-400 mt-2 text-left">
          {new Date().toLocaleTimeString('he-IL')}
        </div>
      </div>

      <style>{`
        @keyframes slide-in-right {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        .animate-slide-in-right {
          animation: slide-in-right 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}
