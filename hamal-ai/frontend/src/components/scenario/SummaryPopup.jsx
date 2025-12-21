/**
 * SummaryPopup - End-of-scenario summary
 *
 * Shows when the scenario ends with a summary of what happened.
 */

import { useScenario } from '../../context/ScenarioContext';

export default function SummaryPopup() {
  const { summaryPopup, closeSummary, config } = useScenario();

  if (!summaryPopup) {
    return null;
  }

  const formatDuration = (ms) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const getReasonText = (reason) => {
    switch (reason) {
      case 'false_alarm':
        return 'אזעקת שווא';
      case 'neutralized':
        return 'איום נוטרל';
      case 'manual':
        return 'סיום ידני';
      default:
        return 'הסתיים';
    }
  };

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 flex items-center justify-center">
      <div className="bg-gray-800 rounded-lg shadow-xl p-6 max-w-lg w-full mx-4">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="text-4xl mb-2">&#9989;</div>
          <h2 className="text-2xl font-bold text-white">
            {summaryPopup.title || config?.ui?.situationEndedTitle || 'אירוע הסתיים'}
          </h2>
          <div className="text-gray-400 mt-2">
            {getReasonText(summaryPopup.reason)}
          </div>
        </div>

        {/* Stats */}
        <div className="space-y-3 mb-6">
          {/* Duration */}
          <div className="bg-gray-900 rounded p-3 flex items-center justify-between">
            <span className="text-white font-medium">משך האירוע</span>
            <span className="text-blue-400 font-bold">
              {formatDuration(summaryPopup.duration)}
            </span>
          </div>

          {/* Vehicle */}
          {summaryPopup.vehicle && (
            <div className="bg-gray-900 rounded p-3">
              <div className="text-gray-400 text-sm mb-1">רכב חשוד</div>
              <div className="text-white">
                {summaryPopup.vehicle.licensePlate}
                {summaryPopup.vehicle.color && ` - ${summaryPopup.vehicle.color}`}
                {summaryPopup.vehicle.make && ` ${summaryPopup.vehicle.make}`}
              </div>
            </div>
          )}

          {/* Armed persons count */}
          {summaryPopup.armedCount > 0 && (
            <div className="bg-gray-900 rounded p-3 flex items-center justify-between">
              <span className="text-white font-medium">חמושים שזוהו</span>
              <span className="text-red-400 font-bold">
                {summaryPopup.armedCount}
              </span>
            </div>
          )}
        </div>

        {/* Recording note */}
        <div className="bg-blue-900/30 border border-blue-500/50 rounded p-3 mb-6 text-center">
          <div className="text-blue-300 text-sm">
            &#127909; הקלטה נשמרה
          </div>
        </div>

        {/* Close button */}
        <button
          onClick={closeSummary}
          className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 rounded"
        >
          {config?.ui?.closeButton || 'סגור'}
        </button>
      </div>
    </div>
  );
}
