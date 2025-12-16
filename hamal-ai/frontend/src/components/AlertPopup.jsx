import { useApp } from '../context/AppContext';

export default function AlertPopup() {
  const { emergencyData, acknowledgeEmergency, endEmergency } = useApp();

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-red-900 border-4 border-red-500 rounded-xl p-8 max-w-xl w-full animate-pulse shadow-2xl">
        {/* Header */}
        <h1 className="text-4xl font-bold text-center mb-6 flex items-center justify-center gap-3">
          <span className="animate-bounce">🚨</span>
          <span>אירוע חירום</span>
          <span className="animate-bounce">🚨</span>
        </h1>

        {/* Event title */}
        <h2 className="text-2xl text-center mb-6 text-red-200">
          {emergencyData?.title || 'חדירה ודאית'}
        </h2>

        {/* Details */}
        {emergencyData?.details && (
          <div className="bg-black/40 rounded-lg p-4 mb-6 text-lg space-y-3">
            {/* People count */}
            {emergencyData.details.people && (
              <div className="flex items-center gap-3">
                <span className="text-2xl">👥</span>
                <span>מספר אנשים: {emergencyData.details.people.count}</span>
              </div>
            )}

            {/* Armed status */}
            {emergencyData.details.people?.armed && (
              <div className="flex items-center gap-3 text-yellow-300 font-bold">
                <span className="text-2xl">⚠️</span>
                <span>חמושים: כן</span>
              </div>
            )}

            {/* Vehicle info */}
            {emergencyData.details.vehicle && (
              <div className="flex items-center gap-3">
                <span className="text-2xl">🚗</span>
                <span>
                  רכב: {emergencyData.details.vehicle.color} {emergencyData.details.vehicle.type}
                </span>
              </div>
            )}

            {/* License plate */}
            {emergencyData.details.vehicle?.number && (
              <div className="flex items-center gap-3">
                <span className="text-2xl">🔢</span>
                <span>מספר רכב: {emergencyData.details.vehicle.number}</span>
              </div>
            )}

            {/* Weapons */}
            {emergencyData.details.weapons && emergencyData.details.weapons.length > 0 && (
              <div className="flex items-center gap-3 text-red-300 font-bold">
                <span className="text-2xl">🔫</span>
                <span>
                  נשק: {emergencyData.details.weapons.map(w => w.type).join(', ')}
                </span>
              </div>
            )}

            {/* Camera */}
            {emergencyData.cameraId && (
              <div className="flex items-center gap-3 text-gray-300">
                <span className="text-2xl">📹</span>
                <span>מצלמה: {emergencyData.cameraId}</span>
              </div>
            )}

            {/* Time */}
            {emergencyData.timestamp && (
              <div className="flex items-center gap-3 text-gray-300 text-sm">
                <span className="text-2xl">🕐</span>
                <span>
                  זמן: {new Date(emergencyData.timestamp).toLocaleTimeString('he-IL')}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="space-y-3">
          <button
            onClick={acknowledgeEmergency}
            className="w-full bg-white text-red-900 font-bold py-4 rounded-lg text-xl hover:bg-gray-200 transition flex items-center justify-center gap-2"
          >
            <span>✓</span>
            <span>אני מודע לאירוע - החזר מסך</span>
          </button>

          <button
            onClick={endEmergency}
            className="w-full bg-green-600 text-white font-bold py-3 rounded-lg text-lg hover:bg-green-500 transition flex items-center justify-center gap-2"
          >
            <span>🏁</span>
            <span>חדל - סיים אירוע</span>
          </button>
        </div>

        {/* Auto-triggered actions notice */}
        <div className="mt-6 pt-4 border-t border-red-700 text-sm text-red-300 text-center">
          <p>פעולות אוטומטיות הופעלו:</p>
          <p className="mt-1">📞 חיוג למפקד תורן | 🚁 הקפצת רחפן</p>
        </div>
      </div>
    </div>
  );
}
