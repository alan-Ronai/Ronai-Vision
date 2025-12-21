/**
 * DangerModeOverlay - Full-screen danger mode visual indicator
 *
 * Shows a pulsing red border and banner when the system is in emergency mode.
 */

import { useScenario } from '../../context/ScenarioContext';

export default function DangerModeOverlay({ children }) {
  const { dangerMode, scenario, config } = useScenario();

  if (!dangerMode) {
    return <>{children}</>;
  }

  return (
    <div className="relative min-h-screen">
      {/* Pulsing red border */}
      <div
        className="fixed inset-0 pointer-events-none z-40"
        style={{
          boxShadow: 'inset 0 0 0 4px rgba(239, 68, 68, 0.8)',
          animation: 'pulse-border 1s ease-in-out infinite',
        }}
      />

      {/* Red tint overlay */}
      <div
        className="fixed inset-0 pointer-events-none z-30"
        style={{
          background: 'rgba(127, 29, 29, 0.15)',
        }}
      />

      {/* Top danger banner */}
      <div
        className="fixed top-0 left-0 right-0 z-50 bg-red-600 text-white py-2 px-4 flex items-center justify-center gap-4"
        style={{ animation: 'pulse 2s ease-in-out infinite' }}
      >
        <span className="text-2xl animate-bounce">&#128680;</span>
        <span className="text-lg font-bold">
          {config?.ui?.dangerBannerText || 'חדירה ודאית - אירוע חירום פעיל'}
        </span>
        <span className="text-2xl animate-bounce">&#128680;</span>
      </div>

      {/* Stage indicator */}
      <div className="fixed top-12 right-4 z-50 bg-black/80 text-white px-3 py-1 rounded text-sm">
        שלב: {getStageLabel(scenario.stage)}
        {scenario.armedCount > 0 && (
          <span className="mr-2 text-red-400">
            | {scenario.armedCount} חמושים
          </span>
        )}
      </div>

      {/* Content with top padding for banner */}
      <div className="pt-12">
        {children}
      </div>

      <style>{`
        @keyframes pulse-border {
          0%, 100% {
            box-shadow: inset 0 0 0 4px rgba(239, 68, 68, 0.8);
          }
          50% {
            box-shadow: inset 0 0 0 6px rgba(239, 68, 68, 1);
          }
        }
      `}</style>
    </div>
  );
}

function getStageLabel(stage) {
  const labels = {
    idle: 'רגיל',
    vehicle_detected: 'רכב זוהה',
    vehicle_alert: 'התראת רכב',
    armed_persons_detected: 'זיהוי חמושים',
    emergency_mode: 'מצב חירום',
    response_initiated: 'תגובה הופעלה',
    drone_dispatched: 'רחפן הוקפץ',
    civilian_alert: 'התרעה למגורים',
    code_broadcast: 'קוד שודר',
    soldier_video: 'סרטון מלוחם',
    new_camera: 'מצלמה חדשה',
    situation_end: 'סיום אירוע',
  };
  return labels[stage] || stage;
}
