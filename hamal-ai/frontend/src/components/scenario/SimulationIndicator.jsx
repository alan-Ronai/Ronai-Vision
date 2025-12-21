/**
 * SimulationIndicator - Shows current simulation status
 *
 * Displays when simulations like drone dispatch, phone call, etc. are active.
 */

import { useScenario } from '../../context/ScenarioContext';

export default function SimulationIndicator() {
  const { simulation, ttsMessage } = useScenario();

  if (!simulation && !ttsMessage) {
    return null;
  }

  const getSimulationConfig = () => {
    if (!simulation) return null;

    const configs = {
      phone_call: {
        icon: '&#128222;',
        color: 'bg-blue-600',
        animation: 'animate-pulse',
      },
      phone_connected: {
        icon: '&#9989;',
        color: 'bg-green-600',
        animation: '',
      },
      drone_dispatch: {
        icon: '&#128681;',
        color: 'bg-purple-600',
        animation: 'animate-bounce',
      },
      drone_enroute: {
        icon: '&#128681;',
        color: 'bg-purple-500',
        animation: 'animate-pulse',
      },
      pa_announcement: {
        icon: '&#128266;',
        color: 'bg-orange-600',
        animation: 'animate-pulse',
      },
      code_broadcast: {
        icon: '&#128251;',
        color: 'bg-red-600',
        animation: 'animate-ping',
      },
    };

    return configs[simulation.type] || {
      icon: '&#8505;',
      color: 'bg-gray-600',
      animation: '',
    };
  };

  const config = getSimulationConfig();

  return (
    <div className="fixed bottom-20 right-4 z-[80] space-y-2">
      {/* Simulation indicator */}
      {simulation && config && (
        <div
          className={`${config.color} text-white px-4 py-3 rounded-lg shadow-xl flex items-center gap-3 ${config.animation}`}
        >
          <span
            className="text-2xl"
            dangerouslySetInnerHTML={{ __html: config.icon }}
          />
          <span className="text-lg font-semibold">
            {simulation.title}
          </span>
        </div>
      )}

      {/* TTS message indicator */}
      {ttsMessage && (
        <div className="bg-indigo-600 text-white px-4 py-3 rounded-lg shadow-xl flex items-center gap-3 animate-pulse">
          <span className="text-2xl">&#128266;</span>
          <div>
            <div className="text-sm text-indigo-200">משדר לרדיו:</div>
            <div className="font-semibold text-right" dir="rtl">
              {ttsMessage.length > 50 ? `${ttsMessage.substring(0, 50)}...` : ttsMessage}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
