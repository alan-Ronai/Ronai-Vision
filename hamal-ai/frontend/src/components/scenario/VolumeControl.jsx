/**
 * VolumeControl - System sound volume slider with mute button
 *
 * A compact floating control for adjusting system sound volume.
 */

import { useState } from 'react';
import { useScenario } from '../../context/ScenarioContext';

export default function VolumeControl() {
  const { soundVolume, setSoundVolume, soundMuted, setSoundMuted } = useScenario();
  const [expanded, setExpanded] = useState(false);

  const handleVolumeChange = (e) => {
    const newVolume = parseFloat(e.target.value);
    setSoundVolume(newVolume);
    // If adjusting volume and was muted, unmute
    if (soundMuted && newVolume > 0) {
      setSoundMuted(false);
    }
  };

  const toggleMute = () => {
    setSoundMuted(!soundMuted);
  };

  const getVolumeIcon = () => {
    if (soundMuted || soundVolume === 0) {
      return '🔇'; // Muted
    } else if (soundVolume < 0.3) {
      return '🔈'; // Low
    } else if (soundVolume < 0.7) {
      return '🔉'; // Medium
    } else {
      return '🔊'; // High
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-40">
      {/* Expanded panel */}
      {expanded && (
        <div className="bg-gray-800 rounded-lg shadow-xl p-3 mb-2 w-48">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-white">עוצמת צלילים</span>
            <button
              onClick={toggleMute}
              className={`p-1 rounded ${soundMuted ? 'bg-red-600' : 'bg-gray-700'} hover:bg-gray-600`}
              title={soundMuted ? 'בטל השתקה' : 'השתק'}
            >
              <span className="text-lg">{getVolumeIcon()}</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">🔈</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={soundMuted ? 0 : soundVolume}
              onChange={handleVolumeChange}
              className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(soundMuted ? 0 : soundVolume) * 100}%, #374151 ${(soundMuted ? 0 : soundVolume) * 100}%, #374151 100%)`
              }}
            />
            <span className="text-xs text-gray-400">🔊</span>
          </div>

          <div className="text-center text-xs text-gray-500 mt-2">
            {soundMuted ? 'מושתק' : `${Math.round(soundVolume * 100)}%`}
          </div>
        </div>
      )}

      {/* Toggle button */}
      <button
        onClick={() => setExpanded(!expanded)}
        className={`
          w-12 h-12 rounded-full shadow-lg flex items-center justify-center text-xl
          transition-all duration-200
          ${soundMuted ? 'bg-red-600 hover:bg-red-500' : 'bg-gray-700 hover:bg-gray-600'}
        `}
        title="הגדרות צלילים"
      >
        {getVolumeIcon()}
      </button>
    </div>
  );
}
