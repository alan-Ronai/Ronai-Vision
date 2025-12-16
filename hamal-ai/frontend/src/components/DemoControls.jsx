import { useApp } from '../context/AppContext';
import { useState } from 'react';

/**
 * Demo controls for testing the system
 * Can be hidden in production
 */
export default function DemoControls() {
  const { triggerSimulation, createDemoEvent, connected } = useApp();
  const [isOpen, setIsOpen] = useState(false);

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 left-4 bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded-lg text-sm z-30 flex items-center gap-2"
      >
        <span>🎮</span>
        <span>דמו</span>
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 left-4 bg-gray-800 border border-gray-600 rounded-lg p-4 z-30 w-64">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-bold flex items-center gap-2">
          <span>🎮</span>
          <span>פקדי הדגמה</span>
        </h3>
        <button
          onClick={() => setIsOpen(false)}
          className="text-gray-400 hover:text-white"
        >
          ✕
        </button>
      </div>

      {/* Connection status */}
      <div className={`mb-3 p-2 rounded text-sm ${connected ? 'bg-green-900' : 'bg-red-900'}`}>
        {connected ? '🟢 מחובר לשרת' : '🔴 לא מחובר'}
      </div>

      {/* Event triggers */}
      <div className="space-y-2 mb-4">
        <p className="text-xs text-gray-400">יצירת אירועים:</p>
        <button
          onClick={() => createDemoEvent('info')}
          className="w-full bg-blue-600 hover:bg-blue-500 px-3 py-2 rounded text-sm"
        >
          📋 אירוע רגיל
        </button>
        <button
          onClick={() => createDemoEvent('warning')}
          className="w-full bg-yellow-600 hover:bg-yellow-500 px-3 py-2 rounded text-sm"
        >
          ⚠️ אזהרה
        </button>
        <button
          onClick={() => createDemoEvent('critical')}
          className="w-full bg-red-600 hover:bg-red-500 px-3 py-2 rounded text-sm"
        >
          🚨 אירוע קריטי (חירום)
        </button>
      </div>

      {/* Simulation triggers */}
      <div className="space-y-2">
        <p className="text-xs text-gray-400">סימולציות:</p>
        <button
          onClick={() => triggerSimulation('drone_dispatch')}
          className="w-full bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm text-right"
        >
          🚁 הקפצת רחפן
        </button>
        <button
          onClick={() => triggerSimulation('phone_call')}
          className="w-full bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm text-right"
        >
          📞 חיוג למפקד
        </button>
        <button
          onClick={() => triggerSimulation('pa_announcement')}
          className="w-full bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm text-right"
        >
          📢 כריזה
        </button>
        <button
          onClick={() => triggerSimulation('code_broadcast')}
          className="w-full bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm text-right"
        >
          📻 שידור קוד צפרדע
        </button>
        <button
          onClick={() => triggerSimulation('threat_neutralized')}
          className="w-full bg-green-700 hover:bg-green-600 px-3 py-2 rounded text-sm text-right"
        >
          ✅ חדל - סוף אירוע
        </button>
      </div>
    </div>
  );
}
