import { useApp } from '../context/AppContext';
import { useState, useEffect } from 'react';

const AI_SERVICE_URL = import.meta.env.VITE_AI_SERVICE_URL || 'http://localhost:8000';
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3000';

/**
 * Demo controls for testing the system
 * Can be hidden in production
 */
export default function DemoControls() {
  const { triggerSimulation, createDemoEvent, connected } = useApp();
  const [isOpen, setIsOpen] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [message, setMessage] = useState('');

  // Fetch current demo mode state
  useEffect(() => {
    const fetchDemoMode = async () => {
      try {
        const response = await fetch(`${AI_SERVICE_URL}/detection/stats`);
        if (response.ok) {
          const data = await response.json();
          // Check if FPS is low (demo mode)
          setDemoMode(data.detection_fps <= 10);
        }
      } catch (error) {
        console.error('Failed to fetch demo mode:', error);
      }
    };
    if (isOpen) fetchDemoMode();
  }, [isOpen]);

  // Toggle demo mode
  const toggleDemoMode = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `${AI_SERVICE_URL}/detection/config/demo-mode?enabled=${!demoMode}`,
        { method: 'POST' }
      );
      if (response.ok) {
        setDemoMode(!demoMode);
        setMessage(demoMode ? 'מצב רגיל הופעל' : 'מצב הדגמה הופעל');
        setTimeout(() => setMessage(''), 2000);
      }
    } catch (error) {
      console.error('Failed to toggle demo mode:', error);
      setMessage('שגיאה בהחלפת מצב');
    } finally {
      setLoading(false);
    }
  };

  // Scenario triggers
  const triggerScenario = async (type) => {
    setScenarioLoading(true);
    try {
      let endpoint = '';
      let body = {};

      switch (type) {
        case 'full':
          endpoint = '/api/scenario/demo/full-scenario';
          break;
        case 'vehicle':
          endpoint = '/api/scenario/demo/stolen-vehicle';
          break;
        case 'armed':
          endpoint = '/api/scenario/demo/armed-persons';
          body = { count: 3 };
          break;
        case 'drone':
          endpoint = '/api/scenario/demo/keyword';
          body = { keyword: 'drone' };
          break;
        case 'code':
          endpoint = '/api/scenario/demo/keyword';
          body = { keyword: 'code' };
          break;
        case 'end':
          endpoint = '/api/scenario/demo/keyword';
          body = { keyword: 'end' };
          break;
        case 'reset':
          endpoint = '/api/scenario/reset';
          break;
        default:
          return;
      }

      const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (response.ok) {
        const result = await response.json();
        setMessage(result.message || 'תרחיש הופעל בהצלחה');
        setTimeout(() => setMessage(''), 3000);
      } else {
        const error = await response.json();
        setMessage(error.error || 'שגיאה');
      }
    } catch (error) {
      console.error('Failed to trigger scenario:', error);
      setMessage('שגיאה בהפעלת תרחיש');
    } finally {
      setScenarioLoading(false);
    }
  };

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
    <div className="fixed bottom-4 left-4 bg-gray-800 border border-gray-600 rounded-lg p-4 z-30 w-72 max-h-[80vh] overflow-y-auto">
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

      {/* Message display */}
      {message && (
        <div className="mb-3 p-2 rounded text-sm bg-blue-900 text-blue-200">
          {message}
        </div>
      )}

      {/* Demo Mode Toggle */}
      <div className="mb-4 p-3 bg-gray-700/50 rounded">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">מצב הדגמה (FPS נמוך)</span>
          <button
            onClick={toggleDemoMode}
            disabled={loading}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              demoMode
                ? 'bg-green-600 hover:bg-green-700'
                : 'bg-gray-600 hover:bg-gray-500'
            }`}
          >
            {loading ? '...' : demoMode ? '🐌 פעיל' : '⚡ כבוי'}
          </button>
        </div>
        <p className="text-xs text-gray-400">
          מצב הדגמה מאט את עיבוד הווידאו כדי שהסרטון יימשך יותר זמן
        </p>
      </div>

      {/* Scenario Triggers */}
      <div className="mb-4">
        <p className="text-xs text-gray-400 mb-2 font-medium">🎬 תרחישים מוכנים:</p>
        <div className="space-y-2">
          <button
            onClick={() => triggerScenario('full')}
            disabled={scenarioLoading}
            className="w-full bg-red-700 hover:bg-red-600 px-3 py-2 rounded text-sm text-right disabled:opacity-50"
          >
            🚗💥 תרחיש מלא (רכב גנוב + חמושים)
          </button>
          <button
            onClick={() => triggerScenario('vehicle')}
            disabled={scenarioLoading}
            className="w-full bg-orange-700 hover:bg-orange-600 px-3 py-2 rounded text-sm text-right disabled:opacity-50"
          >
            🚗 זיהוי רכב גנוב
          </button>
          <button
            onClick={() => triggerScenario('armed')}
            disabled={scenarioLoading}
            className="w-full bg-yellow-700 hover:bg-yellow-600 px-3 py-2 rounded text-sm text-right disabled:opacity-50"
          >
            🔫 הוספת 3 חמושים
          </button>
        </div>
      </div>

      {/* Keyword Triggers */}
      <div className="mb-4">
        <p className="text-xs text-gray-400 mb-2 font-medium">🎤 מילות מפתח (רדיו):</p>
        <div className="space-y-2">
          <button
            onClick={() => triggerScenario('drone')}
            disabled={scenarioLoading}
            className="w-full bg-purple-700 hover:bg-purple-600 px-3 py-2 rounded text-sm text-right disabled:opacity-50"
          >
            🚁 "שלחו רחפן"
          </button>
          <button
            onClick={() => triggerScenario('code')}
            disabled={scenarioLoading}
            className="w-full bg-indigo-700 hover:bg-indigo-600 px-3 py-2 rounded text-sm text-right disabled:opacity-50"
          >
            📻 "קוד צפרדע"
          </button>
          <button
            onClick={() => triggerScenario('end')}
            disabled={scenarioLoading}
            className="w-full bg-green-700 hover:bg-green-600 px-3 py-2 rounded text-sm text-right disabled:opacity-50"
          >
            ✅ "חדל חדל חדל"
          </button>
          <button
            onClick={() => triggerScenario('reset')}
            disabled={scenarioLoading}
            className="w-full bg-gray-600 hover:bg-gray-500 px-3 py-2 rounded text-sm text-right disabled:opacity-50"
          >
            🔄 איפוס תרחיש
          </button>
        </div>
      </div>

      {/* Event triggers */}
      <div className="mb-4">
        <p className="text-xs text-gray-400 mb-2 font-medium">📋 יצירת אירועים:</p>
        <div className="grid grid-cols-3 gap-2">
          <button
            onClick={() => createDemoEvent('info')}
            className="bg-blue-600 hover:bg-blue-500 px-2 py-2 rounded text-xs"
          >
            📋 רגיל
          </button>
          <button
            onClick={() => createDemoEvent('warning')}
            className="bg-yellow-600 hover:bg-yellow-500 px-2 py-2 rounded text-xs"
          >
            ⚠️ אזהרה
          </button>
          <button
            onClick={() => createDemoEvent('critical')}
            className="bg-red-600 hover:bg-red-500 px-2 py-2 rounded text-xs"
          >
            🚨 קריטי
          </button>
        </div>
      </div>

      {/* Simulation triggers */}
      <div className="space-y-2">
        <p className="text-xs text-gray-400 font-medium">🎭 סימולציות UI:</p>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => triggerSimulation('drone_dispatch')}
            className="bg-gray-700 hover:bg-gray-600 px-2 py-2 rounded text-xs"
          >
            🚁 רחפן
          </button>
          <button
            onClick={() => triggerSimulation('phone_call')}
            className="bg-gray-700 hover:bg-gray-600 px-2 py-2 rounded text-xs"
          >
            📞 מפקד
          </button>
          <button
            onClick={() => triggerSimulation('pa_announcement')}
            className="bg-gray-700 hover:bg-gray-600 px-2 py-2 rounded text-xs"
          >
            📢 כריזה
          </button>
          <button
            onClick={() => triggerSimulation('code_broadcast')}
            className="bg-gray-700 hover:bg-gray-600 px-2 py-2 rounded text-xs"
          >
            📻 קוד
          </button>
        </div>
      </div>
    </div>
  );
}
