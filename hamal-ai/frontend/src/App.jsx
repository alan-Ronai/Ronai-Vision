import { useState, useRef } from 'react';
import { AppProvider, useApp } from './context/AppContext';
import { ScenarioProvider, useScenario } from './context/ScenarioContext';
import CameraGrid from './components/CameraGrid';
import MainCamera from './components/MainCamera';
import EventLog from './components/EventLog';
import RadioTranscript from './components/RadioTranscript';
import AlertPopup from './components/AlertPopup';
import SimulationPopup from './components/SimulationPopup';
import StatusBar from './components/StatusBar';
import DemoControls from './components/DemoControls';
import CameraManager from './components/CameraManager';
import DetectionSettings from './components/DetectionSettings';
import GlobalIDStore from './components/GlobalIDStore';
import AudioTransmitter from './components/AudioTransmitter';
import EventRuleManager from './components/EventRuleManager';
import AIStatsPanel from './components/AIStatsPanel';

// Scenario components
import {
  DangerModeOverlay,
  EmergencyModal,
  VehicleAlertModal,
  ScenarioAlertPopup,
  SimulationIndicator,
  SoldierVideoPanel,
  NewCameraDialog,
  SummaryPopup,
} from './components/scenario';

function Dashboard() {
  const { isEmergency, simulationPopup, loading } = useApp();
  const { dangerMode } = useScenario();
  const [showCameraManager, setShowCameraManager] = useState(false);
  const [showDetectionSettings, setShowDetectionSettings] = useState(false);
  const [showGlobalIDStore, setShowGlobalIDStore] = useState(false);
  const [showAudioTransmitter, setShowAudioTransmitter] = useState(false);
  const [showEventRules, setShowEventRules] = useState(false);
  const [showAIStats, setShowAIStats] = useState(false);

  if (loading) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-white text-xl">טוען מערכת חמ"ל...</p>
        </div>
      </div>
    );
  }

  return (
    <div
      dir="rtl"
      className={`
        h-screen w-screen bg-gray-900 text-white overflow-hidden flex flex-col
        ${isEmergency ? 'border-4 border-red-500 emergency-glow' : ''}
      `}
      style={{
        backgroundColor: isEmergency ? 'rgba(127, 29, 29, 0.3)' : undefined
      }}
    >
      {/* Emergency banner */}
      {isEmergency && (
        <div className="bg-red-600 text-white text-center py-2 text-2xl font-bold animate-pulse z-40 flex-shrink-0">
          <span className="mx-2">🚨</span>
          חדירה ודאית - אירוע חירום פעיל
          <span className="mx-2">🚨</span>
        </div>
      )}

      {/* Status bar */}
      <StatusBar onOpenAIStats={() => setShowAIStats(true)} />

      {/* Main content */}
      <div className={`flex-1 p-3 grid grid-cols-12 gap-3 min-h-0`}>
        {/* Main camera - 8 columns */}
        <div className="col-span-8 row-span-2 min-h-0">
          <MainCamera />
        </div>

        {/* Camera thumbnails - 4 columns */}
        <div className="col-span-4 row-span-1 min-h-0">
          <CameraGrid />
        </div>

        {/* Event log - 4 columns */}
        <div className="col-span-4 row-span-1 min-h-0">
          <EventLog />
        </div>

        {/* Radio transcript - full width bottom */}
        <div className="col-span-12 row-span-1 min-h-0">
          <RadioTranscript />
        </div>
      </div>

      {/* Demo controls (for testing) */}
      <DemoControls />

      {/* Settings buttons */}
      <div className="fixed bottom-4 left-4 flex flex-col gap-2 z-30">
        <button
          onClick={() => setShowDetectionSettings(true)}
          className="bg-blue-700 hover:bg-blue-600 px-4 py-2 rounded-lg shadow-lg flex items-center gap-2"
        >
          <span>⚙️</span>
          <span>הגדרות זיהוי AI</span>
        </button>
        <button
          onClick={() => setShowGlobalIDStore(true)}
          className="bg-purple-700 hover:bg-purple-600 px-4 py-2 rounded-lg shadow-lg flex items-center gap-2"
        >
          <span>🆔</span>
          <span>מאגר זיהויים</span>
        </button>
        <button
          onClick={() => setShowAudioTransmitter(true)}
          className="bg-orange-700 hover:bg-orange-600 px-4 py-2 rounded-lg shadow-lg flex items-center gap-2"
        >
          <span>📻</span>
          <span>שידור לקשר</span>
        </button>
        <button
          onClick={() => setShowEventRules(true)}
          className="bg-yellow-700 hover:bg-yellow-600 px-4 py-2 rounded-lg shadow-lg flex items-center gap-2"
        >
          <span>⚡</span>
          <span>חוקי אירועים</span>
        </button>
      </div>

      {/* Camera Manager Modal */}
      <CameraManager
        isOpen={showCameraManager}
        onClose={() => setShowCameraManager(false)}
      />

      {/* Detection Settings Modal */}
      <DetectionSettings
        isOpen={showDetectionSettings}
        onClose={() => setShowDetectionSettings(false)}
      />

      {/* Global ID Store Modal */}
      <GlobalIDStore
        isOpen={showGlobalIDStore}
        onClose={() => setShowGlobalIDStore(false)}
      />

      {/* Audio Transmitter Modal */}
      <AudioTransmitter
        isOpen={showAudioTransmitter}
        onClose={() => setShowAudioTransmitter(false)}
      />

      {/* Event Rule Manager Modal */}
      <EventRuleManager
        isOpen={showEventRules}
        onClose={() => setShowEventRules(false)}
      />

      {/* AI Stats Panel Modal */}
      <AIStatsPanel
        isOpen={showAIStats}
        onClose={() => setShowAIStats(false)}
      />

      {/* Legacy emergency popup (for non-scenario emergencies) */}
      {isEmergency && !dangerMode && <AlertPopup />}

      {/* Simulation popup (drone dispatch, phone call, etc.) */}
      {simulationPopup && <SimulationPopup />}

      {/* Scenario components */}
      <EmergencyModal />
      <VehicleAlertModal />
      <ScenarioAlertPopup />
      <SimulationIndicator />
      <SoldierVideoPanel />
      <NewCameraDialog />
      <SummaryPopup />
    </div>
  );
}

// Scenario Test Controls - Demo buttons for testing the scenario
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3000';

function ScenarioTestControls() {
  const {
    scenario,
    testStartScenario,
    testAddArmedPerson,
    testAdvanceStage,
    testTranscription,
    resetScenario,
    STAGES,
  } = useScenario();

  const [showControls, setShowControls] = useState(false);
  const [realDataLoading, setRealDataLoading] = useState(false);
  const [realDataMessage, setRealDataMessage] = useState('');
  const [soldierVideoUploading, setSoldierVideoUploading] = useState(false);
  const soldierVideoInputRef = useRef(null);

  // Trigger real data scenario
  const triggerRealScenario = async (type) => {
    setRealDataLoading(true);
    setRealDataMessage('');
    try {
      let endpoint = '';
      let body = {};

      switch (type) {
        case 'real-full':
          endpoint = '/api/scenario/demo/real/full-scenario';
          break;
        case 'real-vehicle':
          endpoint = '/api/scenario/demo/real/stolen-vehicle';
          break;
        case 'real-armed':
          endpoint = '/api/scenario/demo/real/armed-persons';
          body = { count: 3 };
          break;
        default:
          return;
      }

      const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      const result = await response.json();
      if (response.ok) {
        setRealDataMessage(result.message || 'הופעל בהצלחה!');
      } else {
        setRealDataMessage(result.error || 'שגיאה');
      }
      setTimeout(() => setRealDataMessage(''), 3000);
    } catch (error) {
      console.error('Real scenario error:', error);
      setRealDataMessage('שגיאה בחיבור');
    } finally {
      setRealDataLoading(false);
    }
  };

  // Handle soldier video upload
  const handleSoldierVideoUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.mp4')) {
      setRealDataMessage('יש להעלות קובץ MP4 בלבד');
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      setRealDataMessage('הקובץ גדול מדי (מקס 100MB)');
      return;
    }

    setSoldierVideoUploading(true);
    setRealDataMessage('מעלה סרטון...');

    try {
      const formData = new FormData();
      formData.append('video', file);

      const response = await fetch(`${BACKEND_URL}/api/scenario/soldier-video`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setRealDataMessage('סרטון נשלח, מתמלל...');
        setTimeout(() => setRealDataMessage(''), 5000);
      } else {
        setRealDataMessage(result.error || 'שגיאה');
      }
    } catch (error) {
      console.error('Soldier video upload error:', error);
      setRealDataMessage('שגיאה בהעלאה');
    } finally {
      setSoldierVideoUploading(false);
      if (soldierVideoInputRef.current) {
        soldierVideoInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <button
        onClick={() => setShowControls(!showControls)}
        className="bg-red-700 hover:bg-red-600 px-4 py-2 rounded-lg shadow-lg mb-2"
      >
        {showControls ? 'סגור' : 'תרחיש הדגמה'}
      </button>

      {showControls && (
        <div className="bg-gray-800 rounded-lg shadow-xl p-4 w-80 space-y-2 max-h-[80vh] overflow-y-auto">
          <h3 className="text-lg font-bold text-white mb-3">בקרת תרחיש</h3>

          {/* Current stage */}
          <div className="bg-gray-900 rounded p-2 text-sm">
            <span className="text-gray-400">שלב נוכחי: </span>
            <span className="text-blue-400">{scenario.stage}</span>
            {scenario.armedCount > 0 && (
              <span className="text-red-400 mr-2">| חמושים: {scenario.armedCount}</span>
            )}
          </div>

          {/* Real Data Message */}
          {realDataMessage && (
            <div className="bg-blue-900 text-blue-200 p-2 rounded text-sm">
              {realDataMessage}
            </div>
          )}

          {/* REAL DATA BUTTONS - NEW! */}
          <div className="border-2 border-green-500 rounded-lg p-3 bg-green-900/20">
            <h4 className="text-green-400 font-bold mb-2 text-sm">🎯 נתונים אמיתיים מהסצנה:</h4>
            <div className="space-y-2">
              <button
                onClick={() => triggerRealScenario('real-full')}
                disabled={realDataLoading}
                className="w-full bg-green-600 hover:bg-green-500 disabled:bg-gray-600 px-3 py-2 rounded text-sm font-bold"
              >
                🎯 תרחיש מלא (נתונים אמיתיים)
              </button>
              <button
                onClick={() => triggerRealScenario('real-vehicle')}
                disabled={realDataLoading}
                className="w-full bg-green-700 hover:bg-green-600 disabled:bg-gray-600 px-3 py-2 rounded text-sm"
              >
                🚗 רכב גנוב מהסצנה
              </button>
              <button
                onClick={() => triggerRealScenario('real-armed')}
                disabled={realDataLoading}
                className="w-full bg-green-700 hover:bg-green-600 disabled:bg-gray-600 px-3 py-2 rounded text-sm"
              >
                🔫 סמן אנשים כחמושים
              </button>
            </div>
          </div>

          {/* Soldier Video Upload */}
          <div className="border-2 border-blue-500 rounded-lg p-3 bg-blue-900/20">
            <h4 className="text-blue-400 font-bold mb-2 text-sm">📹 סרטון מלוחם:</h4>
            <input
              ref={soldierVideoInputRef}
              type="file"
              accept=".mp4,video/mp4"
              onChange={handleSoldierVideoUpload}
              className="hidden"
            />
            <button
              onClick={() => soldierVideoInputRef.current?.click()}
              disabled={soldierVideoUploading}
              className={`w-full px-3 py-2 rounded text-sm font-bold flex items-center justify-center gap-2 ${
                soldierVideoUploading
                  ? 'bg-gray-600 cursor-wait'
                  : 'bg-blue-600 hover:bg-blue-500'
              }`}
            >
              {soldierVideoUploading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>מעלה...</span>
                </>
              ) : (
                <>📹 העלה סרטון MP4</>
              )}
            </button>
            <p className="text-xs text-gray-400 mt-1 text-center">
              יפתח פאנל וידאו עם תמלול אוטומטי
            </p>
          </div>

          {/* Divider */}
          <div className="border-t border-gray-600 my-2"></div>

          {/* Test buttons (fake data) */}
          <h4 className="text-gray-400 text-sm">נתונים מדומים:</h4>
          <div className="space-y-2">
            <button
              onClick={() => testStartScenario('cam-1')}
              disabled={scenario.active}
              className="w-full bg-yellow-600 hover:bg-yellow-500 disabled:bg-gray-600 disabled:cursor-not-allowed px-3 py-2 rounded text-sm"
            >
              התחל תרחיש (רכב גנוב)
            </button>

            <button
              onClick={testAddArmedPerson}
              disabled={scenario.stage !== STAGES.VEHICLE_ALERT}
              className="w-full bg-red-600 hover:bg-red-500 disabled:bg-gray-600 disabled:cursor-not-allowed px-3 py-2 rounded text-sm"
            >
              הוסף חמוש ({scenario.armedCount}/3)
            </button>

            <button
              onClick={() => testTranscription('רחפן')}
              disabled={scenario.stage !== STAGES.RESPONSE_INITIATED}
              className="w-full bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 disabled:cursor-not-allowed px-3 py-2 rounded text-sm"
            >
              שלח "רחפן"
            </button>

            <button
              onClick={() => testTranscription('צפרדע')}
              disabled={scenario.stage !== STAGES.CIVILIAN_ALERT}
              className="w-full bg-orange-600 hover:bg-orange-500 disabled:bg-gray-600 disabled:cursor-not-allowed px-3 py-2 rounded text-sm"
            >
              שלח "צפרדע"
            </button>

            <button
              onClick={() => testTranscription('חדל חדל חדל')}
              disabled={scenario.stage !== STAGES.NEW_CAMERA}
              className="w-full bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed px-3 py-2 rounded text-sm"
            >
              שלח "חדל" (סיום)
            </button>

            <button
              onClick={testAdvanceStage}
              disabled={!scenario.active}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed px-3 py-2 rounded text-sm"
            >
              קדם שלב (דלג)
            </button>

            <button
              onClick={resetScenario}
              className="w-full bg-gray-600 hover:bg-gray-500 px-3 py-2 rounded text-sm"
            >
              איפוס תרחיש
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <ScenarioProvider>
        <DangerModeOverlay>
          <Dashboard />
        </DangerModeOverlay>
        <ScenarioTestControls />
      </ScenarioProvider>
    </AppProvider>
  );
}
