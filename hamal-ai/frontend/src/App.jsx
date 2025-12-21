import { useState } from 'react';
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

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <button
        onClick={() => setShowControls(!showControls)}
        className="bg-red-700 hover:bg-red-600 px-4 py-2 rounded-lg shadow-lg mb-2"
      >
        {showControls ? 'סגור' : 'תרחיש הדגמה'}
      </button>

      {showControls && (
        <div className="bg-gray-800 rounded-lg shadow-xl p-4 w-80 space-y-2">
          <h3 className="text-lg font-bold text-white mb-3">בקרת תרחיש</h3>

          {/* Current stage */}
          <div className="bg-gray-900 rounded p-2 text-sm">
            <span className="text-gray-400">שלב נוכחי: </span>
            <span className="text-blue-400">{scenario.stage}</span>
            {scenario.armedCount > 0 && (
              <span className="text-red-400 mr-2">| חמושים: {scenario.armedCount}</span>
            )}
          </div>

          {/* Test buttons */}
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
