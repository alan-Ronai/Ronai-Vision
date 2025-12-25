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
import ActiveScenariosModal from './components/ActiveScenariosModal';
import GIDPickerModal from './components/GIDPickerModal';

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
  const [showActiveScenarios, setShowActiveScenarios] = useState(false);

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
      <StatusBar
        onOpenAIStats={() => setShowAIStats(true)}
        onOpenActiveScenarios={() => setShowActiveScenarios(true)}
      />

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

      {/* Active Scenarios Modal */}
      <ActiveScenariosModal
        isOpen={showActiveScenarios}
        onClose={() => setShowActiveScenarios(false)}
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

// Scenario Test Controls - Dynamic demo buttons based on config and active variant
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3000';

// Stage display names in Hebrew
const STAGE_NAMES = {
  idle: 'מצב רגיל',
  vehicle_detected: 'רכב זוהה',
  vehicle_alert: 'התראת רכב',
  armed_person_detected: 'חמוש זוהה',
  emergency_mode: 'מצב חירום',
  response_initiated: 'תגובה הופעלה',
  drone_dispatched: 'רחפן הוקפץ',
  civilian_alert: 'כריזה למגורים',
  code_broadcast: 'קוד שודר',
  soldier_video: 'סרטון מלוחם',
  new_camera: 'מצלמה חדשה',
  situation_end: 'סיום אירוע',
};

function ScenarioTestControls() {
  const {
    scenario,
    config,
    testStartScenario,
    testStartArmedScenario,
    testAddArmedPerson,
    testAdvanceStage,
    testTranscription,
    resetScenario,
    STAGES,
  } = useScenario();

  const [showControls, setShowControls] = useState(false);
  const [dataSource, setDataSource] = useState('fake'); // 'fake' or 'real'
  const [realDataLoading, setRealDataLoading] = useState(false);
  const [realDataMessage, setRealDataMessage] = useState('');
  const [soldierVideoUploading, setSoldierVideoUploading] = useState(false);
  const soldierVideoInputRef = useRef(null);

  // GID picker state
  const [showGidPicker, setShowGidPicker] = useState(false);
  const [gidPickerType, setGidPickerType] = useState(null); // 'vehicle' or 'person'

  // Get active variant info
  const activeVariant = scenario.variant;
  const variantInfo = config?.variants?.[activeVariant] || scenario.variantInfo;

  // Get stage flow for active variant (or show both if not active)
  const getStageFlow = () => {
    if (activeVariant && variantInfo?.stageFlow) {
      return variantInfo.stageFlow;
    }
    // Default stolen-vehicle flow when no scenario is active
    return config?.variants?.['stolen-vehicle']?.stageFlow || [
      'idle', 'vehicle_detected', 'vehicle_alert', 'emergency_mode',
      'response_initiated', 'drone_dispatched', 'civilian_alert',
      'code_broadcast', 'soldier_video', 'new_camera', 'situation_end'
    ];
  };

  // Check if current stage is in flow
  const currentStageIndex = getStageFlow().indexOf(scenario.stage?.toLowerCase() || 'idle');

  // Trigger real data scenario
  const triggerRealScenario = async (type, selectedObject = null) => {
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
          if (selectedObject) {
            body = { gid: selectedObject.gid };
          }
          break;
        case 'real-armed':
          endpoint = '/api/scenario/demo/real/armed-persons';
          body = { count: 1 };
          if (selectedObject) {
            body.gid = selectedObject.gid;
          }
          break;
        case 'real-armed-direct':
          endpoint = '/api/scenario/test/start-armed';
          if (selectedObject) {
            body = { gid: selectedObject.gid };
          }
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
        const gidInfo = selectedObject ? ` (GID: ${selectedObject.gid})` : ' (אקראי)';
        setRealDataMessage((result.message || 'הופעל בהצלחה!') + gidInfo);
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

  // Open GID picker
  const openGidPicker = (type) => {
    setGidPickerType(type);
    setShowGidPicker(true);
  };

  // Handle GID selection
  const handleGidSelect = (selectedObject) => {
    if (gidPickerType === 'vehicle') {
      triggerRealScenario('real-vehicle', selectedObject);
    } else if (gidPickerType === 'person') {
      triggerRealScenario('real-armed-direct', selectedObject);
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

  // Render stage progress indicator
  const renderStageProgress = () => {
    const stages = getStageFlow();
    return (
      <div className="bg-gray-900 rounded-lg p-3 mb-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-gray-400">התקדמות תרחיש:</span>
          {variantInfo && (
            <span className="text-xs bg-gray-700 px-2 py-0.5 rounded">
              {variantInfo.icon} {variantInfo.name}
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-1">
          {stages.map((stage, i) => {
            const isCurrent = i === currentStageIndex;
            const isPast = i < currentStageIndex;
            const stageName = STAGE_NAMES[stage] || stage;

            return (
              <div
                key={stage}
                className={`
                  px-1.5 py-0.5 rounded text-[10px] whitespace-nowrap
                  ${isCurrent ? 'bg-orange-500 text-white font-bold animate-pulse' : ''}
                  ${isPast ? 'bg-green-600 text-white' : ''}
                  ${!isCurrent && !isPast ? 'bg-gray-700 text-gray-400' : ''}
                `}
                title={stageName}
              >
                {stageName}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Render variant-specific controls
  const renderVariantControls = () => {
    // If scenario is active, show controls based on active variant
    if (scenario.active) {
      const isVehicleVariant = activeVariant === 'stolen-vehicle';
      const isArmedVariant = activeVariant === 'armed-person';

      return (
        <div className="space-y-2">
          {/* Show "add armed person" only for stolen-vehicle variant at VEHICLE_ALERT stage */}
          {isVehicleVariant && scenario.stage === STAGES.VEHICLE_ALERT && (
            <button
              onClick={testAddArmedPerson}
              className="w-full bg-red-600 hover:bg-red-500 px-3 py-2 rounded text-sm font-bold animate-pulse"
            >
              🔫 הוסף חמוש ({scenario.armedCount}/{config?.armedThreshold || 1})
            </button>
          )}

          {/* Stage-specific keyword buttons */}
          {scenario.stage === STAGES.RESPONSE_INITIATED && (
            <button
              onClick={() => testTranscription('רחפן')}
              className="w-full bg-purple-600 hover:bg-purple-500 px-3 py-2 rounded text-sm"
            >
              🚁 שלח "רחפן"
            </button>
          )}

          {scenario.stage === STAGES.CIVILIAN_ALERT && (
            <button
              onClick={() => testTranscription('צפרדע')}
              className="w-full bg-orange-600 hover:bg-orange-500 px-3 py-2 rounded text-sm"
            >
              🐸 שלח "צפרדע"
            </button>
          )}

          {scenario.stage === STAGES.NEW_CAMERA && (
            <button
              onClick={() => testTranscription('חדל חדל חדל')}
              className="w-full bg-green-600 hover:bg-green-500 px-3 py-2 rounded text-sm"
            >
              ✅ שלח "חדל" (סיום)
            </button>
          )}

          {/* Always show advance and reset when active */}
          <div className="flex gap-2">
            <button
              onClick={testAdvanceStage}
              className="flex-1 bg-blue-600 hover:bg-blue-500 px-3 py-2 rounded text-sm"
            >
              ⏩ קדם שלב
            </button>
            <button
              onClick={resetScenario}
              className="flex-1 bg-gray-600 hover:bg-gray-500 px-3 py-2 rounded text-sm"
            >
              ⏹️ עצור
            </button>
          </div>
        </div>
      );
    }

    // If scenario is not active, show start buttons for each variant
    return (
      <div className="space-y-3">
        {/* Variant selection header */}
        <div className="text-center text-sm text-gray-400 mb-2">
          בחר סוג תרחיש להפעלה:
        </div>

        {/* Stolen Vehicle Variant */}
        <div className="border border-yellow-500/50 rounded-lg p-3 bg-yellow-900/10">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">🚗</span>
            <div>
              <div className="font-bold text-yellow-400 text-sm">רכב גנוב + חמושים</div>
              <div className="text-xs text-gray-400">זיהוי רכב גנוב, המתנה לחמושים</div>
            </div>
          </div>
          <div className="flex gap-2">
            {dataSource === 'fake' ? (
              <button
                onClick={() => testStartScenario('cam-1')}
                className="flex-1 bg-yellow-600 hover:bg-yellow-500 px-3 py-2 rounded text-sm"
              >
                ▶️ הפעל (מדומה)
              </button>
            ) : (
              <button
                onClick={() => openGidPicker('vehicle')}
                disabled={realDataLoading}
                className="flex-1 bg-yellow-600 hover:bg-yellow-500 disabled:bg-gray-600 px-3 py-2 rounded text-sm"
              >
                🎯 בחר רכב
              </button>
            )}
          </div>
        </div>

        {/* Armed Person Variant */}
        <div className="border border-red-500/50 rounded-lg p-3 bg-red-900/10">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">🔫</span>
            <div>
              <div className="font-bold text-red-400 text-sm">חמוש זוהה (ישיר)</div>
              <div className="text-xs text-gray-400">מעבר מיידי למצב חירום</div>
            </div>
          </div>
          <div className="flex gap-2">
            {dataSource === 'fake' ? (
              <button
                onClick={() => testStartArmedScenario('cam-1')}
                className="flex-1 bg-red-600 hover:bg-red-500 px-3 py-2 rounded text-sm"
              >
                ▶️ הפעל (מדומה)
              </button>
            ) : (
              <button
                onClick={() => openGidPicker('person')}
                disabled={realDataLoading}
                className="flex-1 bg-red-600 hover:bg-red-500 disabled:bg-gray-600 px-3 py-2 rounded text-sm"
              >
                🎯 בחר אדם
              </button>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <button
        onClick={() => setShowControls(!showControls)}
        className={`px-4 py-2 rounded-lg shadow-lg mb-2 transition-colors ${
          scenario.active
            ? 'bg-orange-600 hover:bg-orange-500 animate-pulse'
            : 'bg-red-700 hover:bg-red-600'
        }`}
      >
        {showControls ? 'סגור' : scenario.active ? `🎬 ${variantInfo?.name || 'תרחיש פעיל'}` : 'תרחיש הדגמה'}
      </button>

      {showControls && (
        <div className="bg-gray-800 rounded-lg shadow-xl p-4 w-80 space-y-2 max-h-[80vh] overflow-y-auto">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-bold text-white">בקרת תרחיש</h3>
            {scenario.active && (
              <span className="text-xs bg-orange-600 px-2 py-1 rounded animate-pulse">
                פעיל
              </span>
            )}
          </div>

          {/* Stage Progress */}
          {renderStageProgress()}

          {/* Current Status */}
          <div className="bg-gray-900 rounded p-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">שלב:</span>
              <span className="text-blue-400 font-bold">
                {STAGE_NAMES[scenario.stage?.toLowerCase()] || scenario.stage}
              </span>
            </div>
            {scenario.armedCount > 0 && (
              <div className="flex justify-between mt-1">
                <span className="text-gray-400">חמושים:</span>
                <span className="text-red-400 font-bold">{scenario.armedCount}</span>
              </div>
            )}
            {scenario.vehicle && (
              <div className="flex justify-between mt-1">
                <span className="text-gray-400">רכב:</span>
                <span className="text-yellow-400 text-xs">{scenario.vehicle.licensePlate}</span>
              </div>
            )}
          </div>

          {/* Status Message */}
          {realDataMessage && (
            <div className="bg-blue-900 text-blue-200 p-2 rounded text-sm text-center">
              {realDataMessage}
            </div>
          )}

          {/* Data Source Toggle */}
          {!scenario.active && (
            <div className="flex rounded-lg overflow-hidden border border-gray-600">
              <button
                onClick={() => setDataSource('fake')}
                className={`flex-1 py-2 text-sm transition-colors ${
                  dataSource === 'fake'
                    ? 'bg-gray-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                🎭 נתונים מדומים
              </button>
              <button
                onClick={() => setDataSource('real')}
                className={`flex-1 py-2 text-sm transition-colors ${
                  dataSource === 'real'
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                🎯 נתוני סצנה
              </button>
            </div>
          )}

          {/* Variant-specific Controls */}
          {renderVariantControls()}

          {/* Soldier Video Upload - show when relevant */}
          {scenario.stage === STAGES.CODE_BROADCAST && (
            <div className="border-2 border-blue-500 rounded-lg p-3 bg-blue-900/20 mt-3">
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
            </div>
          )}

          {/* Additional Real Data Options when in real mode */}
          {dataSource === 'real' && !scenario.active && (
            <div className="border-t border-gray-600 pt-3 mt-3">
              <h4 className="text-green-400 text-sm font-bold mb-2">🎯 אפשרויות נוספות:</h4>
              <button
                onClick={() => triggerRealScenario('real-full')}
                disabled={realDataLoading}
                className="w-full bg-green-700 hover:bg-green-600 disabled:bg-gray-600 px-3 py-2 rounded text-sm"
              >
                🎬 תרחיש מלא אוטומטי
              </button>
            </div>
          )}
        </div>
      )}

      {/* GID Picker Modal */}
      <GIDPickerModal
        isOpen={showGidPicker}
        onClose={() => setShowGidPicker(false)}
        onSelect={handleGidSelect}
        type={gidPickerType}
      />
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
