import { useState } from 'react';
import { AppProvider, useApp } from './context/AppContext';
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

function Dashboard() {
  const { isEmergency, simulationPopup, loading } = useApp();
  const [showCameraManager, setShowCameraManager] = useState(false);
  const [showDetectionSettings, setShowDetectionSettings] = useState(false);
  const [showGlobalIDStore, setShowGlobalIDStore] = useState(false);
  const [showAudioTransmitter, setShowAudioTransmitter] = useState(false);

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
      <StatusBar />

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

      {/* Emergency popup */}
      {isEmergency && <AlertPopup />}

      {/* Simulation popup (drone dispatch, phone call, etc.) */}
      {simulationPopup && <SimulationPopup />}
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Dashboard />
    </AppProvider>
  );
}
