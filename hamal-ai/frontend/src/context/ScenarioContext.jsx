/**
 * Scenario Context - State Management for Armed Attack Demo
 *
 * This context manages all scenario-related state including:
 * - Current scenario stage
 * - Danger mode state
 * - Emergency modals and alerts
 * - Simulation indicators
 * - Sound management
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
} from 'react';
import { useApp } from './AppContext';

const ScenarioContext = createContext();

// Scenario stages (matches backend)
export const STAGES = {
  IDLE: 'idle',
  VEHICLE_DETECTED: 'vehicle_detected',
  VEHICLE_ALERT: 'vehicle_alert',
  ARMED_PERSONS_DETECTED: 'armed_persons_detected',
  EMERGENCY_MODE: 'emergency_mode',
  RESPONSE_INITIATED: 'response_initiated',
  DRONE_DISPATCHED: 'drone_dispatched',
  CIVILIAN_ALERT: 'civilian_alert',
  CODE_BROADCAST: 'code_broadcast',
  SOLDIER_VIDEO: 'soldier_video',
  NEW_CAMERA: 'new_camera',
  SITUATION_END: 'situation_end',
};

// API URL
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

export function ScenarioProvider({ children }) {
  const { socket } = useApp();

  // Core scenario state
  const [scenario, setScenario] = useState({
    active: false,
    stage: STAGES.IDLE,
    scenarioId: null,
    startedAt: null,
    vehicle: null,
    persons: [],
    armedCount: 0,
    acknowledged: false,
    stageHistory: [],
  });

  // UI states
  const [dangerMode, setDangerMode] = useState(false);
  const [emergencyModal, setEmergencyModal] = useState(null);
  const [alertPopup, setAlertPopup] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [soldierVideo, setSoldierVideo] = useState(null);
  const [newCameraDialog, setNewCameraDialog] = useState(false);
  const [summaryPopup, setSummaryPopup] = useState(null);
  const [ttsMessage, setTtsMessage] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  // Sound management
  const audioRef = useRef({});
  const [activeSounds, setActiveSounds] = useState([]);
  const [soundVolume, setSoundVolume] = useState(0.5); // Default 50% volume
  const [soundMuted, setSoundMuted] = useState(false);

  // Configuration (loaded from backend)
  const [config, setConfig] = useState(null);

  // Load configuration on mount
  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/config`);
      if (response.ok) {
        const data = await response.json();
        setConfig(data);
      }
    } catch (error) {
      console.error('[ScenarioContext] Failed to load config:', error);
    }
  };

  // Socket event listeners
  useEffect(() => {
    if (!socket) return;

    // Wrapper functions for sound handlers (use refs to avoid stale closures)
    const handlePlaySound = (data) => {
      if (handlePlaySoundRef.current) {
        handlePlaySoundRef.current(data);
      }
    };
    const handleStopSound = (data) => {
      if (handleStopSoundRef.current) {
        handleStopSoundRef.current(data);
      }
    };

    // Stage changed (legacy scenario manager)
    socket.on('scenario:stage-changed', handleStageChanged);

    // Engine stage transition (new rule engine)
    socket.on('scenario:engine-transition', handleEngineTransition);

    // Engine stage update (new rule engine)
    socket.on('scenario:engine-stage', handleEngineStage);

    // Danger mode
    socket.on('scenario:danger-mode', handleDangerMode);

    // Emergency modal
    socket.on('scenario:emergency-modal', handleEmergencyModal);

    // Alert popup
    socket.on('scenario:alert', handleAlert);

    // Simulation indicator
    socket.on('scenario:simulation', handleSimulation);

    // Soldier video panel
    socket.on('scenario:soldier-video', handleSoldierVideo);

    // New camera dialog
    socket.on('scenario:new-camera-dialog', handleNewCameraDialog);

    // Summary popup
    socket.on('scenario:summary', handleSummary);

    // TTS message
    socket.on('scenario:tts', handleTTS);

    // Sound controls
    socket.on('scenario:play-sound', handlePlaySound);
    socket.on('scenario:stop-sound', handleStopSound);

    // Scenario ended
    socket.on('scenario:ended', handleScenarioEnded);

    // Stop all sounds event
    socket.on('scenario:stop-all-sounds', handleStopAllSounds);

    // Recording control
    socket.on('scenario:recording', handleRecording);

    return () => {
      socket.off('scenario:stage-changed', handleStageChanged);
      socket.off('scenario:engine-transition', handleEngineTransition);
      socket.off('scenario:engine-stage', handleEngineStage);
      socket.off('scenario:danger-mode', handleDangerMode);
      socket.off('scenario:emergency-modal', handleEmergencyModal);
      socket.off('scenario:alert', handleAlert);
      socket.off('scenario:simulation', handleSimulation);
      socket.off('scenario:soldier-video', handleSoldierVideo);
      socket.off('scenario:new-camera-dialog', handleNewCameraDialog);
      socket.off('scenario:summary', handleSummary);
      socket.off('scenario:tts', handleTTS);
      socket.off('scenario:play-sound', handlePlaySound);
      socket.off('scenario:stop-sound', handleStopSound);
      socket.off('scenario:stop-all-sounds', handleStopAllSounds);
      socket.off('scenario:ended', handleScenarioEnded);
      socket.off('scenario:recording', handleRecording);
    };
  }, [socket]);

  // Event handlers
  const handleStageChanged = useCallback((data) => {
    console.log('[ScenarioContext] Stage changed:', data);
    setScenario(prev => ({
      ...prev,
      ...data.state,
      stage: data.stage,
    }));
  }, []);

  // Handle engine transition (new rule engine)
  const handleEngineTransition = useCallback((data) => {
    console.log('[ScenarioContext] Engine transition:', data);
    setScenario(prev => ({
      ...prev,
      active: true,
      scenarioId: data.scenarioId,
      stage: data.toStage,
      vehicle: data.context?.vehicle || prev.vehicle,
      persons: data.context?.persons || prev.persons,
      armedCount: data.context?.armedCount || prev.armedCount,
      stageHistory: [
        ...prev.stageHistory,
        { from: data.fromStage, to: data.toStage, at: data.timestamp }
      ],
    }));
  }, []);

  // Handle engine stage updates (new rule engine)
  const handleEngineStage = useCallback((data) => {
    console.log('[ScenarioContext] Engine stage update:', data);
    setScenario(prev => ({
      ...prev,
      active: true,
      scenarioId: data.scenarioId || prev.scenarioId,
      stage: data.stage || prev.stage,
      vehicle: data.context?.vehicle || prev.vehicle,
      persons: data.context?.persons || prev.persons,
      armedCount: data.context?.armedCount || prev.armedCount,
    }));
  }, []);

  // Handle stop all sounds
  const handleStopAllSounds = useCallback(() => {
    console.log('[ScenarioContext] Stop all sounds');
    stopAllSounds();
  }, []);

  const handleDangerMode = useCallback((data) => {
    console.log('[ScenarioContext] Danger mode:', data.active);
    setDangerMode(data.active);

    if (data.active) {
      // Play alarm sound twice (not infinite loop)
      playSound('alarm', false, soundVolume, 2);
    } else {
      // Stop alarm sound
      stopSound('alarm');
    }
  }, [soundVolume]);

  const handleEmergencyModal = useCallback((data) => {
    console.log('[ScenarioContext] Emergency modal:', data);
    if (data.close) {
      setEmergencyModal(null);
    } else {
      setEmergencyModal(data);
    }
  }, []);

  const handleAlert = useCallback((data) => {
    console.log('[ScenarioContext] Alert:', data);
    setAlertPopup(data);

    // Auto-dismiss after 10 seconds if not critical
    if (data.autoDismiss !== false && data.type !== 'critical') {
      setTimeout(() => {
        setAlertPopup(prev => prev === data ? null : prev);
      }, 10000);
    }
  }, []);

  const handleSimulation = useCallback((data) => {
    console.log('[ScenarioContext] Simulation:', data);
    setSimulation(data);

    // Play associated sound if specified
    if (data.sound) {
      playSound(data.sound);
    }

    // Auto-hide after 5 seconds
    setTimeout(() => {
      setSimulation(prev => prev === data ? null : prev);
    }, 5000);
  }, []);

  const handleSoldierVideo = useCallback((data) => {
    console.log('[ScenarioContext] Soldier video:', data);
    if (data.open) {
      setSoldierVideo(data);
    } else {
      setSoldierVideo(null);
    }
  }, []);

  const handleNewCameraDialog = useCallback((data) => {
    console.log('[ScenarioContext] New camera dialog:', data.open);
    setNewCameraDialog(data.open);
  }, []);

  const handleSummary = useCallback((data) => {
    console.log('[ScenarioContext] Summary:', data);
    setSummaryPopup(data);
  }, []);

  const handleTTS = useCallback((data) => {
    console.log('[ScenarioContext] TTS:', data.message);
    setTtsMessage(data.message);

    // Clear after display
    setTimeout(() => {
      setTtsMessage(null);
    }, 5000);
  }, []);

  const handleRecording = useCallback((data) => {
    console.log('[ScenarioContext] Recording:', data);
    if (data.action === 'start') {
      setIsRecording(true);
      // Emit camera recording start to all cameras if allCameras is true
      if (data.allCameras && socket) {
        socket.emit('camera:start-recording', {
          scenarioId: data.scenarioId,
          preBuffer: data.preBuffer || 30,
          allCameras: true
        });
      }
    } else if (data.action === 'stop') {
      setIsRecording(false);
      if (socket) {
        socket.emit('camera:stop-recording', {
          scenarioId: data.scenarioId
        });
      }
    }
  }, [socket]);

  // These handlers are defined as regular functions since they need to call
  // playSound/stopSound which are defined later. They're wrapped in useCallback
  // in the socket effect setup.
  const handlePlaySoundRef = useRef(null);
  const handleStopSoundRef = useRef(null);

  const handleScenarioEnded = useCallback(() => {
    console.log('[ScenarioContext] Scenario ended');
    setScenario({
      active: false,
      stage: STAGES.IDLE,
      scenarioId: null,
      startedAt: null,
      vehicle: null,
      persons: [],
      armedCount: 0,
      acknowledged: false,
      stageHistory: [],
    });
    setDangerMode(false);
    setEmergencyModal(null);
    setAlertPopup(null);
    setSimulation(null);
    setSoldierVideo(null);
    setNewCameraDialog(false);
    stopAllSounds();
  }, []);

  // Sound functions
  const playSound = useCallback((soundName, loop = false, volume = null, repeatCount = 1) => {
    // Use global volume if not specified, apply mute
    const effectiveVolume = soundMuted ? 0 : (volume ?? soundVolume);

    if (effectiveVolume === 0) {
      console.log('[ScenarioContext] Sound muted, skipping:', soundName);
      return;
    }

    try {
      // Generate sound using Web Audio API
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      gainNode.gain.value = effectiveVolume * 0.3; // Scale down for comfort

      // Different sounds based on type
      switch (soundName) {
        case 'alarm':
          oscillator.type = 'sawtooth';
          oscillator.frequency.value = 440;

          // Play alarm pattern specified number of times
          let playCount = 0;
          const alarmPattern = () => {
            if (playCount >= repeatCount) {
              // Stop after specified repeats
              try {
                oscillator.stop();
                audioContext.close();
              } catch (e) {}
              return;
            }
            playCount++;
            oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
            oscillator.frequency.setValueAtTime(880, audioContext.currentTime + 0.25);
            oscillator.frequency.setValueAtTime(440, audioContext.currentTime + 0.5);
          };

          alarmPattern();
          if (repeatCount > 1) {
            const interval = setInterval(() => {
              if (playCount >= repeatCount) {
                clearInterval(interval);
                try {
                  oscillator.stop();
                  audioContext.close();
                } catch (e) {}
                setActiveSounds(prev => prev.filter(s => s !== soundName));
                return;
              }
              alarmPattern();
            }, 1000);
            audioRef.current[soundName] = { audioContext, oscillator, interval };
          }
          break;

        case 'phoneDial':
        case 'phone-dial':
          oscillator.type = 'sine';
          oscillator.frequency.value = 350;
          break;

        case 'phoneRing':
        case 'phone-ring':
          oscillator.type = 'sine';
          oscillator.frequency.value = 440;

          // Phone ring pattern with repeat count support
          if (repeatCount > 1) {
            let ringCount = 0;
            const ringPattern = () => {
              if (ringCount >= repeatCount) {
                clearInterval(ringInterval);
                try {
                  oscillator.stop();
                  audioContext.close();
                } catch (e) {}
                setActiveSounds(prev => prev.filter(s => s !== soundName));
                return;
              }
              ringCount++;
              // Ring pattern: high-low-high
              oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
              oscillator.frequency.setValueAtTime(480, audioContext.currentTime + 0.15);
              oscillator.frequency.setValueAtTime(440, audioContext.currentTime + 0.3);
            };
            ringPattern();
            const ringInterval = setInterval(ringPattern, 800);
            audioRef.current[soundName] = { audioContext, oscillator, interval: ringInterval };
          }
          break;

        case 'droneTakeoff':
        case 'drone-takeoff':
          oscillator.type = 'triangle';
          oscillator.frequency.value = 100;
          // Ramp up
          oscillator.frequency.linearRampToValueAtTime(300, audioContext.currentTime + 2);
          break;

        case 'alert':
          oscillator.type = 'sine';
          oscillator.frequency.value = 880;
          break;

        case 'success':
          oscillator.type = 'sine';
          oscillator.frequency.value = 1047; // C6
          break;

        default:
          oscillator.type = 'sine';
          oscillator.frequency.value = 440;
      }

      oscillator.start();

      // Handle sounds with repeat patterns (alarm, phoneRing)
      const soundsWithRepeat = ['alarm', 'phoneRing', 'phone-ring'];
      if (!loop && !soundsWithRepeat.includes(soundName)) {
        oscillator.stop(audioContext.currentTime + 0.5);
      } else if (!loop && soundsWithRepeat.includes(soundName)) {
        // Sounds with repeat count are handled in their switch cases above
      } else {
        audioRef.current[soundName] = { audioContext, oscillator };
      }

      setActiveSounds(prev => [...prev, soundName]);
    } catch (error) {
      console.error('[ScenarioContext] Sound error:', error);
    }
  }, [soundVolume, soundMuted]);

  const stopSound = useCallback((soundName) => {
    try {
      const ref = audioRef.current[soundName];
      if (ref) {
        if (ref.oscillator) {
          ref.oscillator.stop();
        }
        if (ref.interval) {
          clearInterval(ref.interval);
        }
        if (ref.audioContext) {
          ref.audioContext.close();
        }
        delete audioRef.current[soundName];
      }
      setActiveSounds(prev => prev.filter(s => s !== soundName));
    } catch (error) {
      console.error('[ScenarioContext] Stop sound error:', error);
    }
  }, []);

  const stopAllSounds = useCallback(() => {
    Object.keys(audioRef.current).forEach(soundName => {
      stopSound(soundName);
    });
    setActiveSounds([]);
  }, [stopSound]);

  // Update refs for socket handlers (these need to reference playSound/stopSound)
  handlePlaySoundRef.current = useCallback((data) => {
    console.log('[ScenarioContext] Play sound:', data);
    playSound(data.sound, data.loop, data.volume, data.repeatCount || 1);
  }, [playSound]);

  handleStopSoundRef.current = useCallback((data) => {
    console.log('[ScenarioContext] Stop sound:', data);
    stopSound(data.sound);
  }, [stopSound]);

  // API functions
  const acknowledgeEmergency = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/acknowledge`, {
        method: 'POST',
      });
      return response.ok;
    } catch (error) {
      console.error('[ScenarioContext] Acknowledge error:', error);
      return false;
    }
  }, []);

  const declareFalseAlarm = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/false-alarm`, {
        method: 'POST',
      });
      return response.ok;
    } catch (error) {
      console.error('[ScenarioContext] False alarm error:', error);
      return false;
    }
  }, []);

  const closeSoldierVideoPanel = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/video-panel-closed`, {
        method: 'POST',
      });
      setSoldierVideo(null);
      return response.ok;
    } catch (error) {
      console.error('[ScenarioContext] Close video panel error:', error);
      return false;
    }
  }, []);

  const connectNewCamera = useCallback(async (url) => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/new-camera`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      if (response.ok) {
        setNewCameraDialog(false);
      }
      return response.ok;
    } catch (error) {
      console.error('[ScenarioContext] Connect camera error:', error);
      return false;
    }
  }, []);

  const closeSummary = useCallback(() => {
    setSummaryPopup(null);
  }, []);

  const dismissAlert = useCallback(() => {
    setAlertPopup(null);
  }, []);

  // Test functions (for demo)
  const testStartScenario = useCallback(async (cameraId) => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/test/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cameraId }),
      });
      return response.ok;
    } catch (error) {
      console.error('[ScenarioContext] Test start error:', error);
      return false;
    }
  }, []);

  const testAddArmedPerson = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/test/add-armed-person`, {
        method: 'POST',
      });
      return response.ok;
    } catch (error) {
      console.error('[ScenarioContext] Test armed person error:', error);
      return false;
    }
  }, []);

  const testAdvanceStage = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/test/advance`, {
        method: 'POST',
      });
      return response.ok;
    } catch (error) {
      console.error('[ScenarioContext] Test advance error:', error);
      return false;
    }
  }, []);

  const testTranscription = useCallback(async (text) => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/test/transcription`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      return response.ok;
    } catch (error) {
      console.error('[ScenarioContext] Test transcription error:', error);
      return false;
    }
  }, []);

  const resetScenario = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/scenario/reset`, {
        method: 'POST',
      });
      return response.ok;
    } catch (error) {
      console.error('[ScenarioContext] Reset error:', error);
      return false;
    }
  }, []);

  const value = {
    // State
    scenario,
    dangerMode,
    emergencyModal,
    alertPopup,
    simulation,
    soldierVideo,
    newCameraDialog,
    summaryPopup,
    ttsMessage,
    config,
    activeSounds,
    isRecording,

    // Volume controls
    soundVolume,
    setSoundVolume,
    soundMuted,
    setSoundMuted,

    // Actions
    acknowledgeEmergency,
    declareFalseAlarm,
    closeSoldierVideoPanel,
    connectNewCamera,
    closeSummary,
    dismissAlert,
    playSound,
    stopSound,
    stopAllSounds,

    // Test functions
    testStartScenario,
    testAddArmedPerson,
    testAdvanceStage,
    testTranscription,
    resetScenario,

    // Stage helpers
    isActive: scenario.active,
    stage: scenario.stage,
    STAGES,
  };

  return (
    <ScenarioContext.Provider value={value}>
      {children}
    </ScenarioContext.Provider>
  );
}

export function useScenario() {
  const context = useContext(ScenarioContext);
  if (!context) {
    throw new Error('useScenario must be used within a ScenarioProvider');
  }
  return context;
}

export default ScenarioContext;
