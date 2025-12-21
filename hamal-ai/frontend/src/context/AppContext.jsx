import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { io } from 'socket.io-client';

const AppContext = createContext();

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || 'http://localhost:3000';
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

export function AppProvider({ children }) {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [radioTranscript, setRadioTranscript] = useState([]);

  // Emergency state
  const [isEmergency, setIsEmergency] = useState(false);
  const [emergencyData, setEmergencyData] = useState(null);

  // Simulation popup
  const [simulationPopup, setSimulationPopup] = useState(null);

  // Auto-focus state (Feature 6)
  const [autoFocusState, setAutoFocusState] = useState({
    active: false,
    cameraId: null,
    previousCameraId: null,
    reason: null,
    priority: null,
    autoReturn: false,
    returnTimeout: 0,
    showIndicator: true
  });

  // Loading states
  const [loading, setLoading] = useState(true);

  // Initialize socket connection
  useEffect(() => {
    console.log('Connecting to socket server:', SOCKET_URL);
    const newSocket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });

    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to HAMAL server');
      setConnected(true);
      newSocket.emit('identify', { type: 'hamal-ui' });
    });

    newSocket.on('disconnect', (reason) => {
      console.log('Disconnected:', reason);
      setConnected(false);
    });

    newSocket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      setConnected(false);
    });

    // Event listeners
    newSocket.on('event:new', (event) => {
      console.log('New event:', event);
      setEvents(prev => [event, ...prev].slice(0, 100));

      // Check for simulation events - support multiple field locations
      if (event.type === 'simulation') {
        const simulationType = event.simulationType ||
                               event.details?.simulation ||
                               event.metadata?.simulation;
        if (simulationType) {
          setSimulationPopup(simulationType);
        }
      }
    });

    newSocket.on('emergency:start', (data) => {
      console.log('EMERGENCY START:', data);
      setIsEmergency(true);
      setEmergencyData(data);
      // Play alert sound
      playAlertSound();
    });

    newSocket.on('emergency:end', () => {
      console.log('EMERGENCY END');
      setIsEmergency(false);
      setEmergencyData(null);
    });

    newSocket.on('emergency:acknowledged', (data) => {
      console.log('Emergency acknowledged:', data);
    });

    newSocket.on('radio:transcription', (data) => {
      console.log('Radio transcription:', data);
      setRadioTranscript(prev => [...prev, data].slice(-50));
    });

    newSocket.on('camera:selected', (cameraId) => {
      console.log('Camera selected:', cameraId);
      setSelectedCamera(cameraId);
    });

    newSocket.on('camera:status', (data) => {
      setCameras(prev =>
        prev.map(cam =>
          cam.cameraId === data.cameraId
            ? { ...cam, status: data.status, lastSeen: data.lastSeen }
            : cam
        )
      );
    });

    // Camera CRUD events
    newSocket.on('camera:added', (camera) => {
      console.log('Camera added:', camera);
      setCameras(prev => [...prev, camera].sort((a, b) => (a.order || 0) - (b.order || 0)));
    });

    newSocket.on('camera:updated', (camera) => {
      console.log('Camera updated:', camera);
      setCameras(prev =>
        prev.map(cam =>
          cam.cameraId === camera.cameraId ? { ...cam, ...camera } : cam
        )
      );
    });

    newSocket.on('camera:removed', ({ cameraId }) => {
      console.log('Camera removed:', cameraId);
      setCameras(prev => prev.filter(cam => cam.cameraId !== cameraId));
      // If removed camera was selected, select another
      setSelectedCamera(prevSelected => {
        if (prevSelected === cameraId) {
          const remaining = cameras.filter(cam => cam.cameraId !== cameraId);
          return remaining.length > 0 ? remaining[0].cameraId : null;
        }
        return prevSelected;
      });
    });

    newSocket.on('camera:main', ({ cameraId }) => {
      console.log('Main camera changed:', cameraId);
      setCameras(prev =>
        prev.map(cam => ({
          ...cam,
          isMainCamera: cam.cameraId === cameraId
        }))
      );
    });

    // Auto-focus events (Feature 6)
    newSocket.on('camera:auto-focus', (data) => {
      console.log('Auto-focus triggered:', data);
      setSelectedCamera(data.newCameraId);
      setAutoFocusState({
        active: true,
        cameraId: data.newCameraId,
        previousCameraId: data.previousCameraId,
        reason: data.reason,
        priority: data.priority,
        autoReturn: data.autoReturn,
        returnTimeout: data.returnTimeout,
        showIndicator: data.showIndicator
      });
    });

    newSocket.on('camera:auto-focus-return', (data) => {
      console.log('Auto-focus returning to original camera:', data);
      setSelectedCamera(data.returnToCameraId);
      setAutoFocusState({
        active: false,
        cameraId: null,
        previousCameraId: null,
        reason: null,
        priority: null,
        autoReturn: false,
        returnTimeout: 0,
        showIndicator: true
      });
    });

    newSocket.on('camera:auto-focus-cancelled', (data) => {
      console.log('Auto-focus cancelled:', data);
      if (data.returnToCameraId) {
        setSelectedCamera(data.returnToCameraId);
      }
      setAutoFocusState({
        active: false,
        cameraId: null,
        previousCameraId: null,
        reason: null,
        priority: null,
        autoReturn: false,
        returnTimeout: 0,
        showIndicator: true
      });
    });

    // Stolen vehicle events (Feature 1)
    newSocket.on('stolen-vehicle:detected', (data) => {
      console.log('Stolen vehicle detected:', data);
      // The event system will handle alerts
    });

    // Listen for play sound events from rule engine
    newSocket.on('system:play-sound', (data) => {
      console.log('Play sound event:', data);
      playSound(data.sound, data.volume);
    });

    return () => {
      newSocket.close();
    };
  }, []);

  // Fetch initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);

        // Fetch events
        const eventsRes = await fetch(`${API_URL}/api/events/recent?limit=50`);
        if (eventsRes.ok) {
          const eventsData = await eventsRes.json();
          setEvents(eventsData);
        }

        // Fetch cameras
        const camerasRes = await fetch(`${API_URL}/api/cameras`);
        if (camerasRes.ok) {
          const camerasData = await camerasRes.json();
          setCameras(camerasData);

          // Set main camera as selected
          const mainCam = camerasData.find(c => c.isMainCamera);
          if (mainCam) {
            setSelectedCamera(mainCam.cameraId);
          } else if (camerasData.length > 0) {
            setSelectedCamera(camerasData[0].cameraId);
          }
        }

        // Fetch radio transcriptions
        const radioRes = await fetch(`${API_URL}/api/radio/transcriptions?limit=50`);
        if (radioRes.ok) {
          const radioData = await radioRes.json();
          setRadioTranscript(radioData.transcriptions || []);
        }
      } catch (error) {
        console.error('Error fetching initial data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
  }, []);

  // Play sound with different types
  const playSound = useCallback((soundType = 'alert', volume = 1) => {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      // Configure sound based on type
      const soundConfig = {
        alert: { frequency: 880, type: 'sine', duration: 500 },
        warning: { frequency: 660, type: 'triangle', duration: 400 },
        notification: { frequency: 523, type: 'sine', duration: 200 },
        success: { frequency: 1047, type: 'sine', duration: 150 },
        error: { frequency: 220, type: 'sawtooth', duration: 600 },
        beep: { frequency: 1000, type: 'square', duration: 100 },
      };

      const config = soundConfig[soundType] || soundConfig.alert;
      oscillator.frequency.value = config.frequency;
      oscillator.type = config.type;
      gainNode.gain.value = Math.min(1, Math.max(0, volume)) * 0.5;

      oscillator.start();
      setTimeout(() => {
        oscillator.stop();
        audioContext.close();
      }, config.duration);
    } catch (e) {
      console.log('Could not play sound:', e);
    }
  }, []);

  // Play alert sound (legacy, calls playSound)
  const playAlertSound = useCallback(() => {
    playSound('alert', 1);
  }, [playSound]);

  // Acknowledge emergency
  const acknowledgeEmergency = useCallback(() => {
    if (socket) {
      socket.emit('emergency:acknowledge', { operator: 'console-operator' });
    }
    setIsEmergency(false);
  }, [socket]);

  // End emergency
  const endEmergency = useCallback(() => {
    if (socket) {
      socket.emit('emergency:end', { operator: 'console-operator' });
    }
    setIsEmergency(false);
    setEmergencyData(null);
  }, [socket]);

  // Close simulation popup
  const closeSimulationPopup = useCallback(() => {
    setSimulationPopup(null);
  }, []);

  // Cancel auto-focus (Feature 6)
  const cancelAutoFocus = useCallback(async () => {
    try {
      await fetch(`${API_URL}/api/cameras/auto-focus/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'user_cancelled' })
      });
    } catch (error) {
      console.error('Error cancelling auto-focus:', error);
    }
    setAutoFocusState({
      active: false,
      cameraId: null,
      previousCameraId: null,
      reason: null,
      priority: null,
      autoReturn: false,
      returnTimeout: 0,
      showIndicator: true
    });
  }, []);

  // Select camera
  const selectCamera = useCallback((cameraId) => {
    setSelectedCamera(cameraId);
    if (socket) {
      socket.emit('camera:select', cameraId);
    }
  }, [socket]);

  // Trigger simulation (for demo/testing)
  const triggerSimulation = useCallback(async (type) => {
    try {
      await fetch(`${API_URL}/api/radio/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: type })
      });
    } catch (error) {
      console.error('Error triggering simulation:', error);
    }
  }, []);

  // Create demo event (for testing)
  const createDemoEvent = useCallback(async (severity = 'warning') => {
    try {
      await fetch(`${API_URL}/api/events`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'detection',
          severity,
          title: severity === 'critical' ? 'חדירה ודאית!' : 'זיהוי תנועה חשודה',
          cameraId: selectedCamera,
          details: {
            people: { count: severity === 'critical' ? 3 : 1, armed: severity === 'critical' },
            vehicle: severity !== 'info' ? { type: 'car', color: 'שחור' } : null
          }
        })
      });
    } catch (error) {
      console.error('Error creating demo event:', error);
    }
  }, [selectedCamera]);

  // Clear radio transcriptions
  const clearRadioTranscript = () => {
    setRadioTranscript([]);
  };

  // Clear events (local only)
  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  const value = {
    socket,
    connected,
    events,
    clearEvents,
    cameras,
    selectedCamera,
    selectCamera,
    radioTranscript,
    clearRadioTranscript,
    isEmergency,
    emergencyData,
    acknowledgeEmergency,
    endEmergency,
    simulationPopup,
    closeSimulationPopup,
    loading,
    triggerSimulation,
    createDemoEvent,
    API_URL,
    // Feature 6: Auto-focus
    autoFocusState,
    cancelAutoFocus
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};
