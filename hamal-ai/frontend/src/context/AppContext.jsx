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

      // Check for simulation events
      if (event.type === 'simulation' && event.details?.simulation) {
        setSimulationPopup(event.details.simulation);
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

  // Play alert sound
  const playAlertSound = useCallback(() => {
    try {
      // Create audio context for alert sound
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      oscillator.frequency.value = 880;
      oscillator.type = 'sine';
      gainNode.gain.value = 0.5;

      oscillator.start();
      setTimeout(() => {
        oscillator.stop();
        audioContext.close();
      }, 500);
    } catch (e) {
      console.log('Could not play alert sound:', e);
    }
  }, []);

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

  const value = {
    socket,
    connected,
    events,
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
    API_URL
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
