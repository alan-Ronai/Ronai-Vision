/**
 * Socket.IO event handlers for real-time communication
 */

// Store for active connections
const clients = new Map();

export function setupSocket(io) {
  io.on('connection', (socket) => {
    console.log(`📡 Client connected: ${socket.id}`);
    clients.set(socket.id, { type: 'unknown', connectedAt: new Date() });

    // Client identifies itself (hamal UI, soldier app, ai-service, etc.)
    socket.on('identify', (data) => {
      clients.set(socket.id, { ...data, connectedAt: new Date() });
      console.log(`🏷️  Client identified: ${socket.id} as ${data.type}`);

      // Send current state to newly connected client
      socket.emit('system:connected', {
        clientId: socket.id,
        serverTime: new Date(),
        connectedClients: clients.size
      });
    });

    // Camera selection for main view
    socket.on('camera:select', (cameraId) => {
      console.log(`📹 Camera selected: ${cameraId}`);
      socket.broadcast.emit('camera:selected', cameraId);
    });

    // Emergency acknowledgment
    socket.on('emergency:acknowledge', (data) => {
      console.log(`✅ Emergency acknowledged by: ${data.operator}`);
      io.emit('emergency:acknowledged', {
        ...data,
        acknowledgedAt: new Date()
      });
    });

    // End emergency mode
    socket.on('emergency:end', (data) => {
      console.log(`🏁 Emergency ended by: ${data.operator || 'system'}`);
      io.emit('emergency:end', {
        ...data,
        endedAt: new Date()
      });
    });

    // Soldier video upload notification
    socket.on('soldier:upload', (data) => {
      console.log(`📹 Soldier upload received:`, data);
      io.emit('event:new', {
        type: 'soldier_upload',
        severity: 'info',
        title: 'סרטון התקבל מלוחם',
        details: data,
        timestamp: new Date()
      });
    });

    // PTZ control commands
    socket.on('ptz:command', (data) => {
      console.log(`🎮 PTZ command:`, data);
      // Forward to AI service or camera controller
      io.emit('ptz:execute', data);
    });

    // Simulation triggers
    socket.on('simulation:trigger', (data) => {
      console.log(`🎭 Simulation triggered:`, data.type);
      io.emit('event:new', {
        type: 'simulation',
        severity: 'info',
        title: getSimulationTitle(data.type),
        details: { simulation: data.type, ...data },
        timestamp: new Date()
      });
    });

    // Ping/pong for connection health
    socket.on('ping', () => {
      socket.emit('pong', { serverTime: new Date() });
    });

    socket.on('disconnect', (reason) => {
      const clientInfo = clients.get(socket.id);
      clients.delete(socket.id);
      console.log(`📴 Client disconnected: ${socket.id} (${clientInfo?.type || 'unknown'}) - ${reason}`);
    });

    socket.on('error', (error) => {
      console.error(`❌ Socket error for ${socket.id}:`, error);
    });
  });

  // Utility function to emit events from anywhere in the application
  io.emitEvent = (event, data) => {
    io.emit(event, { ...data, timestamp: new Date() });
  };

  // Emit to specific client types
  io.emitToType = (type, event, data) => {
    for (const [socketId, clientInfo] of clients.entries()) {
      if (clientInfo.type === type) {
        io.to(socketId).emit(event, { ...data, timestamp: new Date() });
      }
    }
  };

  // Get connected clients info
  io.getClients = () => {
    return Array.from(clients.entries()).map(([id, info]) => ({
      id,
      ...info
    }));
  };

  return io;
}

function getSimulationTitle(type) {
  const titles = {
    drone_dispatch: 'רחפן הוקפץ',
    phone_call: 'חיוג למפקד תורן',
    pa_announcement: 'כריזה למגורים',
    code_broadcast: 'שידור קוד צפרדע',
    threat_neutralized: 'חדל - סוף אירוע'
  };
  return titles[type] || 'סימולציה';
}

export { clients };
