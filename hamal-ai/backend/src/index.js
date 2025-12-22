import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import mongoose from 'mongoose';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

import cameraRoutes from './routes/cameras.js';
import eventRoutes from './routes/events.js';
import uploadRoutes from './routes/uploads.js';
import radioRoutes from './routes/radio.js';
import streamRoutes from './routes/stream.js';
import trackedRoutes from './routes/tracked.js';
import alertRoutes from './routes/alerts.js';
import eventRulesRoutes from './routes/eventRules.js';
import stolenPlatesRoutes from './routes/stolenPlates.js';
import scenarioRoutes from './routes/scenario.js';
import { setupSocket } from './socket/handlers.js';
import { initScenarioManager } from './services/scenarioManager.js';

// Load environment variables
dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: true, // Allow all origins for Socket.IO
    methods: ['GET', 'POST'],
    credentials: true
  }
});

// CORS configuration - allow multiple origins for EC2/local development
const allowedOrigins = process.env.FRONTEND_URL
  ? process.env.FRONTEND_URL.split(',').map(url => url.trim())
  : null; // null means allow all origins

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps or curl)
    if (!origin) return callback(null, true);

    // If no specific origins configured, allow all
    if (!allowedOrigins) return callback(null, true);

    // Check if origin is in allowed list
    if (allowedOrigins.includes(origin)) {
      return callback(null, true);
    }

    // Also allow localhost variants
    if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
      return callback(null, true);
    }

    callback(null, true); // Allow all for development
  },
  credentials: true
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Static file serving for HLS and clips
const dataPath = process.env.DATA_PATH || path.join(__dirname, '../../data');
app.use('/hls', express.static(path.join(dataPath, 'hls')));
app.use('/clips', express.static(path.join(dataPath, 'clips')));
app.use('/recordings', express.static(path.join(dataPath, 'recordings')));

// Connect MongoDB
const mongoUri = process.env.MONGODB_URI || 'mongodb://localhost:27017/hamal';
mongoose.connect(mongoUri)
  .then(() => console.log('✅ MongoDB connected to:', mongoUri))
  .catch(err => {
    console.error('❌ MongoDB connection error:', err.message);
    console.log('⚠️  Server will continue without database - some features may not work');
  });

// Make io available to routes
app.set('io', io);

// Routes
app.use('/api/cameras', cameraRoutes);
app.use('/api/events', eventRoutes);
app.use('/api/uploads', uploadRoutes);
app.use('/api/radio', radioRoutes);
app.use('/api/stream', streamRoutes);
app.use('/api/tracked', trackedRoutes);
app.use('/api/alerts', alertRoutes);
app.use('/api/event-rules', eventRulesRoutes);
app.use('/api/stolen-plates', stolenPlatesRoutes);
app.use('/api/scenario', scenarioRoutes);

// Serve scenario uploads
app.use('/uploads/scenario', express.static(path.join(__dirname, '../uploads/scenario')));

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date(),
    mongodb: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected'
  });
});

// System routes (for AI service actions)
app.post('/api/system/play-sound', (req, res) => {
  const { sound = 'alert', volume = 1 } = req.body;
  io.emit('system:play-sound', { sound, volume });
  res.json({ success: true, sound, volume });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

// Socket.IO setup
setupSocket(io);

// Initialize Scenario Manager
initScenarioManager(io);
console.log('✅ Scenario Manager initialized');

// Export io for use in other modules
export { io };

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════╗
║                                                        ║
║   🛡️  HAMAL-AI Security Command Center                 ║
║   חמ"ל אירועים מבוסס AI                                  ║
║                                                        ║
║   Server running on http://localhost:${PORT}             ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
  `);
});
