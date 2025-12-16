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
import { setupSocket } from './socket/handlers.js';

// Load environment variables
dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.FRONTEND_URL || '*',
    methods: ['GET', 'POST']
  }
});

// Middleware
app.use(cors({
  origin: process.env.FRONTEND_URL || '*',
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

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date(),
    mongodb: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected'
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

// Socket.IO setup
setupSocket(io);

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
