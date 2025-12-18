import express from 'express';
import mongoose from 'mongoose';
import fetch from 'node-fetch';
import Event from '../models/Event.js';

const router = express.Router();

// AI Service URL for transmission proxy
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

/**
 * Check if MongoDB is connected
 */
function isMongoConnected() {
  return mongoose.connection.readyState === 1;
}

// Store recent transcriptions in memory for quick access
const transcriptionBuffer = [];
const MAX_BUFFER_SIZE = 100;

/**
 * POST /api/radio/transcription
 * Receive transcription from radio service
 */
router.post('/transcription', async (req, res) => {
  try {
    const { text, timestamp, source, confidence } = req.body;

    if (!text || text.trim().length === 0) {
      return res.status(400).json({ error: 'No transcription text provided' });
    }

    const transcription = {
      text: text.trim(),
      timestamp: timestamp || new Date(),
      source: source || 'radio',
      confidence: confidence || null
    };

    // Add to buffer
    transcriptionBuffer.unshift(transcription);
    if (transcriptionBuffer.length > MAX_BUFFER_SIZE) {
      transcriptionBuffer.pop();
    }

    // Emit to all connected clients
    const io = req.app.get('io');
    io.emit('radio:transcription', transcription);

    // Check for command keywords
    const commands = checkForCommands(text);
    if (commands.length > 0) {
      for (const command of commands) {
        // Create simulation event (only if MongoDB is connected)
        let eventData = {
          type: 'simulation',
          severity: command.severity || 'info',
          title: command.title,
          source: 'radio',
          details: {
            simulation: command.type,
            transcription: text,
            metadata: { detectedCommand: command.keyword }
          }
        };

        if (isMongoConnected()) {
          try {
            const event = new Event(eventData);
            await event.save();
            eventData._id = event._id;
          } catch (dbErr) {
            console.warn('Could not save event to DB:', dbErr.message);
          }
        }

        // Emit events (always, even without DB)
        io.emit('event:new', eventData);
        io.emit('command:detected', {
          command: command.type,
          transcription: text,
          eventId: eventData._id
        });
      }
    }

    // Also save as radio event if it's a significant transmission (only if MongoDB connected)
    if (text.length > 10 && isMongoConnected()) {
      try {
        const event = new Event({
          type: 'radio',
          severity: 'info',
          title: 'תמלול קשר',
          source: source || 'radio',
          details: {
            transcription: text,
            metadata: { confidence }
          }
        });
        await event.save();
      } catch (dbErr) {
        console.warn('Could not save radio event to DB:', dbErr.message);
      }
    }

    res.json({
      message: 'Transcription received',
      transcription,
      commands
    });
  } catch (error) {
    console.error('Radio transcription error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/radio/transcriptions
 * Get recent transcriptions
 */
router.get('/transcriptions', (req, res) => {
  const limit = parseInt(req.query.limit) || 50;
  res.json({
    transcriptions: transcriptionBuffer.slice(0, limit)
  });
});

/**
 * POST /api/radio/command
 * Manually trigger a command (for testing/demo)
 */
router.post('/command', async (req, res) => {
  try {
    const { command, params } = req.body;

    const commandInfo = getCommandInfo(command);
    if (!commandInfo) {
      return res.status(400).json({ error: 'Unknown command' });
    }

    // Create event data
    let eventData = {
      type: 'simulation',
      severity: commandInfo.severity,
      title: commandInfo.title,
      source: 'manual',
      details: {
        simulation: command,
        metadata: params
      }
    };

    // Save to MongoDB if connected
    if (isMongoConnected()) {
      try {
        const event = new Event(eventData);
        await event.save();
        eventData = event.toObject();
      } catch (dbErr) {
        console.warn('Could not save command event to DB:', dbErr.message);
      }
    }

    // Emit events (always, even without DB)
    const io = req.app.get('io');
    io.emit('event:new', eventData);
    io.emit('command:executed', {
      command,
      params,
      eventId: eventData._id
    });

    res.json({
      message: 'Command executed',
      command,
      event: eventData
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/radio/commands
 * List available commands
 */
router.get('/commands', (req, res) => {
  res.json({
    commands: [
      { type: 'drone_dispatch', title: 'הקפצת רחפן', keywords: ['רחפן', 'חוזי', 'הקפיצו'] },
      { type: 'phone_call', title: 'חיוג למפקד', keywords: ['התקשרו', 'חייגו', 'טלפון'] },
      { type: 'pa_announcement', title: 'כריזה', keywords: ['כריזה', 'הודעה', 'כרזו'] },
      { type: 'code_broadcast', title: 'שידור קוד', keywords: ['צפרדע', 'קוד', 'שדרו'] },
      { type: 'threat_neutralized', title: 'סוף אירוע', keywords: ['חדל', 'סיום', 'נוטרל'] }
    ]
  });
});

/**
 * Check transcription for command keywords
 */
function checkForCommands(text) {
  const commands = [];
  const lowerText = text.toLowerCase();

  // Drone dispatch
  if (lowerText.includes('רחפן') || lowerText.includes('חוזי') || lowerText.includes('הקפיצו')) {
    commands.push({
      type: 'drone_dispatch',
      title: 'רחפן הוקפץ',
      keyword: 'רחפן',
      severity: 'warning'
    });
  }

  // Code broadcast (צפרדע)
  if (lowerText.includes('צפרדע')) {
    commands.push({
      type: 'code_broadcast',
      title: 'שידור קוד צפרדע',
      keyword: 'צפרדע',
      severity: 'critical'
    });
  }

  // Phone call
  if (lowerText.includes('התקשרו') || lowerText.includes('חייגו')) {
    commands.push({
      type: 'phone_call',
      title: 'חיוג למפקד תורן',
      keyword: 'התקשרו',
      severity: 'warning'
    });
  }

  // PA announcement
  if (lowerText.includes('כריזה') || lowerText.includes('כרזו')) {
    commands.push({
      type: 'pa_announcement',
      title: 'כריזה למגורים',
      keyword: 'כריזה',
      severity: 'warning'
    });
  }

  // Threat neutralized
  if (lowerText.includes('חדל') || lowerText.includes('נוטרל') || lowerText.includes('סיום')) {
    commands.push({
      type: 'threat_neutralized',
      title: 'חדל - סוף אירוע',
      keyword: 'חדל',
      severity: 'info'
    });
  }

  return commands;
}

/**
 * Get command info by type
 */
function getCommandInfo(type) {
  const commands = {
    drone_dispatch: { title: 'רחפן הוקפץ', severity: 'warning' },
    phone_call: { title: 'חיוג למפקד תורן', severity: 'warning' },
    pa_announcement: { title: 'כריזה למגורים', severity: 'warning' },
    code_broadcast: { title: 'שידור קוד צפרדע', severity: 'critical' },
    threat_neutralized: { title: 'חדל - סוף אירוע', severity: 'info' }
  };
  return commands[type];
}

// ============== TRANSMISSION PROXY ROUTES ==============

/**
 * POST /api/radio/transmit/audio
 * Proxy to AI service for audio transmission
 */
router.post('/transmit/audio', async (req, res) => {
  try {
    const response = await fetch(`${AI_SERVICE_URL}/api/radio/transmit/audio`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });

    const data = await response.json();

    // Emit transmission event
    const io = req.app.get('io');
    io.emit('radio:transmission', {
      type: 'audio',
      success: data.success,
      timestamp: new Date()
    });

    res.status(response.status).json(data);
  } catch (error) {
    console.error('Transmission proxy error:', error);
    res.status(503).json({ error: 'AI service unavailable', message: error.message });
  }
});

/**
 * POST /api/radio/transmit/ptt
 * Proxy to AI service for PTT signaling
 */
router.post('/transmit/ptt', async (req, res) => {
  try {
    const response = await fetch(`${AI_SERVICE_URL}/api/radio/transmit/ptt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });

    const data = await response.json();

    // Emit PTT event
    const io = req.app.get('io');
    io.emit('radio:ptt', {
      state: req.body.state,
      success: data.success,
      timestamp: new Date()
    });

    res.status(response.status).json(data);
  } catch (error) {
    console.error('PTT proxy error:', error);
    res.status(503).json({ error: 'AI service unavailable', message: error.message });
  }
});

/**
 * GET /api/radio/transmit/stats
 * Proxy to AI service for transmission stats
 */
router.get('/transmit/stats', async (req, res) => {
  try {
    const response = await fetch(`${AI_SERVICE_URL}/api/radio/transmit/stats`);
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error('Stats proxy error:', error);
    res.status(503).json({ error: 'AI service unavailable', message: error.message });
  }
});

/**
 * POST /api/radio/transmit/connect
 * Proxy to AI service to connect TX
 */
router.post('/transmit/connect', async (req, res) => {
  try {
    const response = await fetch(`${AI_SERVICE_URL}/api/radio/transmit/connect`, {
      method: 'POST'
    });
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error('Connect proxy error:', error);
    res.status(503).json({ error: 'AI service unavailable', message: error.message });
  }
});

/**
 * POST /api/radio/transmit/disconnect
 * Proxy to AI service to disconnect TX
 */
router.post('/transmit/disconnect', async (req, res) => {
  try {
    const response = await fetch(`${AI_SERVICE_URL}/api/radio/transmit/disconnect`, {
      method: 'POST'
    });
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error('Disconnect proxy error:', error);
    res.status(503).json({ error: 'AI service unavailable', message: error.message });
  }
});

export default router;
