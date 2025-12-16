import express from 'express';
import Event from '../models/Event.js';

const router = express.Router();

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
        // Create simulation event
        const event = new Event({
          type: 'simulation',
          severity: command.severity || 'info',
          title: command.title,
          source: 'radio',
          details: {
            simulation: command.type,
            transcription: text,
            metadata: { detectedCommand: command.keyword }
          }
        });
        await event.save();

        // Emit events
        io.emit('event:new', event.toObject());
        io.emit('command:detected', {
          command: command.type,
          transcription: text,
          eventId: event._id
        });
      }
    }

    // Also save as radio event if it's a significant transmission
    if (text.length > 10) {
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

    // Create simulation event
    const event = new Event({
      type: 'simulation',
      severity: commandInfo.severity,
      title: commandInfo.title,
      source: 'manual',
      details: {
        simulation: command,
        metadata: params
      }
    });
    await event.save();

    // Emit events
    const io = req.app.get('io');
    io.emit('event:new', event.toObject());
    io.emit('command:executed', {
      command,
      params,
      eventId: event._id
    });

    res.json({
      message: 'Command executed',
      command,
      event
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

export default router;
