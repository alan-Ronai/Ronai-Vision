import express from 'express';
import Event from '../models/Event.js';

const router = express.Router();

// Check if MongoDB is enabled
const USE_MONGODB = process.env.USE_MONGODB === 'true';

/**
 * POST /api/alerts
 * Receive emergency alert from AI service
 */
router.post('/', async (req, res) => {
  try {
    const alertData = req.body;
    console.log('🚨 ALERT RECEIVED:', JSON.stringify(alertData, null, 2));

    let event = null;
    let eventId = null;

    // Save to MongoDB only if enabled
    if (USE_MONGODB) {
      event = new Event({
        type: 'alert',
        severity: 'critical',
        source: alertData.camera_id,
        cameraId: alertData.camera_id,
        title: `חדירה ודאית - ${alertData.armed_count || 1} חשודים חמושים!`,
        details: {
          ...alertData,
          armed: true,
          triggeredAt: new Date()
        }
      });

      await event.save();
      eventId = event._id;
    } else {
      // Local mode - generate fake ID and log
      eventId = `local-${Date.now()}`;
      console.log('📝 [LOCAL MODE] Alert logged (not saved to DB)');
    }

    // Emit emergency to all clients
    const io = req.app.get('io');

    // Send emergency start event
    io.emit('emergency:start', {
      triggeredAt: new Date(),
      cameraId: alertData.camera_id,
      personCount: alertData.person_count,
      armedCount: alertData.armed_count,
      weaponType: alertData.weapon_type,
      acknowledged: false,
      eventId: eventId
    });

    // Also send as new event for event log
    if (event) {
      io.emit('event:new', event.toObject());
    } else {
      // Send mock event for local mode
      io.emit('event:new', {
        _id: eventId,
        type: 'alert',
        severity: 'critical',
        cameraId: alertData.camera_id,
        title: `חדירה ודאית - ${alertData.armed_count || 1} חשודים חמושים!`,
        createdAt: new Date(),
        details: alertData
      });
    }

    res.status(201).json({
      message: 'Alert received and broadcasted',
      eventId: eventId,
      alert: alertData,
      localMode: !USE_MONGODB
    });

  } catch (error) {
    console.error('Alert error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/alerts/acknowledge
 * Acknowledge an alert
 */
router.post('/acknowledge', async (req, res) => {
  try {
    const { eventId, operator } = req.body;

    // Update MongoDB only if enabled
    if (USE_MONGODB && eventId && !eventId.startsWith('local-')) {
      await Event.findByIdAndUpdate(eventId, {
        acknowledged: true,
        acknowledgedBy: operator || 'console-operator',
        acknowledgedAt: new Date()
      });
    } else {
      console.log('📝 [LOCAL MODE] Alert acknowledged (not saved to DB)');
    }

    const io = req.app.get('io');
    io.emit('emergency:acknowledged', {
      eventId,
      operator: operator || 'console-operator',
      acknowledgedAt: new Date()
    });

    res.json({
      message: 'Alert acknowledged',
      localMode: !USE_MONGODB
    });

  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/alerts/end
 * End emergency mode
 */
router.post('/end', async (req, res) => {
  try {
    const { eventId, operator, reason } = req.body;

    let endEvent = null;

    // Update MongoDB only if enabled
    if (USE_MONGODB) {
      if (eventId && !eventId.startsWith('local-')) {
        await Event.findByIdAndUpdate(eventId, {
          resolved: true,
          resolvedBy: operator || 'console-operator',
          resolvedAt: new Date(),
          resolutionReason: reason
        });
      }

      // Create end event
      endEvent = new Event({
        type: 'simulation',
        severity: 'info',
        title: 'חדל - סוף אירוע',
        details: {
          simulation: 'threat_neutralized',
          originalEventId: eventId,
          resolvedBy: operator,
          reason
        }
      });
      await endEvent.save();
    } else {
      console.log('📝 [LOCAL MODE] Alert ended (not saved to DB)');
    }

    const io = req.app.get('io');

    // End emergency
    io.emit('emergency:end', {
      eventId,
      operator: operator || 'console-operator',
      endedAt: new Date(),
      reason
    });

    // Send end event
    if (endEvent) {
      io.emit('event:new', endEvent.toObject());
    } else {
      // Send mock event for local mode
      io.emit('event:new', {
        _id: `local-end-${Date.now()}`,
        type: 'simulation',
        severity: 'info',
        title: 'חדל - סוף אירוע',
        createdAt: new Date(),
        details: {
          simulation: 'threat_neutralized',
          originalEventId: eventId,
          resolvedBy: operator,
          reason
        }
      });
    }

    res.json({
      message: 'Emergency ended',
      localMode: !USE_MONGODB
    });

  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/alerts/active
 * Get active (unresolved) alerts
 */
router.get('/active', async (req, res) => {
  try {
    let alerts = [];

    // Query MongoDB only if enabled
    if (USE_MONGODB) {
      alerts = await Event.find({
        type: 'alert',
        severity: 'critical',
        resolved: { $ne: true }
      })
      .sort({ createdAt: -1 })
      .limit(10)
      .lean();
    } else {
      // Local mode - return empty array
      // Could implement in-memory storage if needed
      console.log('📝 [LOCAL MODE] No active alerts (not using DB)');
    }

    res.json({
      alerts,
      localMode: !USE_MONGODB
    });

  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
