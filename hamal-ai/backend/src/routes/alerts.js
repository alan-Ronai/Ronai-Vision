import express from 'express';
import Event from '../models/Event.js';

const router = express.Router();

/**
 * POST /api/alerts
 * Receive emergency alert from AI service
 */
router.post('/', async (req, res) => {
  try {
    const alertData = req.body;
    console.log('🚨 ALERT RECEIVED:', JSON.stringify(alertData, null, 2));

    // Create critical event
    const event = new Event({
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
      eventId: event._id
    });

    // Also send as new event for event log
    io.emit('event:new', event.toObject());

    res.status(201).json({
      message: 'Alert received and broadcasted',
      eventId: event._id,
      alert: alertData
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

    if (eventId) {
      await Event.findByIdAndUpdate(eventId, {
        acknowledged: true,
        acknowledgedBy: operator || 'console-operator',
        acknowledgedAt: new Date()
      });
    }

    const io = req.app.get('io');
    io.emit('emergency:acknowledged', {
      eventId,
      operator: operator || 'console-operator',
      acknowledgedAt: new Date()
    });

    res.json({ message: 'Alert acknowledged' });

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

    if (eventId) {
      await Event.findByIdAndUpdate(eventId, {
        resolved: true,
        resolvedBy: operator || 'console-operator',
        resolvedAt: new Date(),
        resolutionReason: reason
      });
    }

    // Create end event
    const endEvent = new Event({
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

    const io = req.app.get('io');

    // End emergency
    io.emit('emergency:end', {
      eventId,
      operator: operator || 'console-operator',
      endedAt: new Date(),
      reason
    });

    // Send end event
    io.emit('event:new', endEvent.toObject());

    res.json({ message: 'Emergency ended' });

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
    const alerts = await Event.find({
      type: 'alert',
      severity: 'critical',
      resolved: { $ne: true }
    })
    .sort({ createdAt: -1 })
    .limit(10)
    .lean();

    res.json(alerts);

  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
