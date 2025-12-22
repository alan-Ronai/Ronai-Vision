import express from 'express';
import mongoose from 'mongoose';
import Event from '../models/Event.js';

const router = express.Router();

// In-memory event store for when MongoDB is not available
const inMemoryEvents = [];
const MAX_INMEMORY_EVENTS = 200;

/**
 * Check if MongoDB is connected
 */
function isMongoConnected() {
  return mongoose.connection.readyState === 1;
}

/**
 * Add event to in-memory store
 */
function addInMemoryEvent(event) {
  inMemoryEvents.unshift({
    ...event,
    _id: `local-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    createdAt: new Date(),
    updatedAt: new Date()
  });
  // Keep only the most recent events
  if (inMemoryEvents.length > MAX_INMEMORY_EVENTS) {
    inMemoryEvents.pop();
  }
}

/**
 * GET /api/events
 * Get events with pagination and filtering
 */
router.get('/', async (req, res) => {
  try {
    const {
      limit = 50,
      offset = 0,
      type,
      severity,
      cameraId,
      acknowledged,
      startDate,
      endDate
    } = req.query;

    // Use in-memory store if MongoDB not available
    if (!isMongoConnected()) {
      let filtered = [...inMemoryEvents];
      if (type) filtered = filtered.filter(e => e.type === type);
      if (severity) filtered = filtered.filter(e => e.severity === severity);
      if (cameraId) filtered = filtered.filter(e => e.cameraId === cameraId);
      if (acknowledged !== undefined) {
        const ack = acknowledged === 'true';
        filtered = filtered.filter(e => e.acknowledged === ack);
      }
      const paginated = filtered.slice(parseInt(offset), parseInt(offset) + parseInt(limit));
      return res.json({
        events: paginated,
        total: filtered.length,
        limit: parseInt(limit),
        offset: parseInt(offset),
        storage: 'in-memory'
      });
    }

    const query = {};

    if (type) query.type = type;
    if (severity) query.severity = severity;
    if (cameraId) query.cameraId = cameraId;
    if (acknowledged !== undefined) query.acknowledged = acknowledged === 'true';

    if (startDate || endDate) {
      query.createdAt = {};
      if (startDate) query.createdAt.$gte = new Date(startDate);
      if (endDate) query.createdAt.$lte = new Date(endDate);
    }

    const [events, total] = await Promise.all([
      Event.find(query)
        .sort({ createdAt: -1 })
        .skip(parseInt(offset))
        .limit(parseInt(limit))
        .lean(),
      Event.countDocuments(query)
    ]);

    res.json({
      events,
      total,
      limit: parseInt(limit),
      offset: parseInt(offset)
    });
  } catch (error) {
    console.error('Error fetching events:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/events/recent
 * Get recent events (simplified endpoint)
 */
router.get('/recent', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 50;

    // Use in-memory store if MongoDB not available
    if (!isMongoConnected()) {
      return res.json(inMemoryEvents.slice(0, limit));
    }

    const events = await Event.getRecent(limit);
    res.json(events);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/events/critical
 * Get unacknowledged critical events
 */
router.get('/critical', async (req, res) => {
  try {
    // Use in-memory store if MongoDB not available
    if (!isMongoConnected()) {
      const critical = inMemoryEvents.filter(e => e.severity === 'critical' && !e.acknowledged);
      return res.json(critical);
    }

    const events = await Event.getUnacknowledgedCritical();
    res.json(events);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/events/:id
 * Get single event by ID
 */
router.get('/:id', async (req, res) => {
  try {
    const event = await Event.findById(req.params.id).lean();
    if (!event) {
      return res.status(404).json({ error: 'Event not found' });
    }
    res.json(event);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/events
 * Create new event (called by AI service or internal)
 */
router.post('/', async (req, res) => {
  try {
    // Default title based on event type if not provided
    const defaultTitles = {
      detection: 'זיהוי חדש',
      alert: 'התראה',
      system: 'הודעת מערכת',
      radio: 'תעתיק קשר',
      simulation: 'סימולציה',
      upload: 'קובץ הועלה',
      armed_person: 'אדם חמוש',
      vehicle: 'רכב',
      suspicious_activity: 'פעילות חשודה',
      video: 'הקלטה נשמרה'
    };

    const eventData = {
      ...req.body,
      // Ensure title is always set - use default based on type if not provided
      title: req.body.title || defaultTitles[req.body.type] || 'אירוע',
      timestamp: new Date()
    };

    let savedEvent = eventData;

    // Try to save to MongoDB if connected
    if (isMongoConnected()) {
      try {
        const event = new Event(eventData);
        savedEvent = await event.save();
      } catch (dbError) {
        console.warn('⚠️  MongoDB save failed:', dbError.message);
        addInMemoryEvent(eventData);
        savedEvent = inMemoryEvents[0];
      }
    } else {
      // MongoDB not available - save to in-memory store
      addInMemoryEvent(eventData);
      savedEvent = inMemoryEvents[0];
    }

    // Emit to all connected clients (works even without MongoDB)
    const io = req.app.get('io');
    io.emit('event:new', savedEvent);

    // Check if this triggers emergency mode
    if (savedEvent.severity === 'critical') {
      console.log('🚨 CRITICAL EVENT - Triggering emergency mode');
      io.emit('emergency:start', {
        eventId: savedEvent._id,
        title: savedEvent.title,
        details: savedEvent.details,
        cameraId: savedEvent.cameraId,
        timestamp: savedEvent.createdAt || savedEvent.timestamp
      });
    }

    res.status(201).json(savedEvent);
  } catch (error) {
    console.error('Error creating event:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * PATCH /api/events/:id/acknowledge
 * Acknowledge an event
 */
router.patch('/:id/acknowledge', async (req, res) => {
  try {
    const { operator } = req.body;

    const event = await Event.findByIdAndUpdate(
      req.params.id,
      {
        acknowledged: true,
        acknowledgedBy: operator || 'unknown',
        acknowledgedAt: new Date()
      },
      { new: true }
    );

    if (!event) {
      return res.status(404).json({ error: 'Event not found' });
    }

    // Notify clients
    const io = req.app.get('io');
    io.emit('event:acknowledged', {
      eventId: event._id,
      acknowledgedBy: event.acknowledgedBy,
      acknowledgedAt: event.acknowledgedAt
    });

    res.json(event);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * PATCH /api/events/:id/resolve
 * Mark event as resolved
 */
router.patch('/:id/resolve', async (req, res) => {
  try {
    const event = await Event.findByIdAndUpdate(
      req.params.id,
      {
        resolved: true,
        resolvedAt: new Date()
      },
      { new: true }
    );

    if (!event) {
      return res.status(404).json({ error: 'Event not found' });
    }

    // Notify clients
    const io = req.app.get('io');
    io.emit('event:resolved', {
      eventId: event._id,
      resolvedAt: event.resolvedAt
    });

    res.json(event);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/events/:id/notes
 * Add note to event
 */
router.post('/:id/notes', async (req, res) => {
  try {
    const { note } = req.body;

    const event = await Event.findByIdAndUpdate(
      req.params.id,
      { $push: { notes: note } },
      { new: true }
    );

    if (!event) {
      return res.status(404).json({ error: 'Event not found' });
    }

    res.json(event);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * DELETE /api/events/:id
 * Delete an event
 */
router.delete('/:id', async (req, res) => {
  try {
    const event = await Event.findByIdAndDelete(req.params.id);
    if (!event) {
      return res.status(404).json({ error: 'Event not found' });
    }
    res.json({ message: 'Event deleted', eventId: req.params.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/events/stats/summary
 * Get event statistics
 */
router.get('/stats/summary', async (req, res) => {
  try {
    const now = new Date();
    const oneDayAgo = new Date(now - 24 * 60 * 60 * 1000);
    const oneWeekAgo = new Date(now - 7 * 24 * 60 * 60 * 1000);

    const [
      totalToday,
      criticalToday,
      unacknowledged,
      byType,
      bySeverity
    ] = await Promise.all([
      Event.countDocuments({ createdAt: { $gte: oneDayAgo } }),
      Event.countDocuments({ createdAt: { $gte: oneDayAgo }, severity: 'critical' }),
      Event.countDocuments({ acknowledged: false }),
      Event.aggregate([
        { $match: { createdAt: { $gte: oneWeekAgo } } },
        { $group: { _id: '$type', count: { $sum: 1 } } }
      ]),
      Event.aggregate([
        { $match: { createdAt: { $gte: oneWeekAgo } } },
        { $group: { _id: '$severity', count: { $sum: 1 } } }
      ])
    ]);

    res.json({
      totalToday,
      criticalToday,
      unacknowledged,
      byType: Object.fromEntries(byType.map(x => [x._id, x.count])),
      bySeverity: Object.fromEntries(bySeverity.map(x => [x._id, x.count]))
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
