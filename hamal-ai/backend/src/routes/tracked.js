import express from 'express';
import TrackedObject from '../models/TrackedObject.js';

const router = express.Router();

/**
 * GET /api/tracked
 * Get tracked objects with filtering
 */
router.get('/', async (req, res) => {
  try {
    const {
      limit = 50,
      type,
      status = 'active',
      armed,
      cameraId
    } = req.query;

    const query = {};

    if (type) query.type = type;
    if (status) query.status = status;
    if (armed !== undefined) query['analysis.armed'] = armed === 'true';
    if (cameraId) {
      query.$or = [
        { cameraId },
        { 'appearances.cameraId': cameraId }
      ];
    }

    const objects = await TrackedObject.find(query)
      .sort({ lastSeen: -1 })
      .limit(parseInt(limit))
      .lean();

    res.json(objects);
  } catch (error) {
    console.error('Error fetching tracked objects:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/tracked/threats
 * Get active threats (armed persons or suspicious objects)
 */
router.get('/threats', async (req, res) => {
  try {
    const threats = await TrackedObject.getActiveThreats();
    res.json(threats);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/tracked/armed
 * Get all armed persons
 */
router.get('/armed', async (req, res) => {
  try {
    const armed = await TrackedObject.getArmedPersons();
    res.json(armed);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/tracked/plate/:plate
 * Find vehicle by license plate
 */
router.get('/plate/:plate', async (req, res) => {
  try {
    const vehicle = await TrackedObject.findByPlate(req.params.plate);
    if (!vehicle) {
      return res.status(404).json({ error: 'Vehicle not found' });
    }
    res.json(vehicle);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/tracked/:trackId
 * Get single tracked object by trackId
 */
router.get('/:trackId', async (req, res) => {
  try {
    const obj = await TrackedObject.findOne({ trackId: req.params.trackId }).lean();
    if (!obj) {
      return res.status(404).json({ error: 'Tracked object not found' });
    }
    res.json(obj);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/tracked
 * Create or update tracked object (called by AI service)
 */
router.post('/', async (req, res) => {
  try {
    const { trackId, type, cameraId, analysis, bbox, confidence } = req.body;

    // Upsert - create if not exists, update if exists
    let tracked = await TrackedObject.findOne({ trackId });

    if (tracked) {
      // Update existing
      tracked.lastSeen = new Date();

      // Update analysis if provided
      if (analysis) {
        tracked.analysis = { ...tracked.analysis, ...analysis };
      }

      // Record appearance
      if (cameraId || bbox) {
        await tracked.recordAppearance(cameraId, bbox, confidence);
      } else {
        await tracked.save();
      }
    } else {
      // Create new
      tracked = new TrackedObject({
        trackId,
        type,
        cameraId,
        analysis,
        firstSeen: new Date(),
        lastSeen: new Date(),
        appearances: bbox ? [{
          cameraId,
          bbox,
          confidence,
          timestamp: new Date()
        }] : []
      });
      await tracked.save();
    }

    // Emit to clients
    const io = req.app.get('io');
    io.emit('tracked:update', tracked.toObject());

    // Check if this is a new armed person
    if (analysis?.armed && !tracked.alertTriggered) {
      tracked.alertTriggered = true;
      await tracked.save();

      io.emit('tracked:armed', {
        trackId: tracked.trackId,
        type: tracked.type,
        analysis: tracked.analysis,
        cameraId: tracked.cameraId
      });
    }

    res.status(201).json(tracked);
  } catch (error) {
    console.error('Error saving tracked object:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * PATCH /api/tracked/:trackId/status
 * Update tracked object status
 */
router.patch('/:trackId/status', async (req, res) => {
  try {
    const { status } = req.body;

    const tracked = await TrackedObject.findOneAndUpdate(
      { trackId: req.params.trackId },
      { status },
      { new: true }
    );

    if (!tracked) {
      return res.status(404).json({ error: 'Tracked object not found' });
    }

    // Notify clients
    const io = req.app.get('io');
    io.emit('tracked:status', {
      trackId: tracked.trackId,
      status: tracked.status
    });

    res.json(tracked);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/tracked/:trackId/note
 * Add note to tracked object
 */
router.post('/:trackId/note', async (req, res) => {
  try {
    const { note } = req.body;

    const tracked = await TrackedObject.findOneAndUpdate(
      { trackId: req.params.trackId },
      { $push: { notes: note } },
      { new: true }
    );

    if (!tracked) {
      return res.status(404).json({ error: 'Tracked object not found' });
    }

    res.json(tracked);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/tracked/camera/:cameraId
 * Get all tracked objects by camera
 */
router.get('/camera/:cameraId', async (req, res) => {
  try {
    const objects = await TrackedObject.getByCameraId(req.params.cameraId);
    res.json(objects);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/tracked/stats/summary
 * Get tracking statistics
 */
router.get('/stats/summary', async (req, res) => {
  try {
    const [
      activeVehicles,
      activePersons,
      armedPersons,
      totalTracked
    ] = await Promise.all([
      TrackedObject.countDocuments({ type: 'vehicle', status: 'active' }),
      TrackedObject.countDocuments({ type: 'person', status: 'active' }),
      TrackedObject.countDocuments({ type: 'person', 'analysis.armed': true, status: 'active' }),
      TrackedObject.countDocuments()
    ]);

    res.json({
      activeVehicles,
      activePersons,
      armedPersons,
      totalTracked
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
