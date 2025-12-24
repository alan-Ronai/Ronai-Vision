import express from 'express';
import mongoose from 'mongoose';
import TrackedObject from '../models/TrackedObject.js';
import trackedStore from '../stores/trackedStore.js';

const router = express.Router();

/**
 * Check if MongoDB is connected
 */
function isMongoConnected() {
  return mongoose.connection.readyState === 1;
}

/**
 * GET /api/tracked
 * List all tracked objects with filtering
 */
router.get('/', async (req, res) => {
  try {
    const {
      type,           // person, vehicle, other
      isActive,       // true/false
      isArmed,        // true/false (persons only)
      status,         // active, lost, archived
      cameraId,       // filter by camera
      limit = 100,
      offset = 0,
      sortBy = 'lastSeen',
      sortOrder = 'desc'
    } = req.query;

    // Use in-memory store if MongoDB not available
    if (!isMongoConnected()) {
      const filter = {
        type: type || undefined,
        isActive: isActive === undefined ? true : isActive === 'true',
        isArmed: isArmed === 'true' ? true : undefined,
      };
      const objects = trackedStore.getAll(filter);
      const paginated = objects.slice(parseInt(offset), parseInt(offset) + parseInt(limit));
      return res.json({
        objects: paginated,
        total: objects.length,
        limit: parseInt(limit),
        offset: parseInt(offset),
        storage: 'in-memory'
      });
    }

    const query = {};
    if (type) query.type = type;
    if (isActive !== undefined) query.isActive = isActive === 'true';
    if (isArmed !== undefined) {
      query.$or = [
        { isArmed: isArmed === 'true' },
        { 'analysis.armed': isArmed === 'true' }
      ];
    }
    if (status) query.status = status;
    if (cameraId) {
      query.$or = [
        { cameraId },
        { 'appearances.cameraId': cameraId }
      ];
    }

    const [objects, total] = await Promise.all([
      TrackedObject.find(query)
        .sort({ [sortBy]: sortOrder === 'desc' ? -1 : 1 })
        .skip(parseInt(offset))
        .limit(parseInt(limit))
        .lean(),
      TrackedObject.countDocuments(query)
    ]);

    res.json({
      objects,
      total,
      limit: parseInt(limit),
      offset: parseInt(offset),
      storage: 'mongodb'
    });
  } catch (error) {
    console.error('Error fetching tracked objects:', error.message);
    // Fallback to in-memory on error
    try {
      const objects = trackedStore.getAll({ isActive: true });
      res.json({
        objects: objects.slice(0, 100),
        total: objects.length,
        limit: 100,
        offset: 0,
        storage: 'in-memory-fallback'
      });
    } catch (e) {
      res.status(500).json({ error: error.message });
    }
  }
});

/**
 * GET /api/tracked/stats
 * Get tracking statistics
 */
router.get('/stats', async (req, res) => {
  try {
    // Use in-memory store if MongoDB not available
    if (!isMongoConnected()) {
      return res.json(trackedStore.getStats());
    }

    const [
      totalPersons,
      totalVehicles,
      activePersons,
      activeVehicles,
      armedPersons,
      recentAppearances,
      totalThreats
    ] = await Promise.all([
      TrackedObject.countDocuments({ type: 'person' }),
      TrackedObject.countDocuments({ type: 'vehicle' }),
      TrackedObject.countDocuments({ type: 'person', isActive: true, status: 'active' }),
      TrackedObject.countDocuments({ type: 'vehicle', isActive: true, status: 'active' }),
      TrackedObject.countDocuments({
        type: 'person',
        $or: [{ isArmed: true }, { 'analysis.armed': true }],
        isActive: true
      }),
      TrackedObject.countDocuments({
        lastSeen: { $gte: new Date(Date.now() - 5 * 60 * 1000) } // Last 5 minutes
      }),
      TrackedObject.countDocuments({
        isActive: true,
        $or: [
          { isArmed: true },
          { 'analysis.armed': true },
          { 'analysis.suspicious': true },
          { threatLevel: { $in: ['high', 'critical'] } }
        ]
      })
    ]);

    res.json({
      persons: { total: totalPersons, active: activePersons, armed: armedPersons },
      vehicles: { total: totalVehicles, active: activeVehicles },
      recentlyActive: recentAppearances,
      threats: totalThreats,
      timestamp: new Date(),
      storage: 'mongodb'
    });
  } catch (error) {
    console.error('Error fetching stats:', error.message);
    // Fallback to in-memory
    try {
      res.json(trackedStore.getStats());
    } catch (e) {
      res.status(500).json({ error: error.message });
    }
  }
});

/**
 * GET /api/tracked/threats
 * Get active threats (armed persons or suspicious objects)
 */
router.get('/threats', async (req, res) => {
  try {
    if (!isMongoConnected()) {
      return res.json(trackedStore.getArmedPersons());
    }
    const threats = await TrackedObject.getActiveThreats();
    res.json(threats);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/tracked/armed
 * Get all armed persons (quick access for security)
 */
router.get('/armed', async (req, res) => {
  try {
    if (!isMongoConnected()) {
      return res.json(trackedStore.getArmedPersons());
    }
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
    if (!isMongoConnected()) {
      const results = trackedStore.search({ licensePlate: req.params.plate, type: 'vehicle' });
      if (results.length === 0) {
        return res.status(404).json({ error: 'Vehicle not found' });
      }
      return res.json(results[0]);
    }
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
 * GET /api/tracked/camera/:cameraId
 * Get all tracked objects by camera
 */
router.get('/camera/:cameraId', async (req, res) => {
  try {
    if (!isMongoConnected()) {
      const objects = trackedStore.getAll({ isActive: true });
      const filtered = objects.filter(o =>
        o.cameraId === req.params.cameraId ||
        o.appearances?.some(a => a.cameraId === req.params.cameraId)
      );
      return res.json(filtered);
    }
    const objects = await TrackedObject.getByCamera(req.params.cameraId);
    res.json(objects);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/tracked/gid/:gid
 * Get single tracked object by Global ID
 */
router.get('/gid/:gid', async (req, res) => {
  try {
    if (!isMongoConnected()) {
      const object = trackedStore.getByGid(req.params.gid);
      if (!object) {
        return res.status(404).json({ error: 'Tracked object not found' });
      }
      return res.json(object);
    }
    const object = await TrackedObject.findByGid(req.params.gid);
    if (!object) {
      return res.status(404).json({ error: 'Tracked object not found' });
    }
    res.json(object);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/tracked/:trackId
 * Get single tracked object by trackId (legacy support)
 */
router.get('/:trackId', async (req, res) => {
  try {
    const gid = parseInt(req.params.trackId);

    if (!isMongoConnected()) {
      const object = trackedStore.getByGid(gid);
      if (!object) {
        return res.status(404).json({ error: 'Tracked object not found' });
      }
      return res.json(object);
    }

    // Try to find by gid first (if numeric)
    let obj;
    if (!isNaN(gid)) {
      obj = await TrackedObject.findOne({ gid }).lean();
    }

    // Fall back to trackId
    if (!obj) {
      obj = await TrackedObject.findOne({ trackId: req.params.trackId }).lean();
    }

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
    const { gid, trackId, type, ...data } = req.body;

    if (!gid && !trackId) {
      return res.status(400).json({ error: 'gid or trackId is required' });
    }
    if (!type) {
      return res.status(400).json({ error: 'type is required' });
    }

    const io = req.app.get('io');

    // Use in-memory store if MongoDB not available
    if (!isMongoConnected()) {
      const object = trackedStore.upsert({
        gid: gid || parseInt(trackId) || Date.now(),
        type,
        ...data
      });

      if (io) {
        io.emit('tracked:update', object);
        if (object.isArmed || data.analysis?.armed) {
          io.emit('tracked:armed', {
            gid: object.gid,
            type: object.type,
            analysis: object.analysis,
            cameraId: object.cameraId
          });
        }
      }

      return res.status(201).json(object);
    }

    // Build update object
    const updateData = {
      type,
      lastSeen: new Date(),
      ...data
    };

    // Sync armed status from analysis
    if (data.analysis?.armed !== undefined) {
      updateData.isArmed = data.analysis.armed;
    }
    if (data.isArmed !== undefined) {
      updateData['analysis.armed'] = data.isArmed;
    }

    // Upsert - create or update
    // IMPORTANT: Query by BOTH gid AND type to prevent collisions
    // (vehicle GID 1 and person GID 1 are different objects)
    const query = gid ? { gid, type } : { trackId };
    const object = await TrackedObject.findOneAndUpdate(
      query,
      {
        $set: updateData,
        $setOnInsert: {
          firstSeen: new Date(),
          gid: gid || Date.now(),
          trackId: trackId || `track_${gid}`
        }
      },
      { upsert: true, new: true, setDefaultsOnInsert: true }
    );

    // Emit socket event for real-time updates
    if (io) {
      io.emit('tracked:update', object.toObject ? object.toObject() : object);

      // Emit armed alert if newly armed
      if (updateData.isArmed || data.analysis?.armed) {
        io.emit('tracked:armed', {
          gid: object.gid,
          trackId: object.trackId,
          type: object.type,
          analysis: object.analysis,
          cameraId: object.cameraId
        });
      }
    }

    res.status(201).json(object);
  } catch (error) {
    console.error('Error creating/updating tracked object:', error.message);
    // Fallback to in-memory
    try {
      const { gid, trackId, type, ...data } = req.body;
      const object = trackedStore.upsert({
        gid: gid || parseInt(trackId) || Date.now(),
        type,
        ...data
      });
      res.status(201).json(object);
    } catch (e) {
      res.status(500).json({ error: error.message });
    }
  }
});

/**
 * POST /api/tracked/:gid/appearance
 * Add new appearance to tracked object
 */
router.post('/:gid/appearance', async (req, res) => {
  try {
    const gid = parseInt(req.params.gid);
    const io = req.app.get('io');

    if (!isMongoConnected()) {
      const object = trackedStore.addAppearance(gid, req.body);
      if (!object) {
        return res.status(404).json({ error: 'Tracked object not found' });
      }
      if (io) {
        io.emit('tracked:appearance', { gid, appearance: req.body });
      }
      return res.json(object);
    }

    let object = await TrackedObject.findOne({ gid });

    // Fall back to trackId
    if (!object) {
      object = await TrackedObject.findOne({ trackId: req.params.gid });
    }

    if (!object) {
      return res.status(404).json({ error: 'Tracked object not found' });
    }

    const appearance = {
      ...req.body,
      timestamp: req.body.timestamp || new Date()
    };

    await object.addAppearance(appearance);

    // Emit update
    if (io) {
      io.emit('tracked:appearance', { gid: object.gid, appearance });
    }

    res.json(object);
  } catch (error) {
    console.error('Error adding appearance:', error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * PATCH /api/tracked/:gid
 * Update tracked object metadata
 */
router.patch('/:gid', async (req, res) => {
  try {
    const gid = parseInt(req.params.gid);
    const io = req.app.get('io');

    if (!isMongoConnected()) {
      const existing = trackedStore.getByGid(gid);
      if (!existing) {
        return res.status(404).json({ error: 'Tracked object not found' });
      }
      const object = trackedStore.upsert({ ...existing, ...req.body, gid });
      if (io) {
        io.emit('tracked:update', object);
      }
      return res.json(object);
    }

    let query = { gid };

    // Fall back to trackId if not numeric
    if (isNaN(gid)) {
      query = { trackId: req.params.gid };
    }

    const object = await TrackedObject.findOneAndUpdate(
      query,
      { $set: { ...req.body, lastSeen: new Date() } },
      { new: true }
    );

    if (!object) {
      return res.status(404).json({ error: 'Tracked object not found' });
    }

    if (io) {
      io.emit('tracked:update', object.toObject ? object.toObject() : object);
    }

    res.json(object);
  } catch (error) {
    console.error('Error updating tracked object:', error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * PATCH /api/tracked/:gid/analysis
 * Update Gemini analysis results (auto-creates object if not exists)
 */
router.patch('/:gid/analysis', async (req, res) => {
  try {
    const gid = parseInt(req.params.gid);
    const io = req.app.get('io');

    // Determine type from analysis content or request body
    const inferType = () => {
      if (req.body.type) return req.body.type;
      // Infer type from analysis fields
      if (req.body.licensePlate || req.body.make || req.body.model || req.body.vehicleType) {
        return 'vehicle';
      }
      return 'person'; // Default to person
    };

    if (!isMongoConnected()) {
      let object = trackedStore.getByGid(gid);

      // Auto-create if doesn't exist
      if (!object) {
        object = trackedStore.upsert({
          gid,
          type: inferType(),
          isActive: true,
          status: 'active'
        });
      }

      // Handle type correction from Gemini
      const typeCorrection = req.body._typeCorrection;
      if (typeCorrection && typeCorrection.correctedType && object.type !== typeCorrection.correctedType) {
        console.log(`[Tracked] Type correction for GID ${gid}: ${typeCorrection.originalType} -> ${typeCorrection.correctedType}`);
        object = trackedStore.changeType(gid, object.type, typeCorrection.correctedType);
        if (!object) {
          return res.status(404).json({ error: 'Failed to change object type' });
        }
        if (io) {
          io.emit('tracked:typeChanged', {
            gid: object.gid,
            oldType: typeCorrection.originalType,
            newType: typeCorrection.correctedType,
            reason: typeCorrection.reason
          });
        }
      }

      object = trackedStore.updateAnalysis(gid, req.body, object.type);
      if (io) {
        io.emit('tracked:analysis', { gid, analysis: object.analysis });
        if (req.body.armed) {
          io.emit('tracked:armed', {
            gid: object.gid,
            type: object.type,
            analysis: object.analysis,
            cameraId: object.cameraId
          });
        }
      }
      return res.json(object);
    }

    let object = await TrackedObject.findOne({ gid });

    // Fall back to trackId
    if (!object) {
      object = await TrackedObject.findOne({ trackId: req.params.gid });
    }

    // Auto-create if doesn't exist
    if (!object) {
      const type = inferType();
      object = new TrackedObject({
        gid,
        trackId: `${type === 'vehicle' ? 'v' : 't'}_${gid}`,
        type,
        isActive: true,
        status: 'active',
        firstSeen: new Date(),
        lastSeen: new Date()
      });
      await object.save();
      console.log(`Auto-created tracked object GID ${gid} (${type})`);
    }

    // Handle type correction from Gemini (MongoDB)
    const typeCorrection = req.body._typeCorrection;
    if (typeCorrection && typeCorrection.correctedType && object.type !== typeCorrection.correctedType) {
      console.log(`[Tracked] Type correction for GID ${gid}: ${typeCorrection.originalType} -> ${typeCorrection.correctedType}`);
      const oldType = object.type;
      object.type = typeCorrection.correctedType;
      object.trackId = `${typeCorrection.correctedType === 'vehicle' ? 'v' : 't'}_${gid}`;

      // If changing to vehicle, remove armed status
      if (typeCorrection.correctedType === 'vehicle') {
        object.isArmed = false;
        if (object.analysis) {
          delete object.analysis.armed;
          delete object.analysis.חמוש;
        }
      }

      await object.save();

      if (io) {
        io.emit('tracked:typeChanged', {
          gid: object.gid,
          oldType: oldType,
          newType: typeCorrection.correctedType,
          reason: typeCorrection.reason
        });
      }
    }

    await object.updateAnalysis(req.body);

    if (io) {
      io.emit('tracked:analysis', { gid: object.gid, analysis: object.analysis });

      // Emit armed alert if analysis shows armed
      if (req.body.armed) {
        io.emit('tracked:armed', {
          gid: object.gid,
          trackId: object.trackId,
          type: object.type,
          analysis: object.analysis,
          cameraId: object.cameraId
        });
      }
    }

    res.json(object);
  } catch (error) {
    console.error('Error updating analysis:', error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * PATCH /api/tracked/:gid/status
 * Update tracked object status
 */
router.patch('/:gid/status', async (req, res) => {
  try {
    const { status, isActive } = req.body;
    const gid = parseInt(req.params.gid);
    const io = req.app.get('io');

    if (!isMongoConnected()) {
      const existing = trackedStore.getByGid(gid);
      if (!existing) {
        return res.status(404).json({ error: 'Tracked object not found' });
      }
      const object = trackedStore.upsert({
        ...existing,
        gid,
        status: status || existing.status,
        isActive: isActive !== undefined ? isActive : existing.isActive
      });
      if (io) {
        io.emit('tracked:status', {
          gid: object.gid,
          status: object.status,
          isActive: object.isActive
        });
      }
      return res.json(object);
    }

    const updateData = {};
    if (status) updateData.status = status;
    if (isActive !== undefined) updateData.isActive = isActive;

    let query = { gid };
    if (isNaN(gid)) {
      query = { trackId: req.params.gid };
    }

    const tracked = await TrackedObject.findOneAndUpdate(
      query,
      { $set: updateData },
      { new: true }
    );

    if (!tracked) {
      return res.status(404).json({ error: 'Tracked object not found' });
    }

    // Notify clients
    if (io) {
      io.emit('tracked:status', {
        gid: tracked.gid,
        trackId: tracked.trackId,
        status: tracked.status,
        isActive: tracked.isActive
      });
    }

    res.json(tracked);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * DELETE /api/tracked/:gid
 * Deactivate tracked object (soft delete)
 */
router.delete('/:gid', async (req, res) => {
  try {
    const gid = parseInt(req.params.gid);
    const io = req.app.get('io');

    if (!isMongoConnected()) {
      const object = trackedStore.deactivate(gid);
      if (!object) {
        return res.status(404).json({ error: 'Tracked object not found' });
      }
      if (io) {
        io.emit('tracked:deactivated', { gid });
      }
      return res.json({ message: 'Tracked object deactivated', object });
    }

    let query = { gid };

    if (isNaN(gid)) {
      query = { trackId: req.params.gid };
    }

    const object = await TrackedObject.findOneAndUpdate(
      query,
      { isActive: false, status: 'archived' },
      { new: true }
    );

    if (!object) {
      return res.status(404).json({ error: 'Tracked object not found' });
    }

    if (io) {
      io.emit('tracked:deactivated', { gid: object.gid, trackId: object.trackId });
    }

    res.json({ message: 'Tracked object deactivated', object });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/tracked/search
 * Search tracked objects by various criteria
 */
router.post('/search', async (req, res) => {
  try {
    const {
      query,           // Text search
      licensePlate,    // Vehicle plate search
      clothingColor,   // Person clothing search
      cameraId,        // Filter by camera
      timeRange,       // { start, end }
      type,
      isActive = true
    } = req.body;

    if (!isMongoConnected()) {
      const results = trackedStore.search({
        type,
        licensePlate,
        clothingColor,
        isActive
      });
      return res.json(results);
    }

    const searchQuery = {};
    if (isActive) searchQuery.isActive = true;
    if (type) searchQuery.type = type;

    if (licensePlate) {
      searchQuery['analysis.licensePlate'] = new RegExp(licensePlate, 'i');
    }

    if (clothingColor) {
      searchQuery.$or = [
        { 'analysis.clothingColor': new RegExp(clothingColor, 'i') },
        { 'analysis.color': new RegExp(clothingColor, 'i') }
      ];
    }

    if (cameraId) {
      searchQuery.$or = searchQuery.$or || [];
      searchQuery.$or.push(
        { cameraId },
        { 'appearances.cameraId': cameraId }
      );
    }

    if (timeRange) {
      searchQuery.lastSeen = {
        $gte: new Date(timeRange.start),
        $lte: new Date(timeRange.end)
      };
    }

    // Text search on description
    if (query) {
      searchQuery.$or = searchQuery.$or || [];
      searchQuery.$or.push(
        { 'analysis.description': new RegExp(query, 'i') },
        { notes: new RegExp(query, 'i') }
      );
    }

    const results = await TrackedObject.find(searchQuery)
      .sort({ lastSeen: -1 })
      .limit(50)
      .lean();

    res.json(results);
  } catch (error) {
    console.error('Error searching:', error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/tracked/:gid/note
 * Add note to tracked object
 */
router.post('/:gid/note', async (req, res) => {
  try {
    const { note } = req.body;
    const gid = parseInt(req.params.gid);

    if (!isMongoConnected()) {
      const existing = trackedStore.getByGid(gid);
      if (!existing) {
        return res.status(404).json({ error: 'Tracked object not found' });
      }
      const object = trackedStore.upsert({ ...existing, gid, notes: note });
      return res.json(object);
    }

    let query = { gid };
    if (isNaN(gid)) {
      query = { trackId: req.params.gid };
    }

    const tracked = await TrackedObject.findOneAndUpdate(
      query,
      { $set: { notes: note } },
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
 * POST /api/tracked/:gid/tags
 * Add tags to tracked object
 */
router.post('/:gid/tags', async (req, res) => {
  try {
    const { tags } = req.body;
    const gid = parseInt(req.params.gid);

    if (!isMongoConnected()) {
      const existing = trackedStore.getByGid(gid);
      if (!existing) {
        return res.status(404).json({ error: 'Tracked object not found' });
      }
      const newTags = Array.isArray(tags) ? tags : [tags];
      const existingTags = existing.tags || [];
      const mergedTags = [...new Set([...existingTags, ...newTags])];
      const object = trackedStore.upsert({ ...existing, gid, tags: mergedTags });
      return res.json(object);
    }

    let query = { gid };
    if (isNaN(gid)) {
      query = { trackId: req.params.gid };
    }

    const tracked = await TrackedObject.findOneAndUpdate(
      query,
      { $addToSet: { tags: { $each: Array.isArray(tags) ? tags : [tags] } } },
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
 * DELETE /api/tracked/cleanup
 * Remove old inactive tracked objects
 */
router.delete('/cleanup', async (req, res) => {
  try {
    if (!isMongoConnected()) {
      // For in-memory, just return success (data is volatile anyway)
      return res.json({
        message: 'Cleanup not needed for in-memory storage',
        deleted: 0
      });
    }

    const { olderThanHours = 24 } = req.query;
    const cutoff = new Date(Date.now() - parseInt(olderThanHours) * 60 * 60 * 1000);

    const result = await TrackedObject.deleteMany({
      isActive: false,
      lastSeen: { $lt: cutoff }
    });

    res.json({
      message: 'Cleanup completed',
      deleted: result.deletedCount
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * DELETE /api/tracked/clear-all
 * Clear ALL tracked objects (used on AI service restart)
 */
router.delete('/clear-all', async (req, res) => {
  try {
    // Clear in-memory store
    trackedStore.clear();

    // Clear MongoDB if connected
    if (isMongoConnected()) {
      await TrackedObject.deleteMany({});
    }

    console.log('[TrackedStore] All tracked objects cleared (AI service restart)');

    res.json({
      message: 'All tracked objects cleared',
      success: true
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============================================
// ReID Gallery Endpoints for Persistent Re-identification
// ============================================

/**
 * GET /api/tracked/gallery
 * Get all tracked objects with ReID features for gallery sync
 */
router.get('/gallery', async (req, res) => {
  try {
    const { type, ttlDays = 7, withFeatures = 'true' } = req.query;
    const ttl = parseInt(ttlDays);

    if (!isMongoConnected()) {
      const entries = trackedStore.getGalleryEntries({
        type: type || undefined,
        ttlDays: ttl
      });
      return res.json({ entries, count: entries.length, storage: 'in-memory' });
    }

    const entries = await TrackedObject.getGalleryEntries({
      type: type || undefined,
      ttlDays: ttl
    });

    // Format entries for gallery
    const formattedEntries = entries.map(obj => ({
      gid: obj.gid,
      type: obj.type,
      feature: obj.reidFeature?.feature,
      feature_dim: obj.reidFeature?.featureDim,
      feature_dtype: obj.reidFeature?.featureDtype || 'float32',
      last_seen: obj.reidFeature?.lastUpdated || obj.lastSeen,
      first_seen: obj.reidFeature?.firstCaptured || obj.firstSeen,
      camera_id: obj.reidFeature?.capturedCameraId,
      confidence: obj.reidFeature?.bestConfidence,
      match_count: obj.reidFeature?.matchCount || 1,
    }));

    res.json({ entries: formattedEntries, count: formattedEntries.length, storage: 'mongodb' });
  } catch (error) {
    console.error('Error getting gallery:', error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/tracked/gallery/sync
 * Sync gallery entries from AI service (batch update)
 */
router.post('/gallery/sync', async (req, res) => {
  try {
    const { entries } = req.body;

    if (!Array.isArray(entries) || entries.length === 0) {
      return res.status(400).json({ error: 'entries array required' });
    }

    let synced = 0;
    let errors = 0;

    if (!isMongoConnected()) {
      for (const entry of entries) {
        try {
          let obj = trackedStore.getByGid(entry.gid, entry.type);

          if (!obj) {
            // Auto-create object
            obj = trackedStore.upsert({
              gid: entry.gid,
              type: entry.type,
              isActive: true,
              status: 'active'
            });
          }

          trackedStore.updateFeature(entry.gid, {
            feature: entry.feature,
            feature_dim: entry.feature_dim,
            feature_dtype: entry.feature_dtype,
            confidence: entry.confidence,
            camera_id: entry.camera_id
          }, entry.type);

          synced++;
        } catch (e) {
          errors++;
          console.error(`Gallery sync error for GID ${entry.gid}:`, e.message);
        }
      }
    } else {
      for (const entry of entries) {
        try {
          let obj = await TrackedObject.findOne({ gid: entry.gid, type: entry.type });

          if (!obj) {
            // Auto-create object
            obj = new TrackedObject({
              gid: entry.gid,
              type: entry.type,
              trackId: `${entry.type === 'vehicle' ? 'v' : 't'}_${entry.gid}`,
              isActive: true,
              status: 'active',
              firstSeen: new Date(),
              lastSeen: new Date()
            });
          }

          await obj.updateFeature({
            feature: entry.feature,
            feature_dim: entry.feature_dim,
            feature_dtype: entry.feature_dtype,
            confidence: entry.confidence,
            camera_id: entry.camera_id
          });

          synced++;
        } catch (e) {
          errors++;
          console.error(`Gallery sync error for GID ${entry.gid}:`, e.message);
        }
      }
    }

    res.json({ synced, errors, total: entries.length });
  } catch (error) {
    console.error('Error syncing gallery:', error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * PATCH /api/tracked/:gid/features
 * Update ReID features for a tracked object
 */
router.patch('/:gid/features', async (req, res) => {
  try {
    const gid = parseInt(req.params.gid);
    const featureData = req.body;

    if (!featureData.feature) {
      return res.status(400).json({ error: 'feature required' });
    }

    if (!isMongoConnected()) {
      let obj = trackedStore.getByGid(gid, featureData.type);

      if (!obj) {
        // Auto-create object
        obj = trackedStore.upsert({
          gid,
          type: featureData.type || 'person',
          isActive: true,
          status: 'active'
        });
      }

      obj = trackedStore.updateFeature(gid, featureData, featureData.type);
      return res.json(obj);
    }

    let obj = await TrackedObject.findOne({ gid });

    if (!obj) {
      // Auto-create object
      const type = featureData.type || 'person';
      obj = new TrackedObject({
        gid,
        type,
        trackId: `${type === 'vehicle' ? 'v' : 't'}_${gid}`,
        isActive: true,
        status: 'active',
        firstSeen: new Date(),
        lastSeen: new Date()
      });
    }

    await obj.updateFeature(featureData);
    res.json(obj);
  } catch (error) {
    console.error('Error updating features:', error.message);
    res.status(500).json({ error: error.message });
  }
});

/**
 * DELETE /api/tracked/gallery/cleanup
 * Remove expired gallery entries
 */
router.delete('/gallery/cleanup', async (req, res) => {
  try {
    const { ttlDays = 7 } = req.query;
    const ttl = parseInt(ttlDays);
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - ttl);

    if (!isMongoConnected()) {
      // For in-memory, we don't actually delete but could mark expired
      return res.json({
        message: 'Cleanup skipped for in-memory store',
        storage: 'in-memory'
      });
    }

    // Clear features for entries older than TTL
    const result = await TrackedObject.updateMany(
      { 'reidFeature.lastUpdated': { $lt: cutoffDate } },
      { $unset: { reidFeature: 1 } }
    );

    res.json({
      cleaned: result.modifiedCount,
      ttlDays: ttl,
      cutoffDate,
      storage: 'mongodb'
    });
  } catch (error) {
    console.error('Error cleaning gallery:', error.message);
    res.status(500).json({ error: error.message });
  }
});

export default router;
