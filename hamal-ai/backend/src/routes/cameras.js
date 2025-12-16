import express from 'express';
import Camera from '../models/Camera.js';

const router = express.Router();

/**
 * GET /api/cameras
 * Get all cameras
 */
router.get('/', async (req, res) => {
  try {
    const cameras = await Camera.find().sort({ order: 1 }).lean();
    res.json(cameras);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/cameras/online
 * Get online cameras only
 */
router.get('/online', async (req, res) => {
  try {
    const cameras = await Camera.getOnline();
    res.json(cameras);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/cameras/main
 * Get main camera
 */
router.get('/main', async (req, res) => {
  try {
    const camera = await Camera.getMain();
    if (!camera) {
      return res.status(404).json({ error: 'No main camera configured' });
    }
    res.json(camera);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/cameras/:id
 * Get single camera
 */
router.get('/:id', async (req, res) => {
  try {
    const camera = await Camera.findOne({ cameraId: req.params.id }).lean();
    if (!camera) {
      return res.status(404).json({ error: 'Camera not found' });
    }
    res.json(camera);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/cameras
 * Add new camera
 */
router.post('/', async (req, res) => {
  try {
    const camera = new Camera(req.body);
    await camera.save();

    // Notify clients
    const io = req.app.get('io');
    io.emit('camera:added', camera.toObject());

    res.status(201).json(camera);
  } catch (error) {
    if (error.code === 11000) {
      return res.status(400).json({ error: 'Camera ID already exists' });
    }
    res.status(500).json({ error: error.message });
  }
});

/**
 * PUT /api/cameras/:id
 * Update camera
 */
router.put('/:id', async (req, res) => {
  try {
    const camera = await Camera.findOneAndUpdate(
      { cameraId: req.params.id },
      req.body,
      { new: true, runValidators: true }
    );

    if (!camera) {
      return res.status(404).json({ error: 'Camera not found' });
    }

    // Notify clients
    const io = req.app.get('io');
    io.emit('camera:updated', camera.toObject());

    res.json(camera);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * PATCH /api/cameras/:id/status
 * Update camera status
 */
router.patch('/:id/status', async (req, res) => {
  try {
    const { status, error } = req.body;

    const camera = await Camera.findOne({ cameraId: req.params.id });
    if (!camera) {
      return res.status(404).json({ error: 'Camera not found' });
    }

    await camera.updateStatus(status, error);

    // Notify clients
    const io = req.app.get('io');
    io.emit('camera:status', {
      cameraId: camera.cameraId,
      status: camera.status,
      lastSeen: camera.lastSeen,
      error: camera.lastError
    });

    res.json(camera);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * PATCH /api/cameras/:id/main
 * Set camera as main
 */
router.patch('/:id/main', async (req, res) => {
  try {
    // Unset any current main camera
    await Camera.updateMany({}, { isMainCamera: false });

    // Set new main camera
    const camera = await Camera.findOneAndUpdate(
      { cameraId: req.params.id },
      { isMainCamera: true },
      { new: true }
    );

    if (!camera) {
      return res.status(404).json({ error: 'Camera not found' });
    }

    // Notify clients
    const io = req.app.get('io');
    io.emit('camera:main', { cameraId: camera.cameraId });

    res.json(camera);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/cameras/:id/test
 * Test camera RTSP connection
 */
router.post('/:id/test', async (req, res) => {
  try {
    let camera = await Camera.findOne({ cameraId: req.params.id });

    if (!camera) {
      // Try by MongoDB _id
      camera = await Camera.findById(req.params.id);
      if (!camera) {
        return res.status(404).json({ error: 'Camera not found' });
      }
    }

    // Update status to connecting
    camera.status = 'connecting';
    await camera.save();

    const io = req.app.get('io');
    io.emit('camera:status', {
      id: camera._id,
      cameraId: camera.cameraId,
      status: 'connecting'
    });

    // TODO: Implement actual RTSP test with FFmpeg
    // For now, simulate connection test
    setTimeout(async () => {
      try {
        // Simulate success (in production, test actual RTSP stream)
        const success = camera.rtspUrl ? true : false;

        camera.status = success ? 'online' : 'error';
        camera.lastSeen = new Date();
        if (!success) {
          camera.lastError = 'No RTSP URL configured';
        }
        await camera.save();

        io.emit('camera:status', {
          id: camera._id,
          cameraId: camera.cameraId,
          status: camera.status
        });
      } catch (e) {
        console.error('Status update failed:', e);
      }
    }, 2000);

    res.json({ message: 'Testing connection...', status: 'connecting' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * DELETE /api/cameras/:id
 * Delete camera
 */
router.delete('/:id', async (req, res) => {
  try {
    const camera = await Camera.findOneAndDelete({ cameraId: req.params.id });
    if (!camera) {
      return res.status(404).json({ error: 'Camera not found' });
    }

    // Notify clients
    const io = req.app.get('io');
    io.emit('camera:removed', { cameraId: req.params.id });

    res.json({ message: 'Camera deleted', cameraId: req.params.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/cameras/:id/snapshot
 * Store camera snapshot
 */
router.post('/:id/snapshot', async (req, res) => {
  try {
    const { thumbnailPath } = req.body;

    const camera = await Camera.findOneAndUpdate(
      { cameraId: req.params.id },
      {
        thumbnail: thumbnailPath,
        lastSeen: new Date()
      },
      { new: true }
    );

    if (!camera) {
      return res.status(404).json({ error: 'Camera not found' });
    }

    res.json({ message: 'Snapshot updated', camera });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/cameras/seed
 * Seed demo cameras (development only)
 */
router.post('/seed', async (req, res) => {
  try {
    const demoCameras = [
      {
        cameraId: 'cam-1',
        name: 'שער ראשי',
        location: 'כניסה מרכזית',
        type: 'simulator',
        status: 'online',
        isMainCamera: true,
        aiEnabled: true,
        order: 1
      },
      {
        cameraId: 'cam-2',
        name: 'חניון',
        location: 'חניון צפוני',
        type: 'simulator',
        status: 'online',
        aiEnabled: true,
        order: 2
      },
      {
        cameraId: 'cam-3',
        name: 'היקף מזרחי',
        location: 'גדר מזרח',
        type: 'simulator',
        status: 'online',
        aiEnabled: true,
        order: 3
      },
      {
        cameraId: 'cam-4',
        name: 'מגורים A',
        location: 'בניין מגורים A',
        type: 'simulator',
        status: 'online',
        aiEnabled: true,
        order: 4
      },
      {
        cameraId: 'cam-5',
        name: 'מגורים B',
        location: 'בניין מגורים B',
        type: 'simulator',
        status: 'offline',
        aiEnabled: true,
        order: 5
      },
      {
        cameraId: 'cam-6',
        name: 'שער אחורי',
        location: 'כניסה משנית',
        type: 'simulator',
        status: 'online',
        aiEnabled: true,
        order: 6
      }
    ];

    // Clear existing cameras
    await Camera.deleteMany({});

    // Insert demo cameras
    const cameras = await Camera.insertMany(demoCameras);

    res.json({
      message: 'Demo cameras seeded',
      count: cameras.length,
      cameras
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
