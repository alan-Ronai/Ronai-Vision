import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';
import Event from '../models/Event.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadPath = process.env.CLIPS_PATH || path.join(__dirname, '../../../data/clips');

    // Create directory if it doesn't exist
    if (!fs.existsSync(uploadPath)) {
      fs.mkdirSync(uploadPath, { recursive: true });
    }

    cb(null, uploadPath);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${uuidv4()}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB max
  },
  fileFilter: (req, file, cb) => {
    // Allow video and image files
    const allowedTypes = /jpeg|jpg|png|gif|mp4|webm|mov|avi|mkv/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);

    if (extname && mimetype) {
      return cb(null, true);
    }
    cb(new Error('Only video and image files are allowed'));
  }
});

/**
 * POST /api/uploads/soldier-video
 * Upload video from soldier app
 */
router.post('/soldier-video', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file provided' });
    }

    const { soldierId, location, notes } = req.body;

    // Create event for this upload
    const event = new Event({
      type: 'soldier_upload',
      severity: 'info',
      title: 'סרטון התקבל מלוחם',
      source: soldierId || 'soldier-app',
      details: {
        metadata: {
          soldierId,
          location,
          notes,
          originalFilename: req.file.originalname,
          fileSize: req.file.size
        }
      },
      videoClip: {
        path: `/clips/${req.file.filename}`,
        duration: null // Could be extracted with ffprobe
      }
    });

    await event.save();

    // Notify all clients
    const io = req.app.get('io');
    io.emit('event:new', event.toObject());
    io.emit('soldier:upload:received', {
      eventId: event._id,
      filename: req.file.filename,
      soldierId
    });

    res.status(201).json({
      message: 'Video uploaded successfully',
      event,
      file: {
        filename: req.file.filename,
        path: `/clips/${req.file.filename}`,
        size: req.file.size
      }
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/uploads/snapshot
 * Upload detection snapshot from AI service
 */
router.post('/snapshot', upload.single('snapshot'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No snapshot file provided' });
    }

    const { eventId, cameraId } = req.body;

    // If eventId provided, update existing event
    if (eventId) {
      await Event.findByIdAndUpdate(eventId, {
        snapshot: {
          path: `/clips/${req.file.filename}`,
          timestamp: new Date()
        }
      });
    }

    res.json({
      message: 'Snapshot uploaded',
      filename: req.file.filename,
      path: `/clips/${req.file.filename}`
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/uploads/clip
 * Upload video clip from AI service (event recording)
 */
router.post('/clip', upload.single('clip'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No clip file provided' });
    }

    const { eventId, cameraId, duration } = req.body;

    // If eventId provided, update existing event
    if (eventId) {
      await Event.findByIdAndUpdate(eventId, {
        videoClip: {
          path: `/clips/${req.file.filename}`,
          duration: parseFloat(duration) || null
        }
      });
    }

    res.json({
      message: 'Clip uploaded',
      filename: req.file.filename,
      path: `/clips/${req.file.filename}`
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/uploads/frame
 * Upload frame for AI analysis
 */
router.post('/frame', upload.single('frame'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No frame file provided' });
    }

    res.json({
      message: 'Frame received',
      filename: req.file.filename,
      path: `/clips/${req.file.filename}`
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/uploads/list
 * List uploaded files
 */
router.get('/list', async (req, res) => {
  try {
    const uploadPath = process.env.CLIPS_PATH || path.join(__dirname, '../../../data/clips');

    if (!fs.existsSync(uploadPath)) {
      return res.json({ files: [] });
    }

    const files = fs.readdirSync(uploadPath).map(filename => {
      const filePath = path.join(uploadPath, filename);
      const stats = fs.statSync(filePath);
      return {
        filename,
        path: `/clips/${filename}`,
        size: stats.size,
        created: stats.birthtime
      };
    });

    // Sort by creation date, newest first
    files.sort((a, b) => b.created - a.created);

    res.json({ files });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * DELETE /api/uploads/:filename
 * Delete uploaded file
 */
router.delete('/:filename', async (req, res) => {
  try {
    const uploadPath = process.env.CLIPS_PATH || path.join(__dirname, '../../../data/clips');
    const filePath = path.join(uploadPath, req.params.filename);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'File not found' });
    }

    fs.unlinkSync(filePath);
    res.json({ message: 'File deleted', filename: req.params.filename });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
