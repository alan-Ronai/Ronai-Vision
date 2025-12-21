/**
 * Scenario API Routes
 *
 * Endpoints for managing the Armed Attack demo scenario.
 * Handles scenario state, triggers, and special endpoints like soldier video upload.
 */

import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import { getScenarioManager } from '../services/scenarioManager.js';
import { SCENARIO_CONFIG, isPlateStolen } from '../config/scenarioConfig.js';

const router = express.Router();
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Configure multer for soldier video uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../../uploads/scenario');
    try {
      await fs.mkdir(uploadDir, { recursive: true });
      cb(null, uploadDir);
    } catch (error) {
      cb(error);
    }
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const ext = path.extname(file.originalname);
    cb(null, `soldier-video-${timestamp}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB max
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['video/mp4', 'video/webm', 'video/quicktime'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only MP4, WebM, and MOV are allowed.'));
    }
  }
});

/**
 * GET /api/scenario/status
 * Get current scenario status
 */
router.get('/status', (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const state = manager.getState();
    res.json(state);
  } catch (error) {
    console.error('[Scenario] Status error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/scenario/config
 * Get scenario configuration
 */
router.get('/config', (req, res) => {
  res.json({
    name: SCENARIO_CONFIG.name,
    displayName: SCENARIO_CONFIG.displayName,
    stages: SCENARIO_CONFIG.stages,
    thresholds: SCENARIO_CONFIG.thresholds,
    keywords: SCENARIO_CONFIG.keywords,
    messages: SCENARIO_CONFIG.messages,
    ui: SCENARIO_CONFIG.ui
  });
});

/**
 * GET /api/scenario/stolen-vehicles
 * Get list of stolen vehicles
 */
router.get('/stolen-vehicles', (req, res) => {
  res.json(SCENARIO_CONFIG.stolenVehicles);
});

/**
 * POST /api/scenario/stolen-vehicles
 * Add a vehicle to stolen list
 */
router.post('/stolen-vehicles', (req, res) => {
  const { licensePlate, description } = req.body;

  if (!licensePlate) {
    return res.status(400).json({ error: 'License plate is required' });
  }

  // Add to config (in memory, for demo)
  SCENARIO_CONFIG.stolenVehicles.push({
    licensePlate,
    description: description || 'Added via API',
    reportedAt: new Date().toISOString()
  });

  res.json({
    success: true,
    message: `Added ${licensePlate} to stolen vehicles list`,
    stolenVehicles: SCENARIO_CONFIG.stolenVehicles
  });
});

/**
 * POST /api/scenario/check-plate
 * Check if a license plate is stolen
 */
router.post('/check-plate', (req, res) => {
  const { licensePlate } = req.body;

  if (!licensePlate) {
    return res.status(400).json({ error: 'License plate is required' });
  }

  const stolenInfo = isPlateStolen(licensePlate);

  res.json({
    licensePlate,
    isStolen: !!stolenInfo,
    stolenInfo
  });
});

/**
 * POST /api/scenario/vehicle-detected
 * Handle vehicle detection from AI service
 */
router.post('/vehicle-detected', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const vehicleData = req.body;
    console.log('[Scenario] Vehicle detected:', vehicleData);

    const triggered = await manager.handleStolenVehicle(vehicleData);

    res.json({
      success: true,
      scenarioTriggered: triggered,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Vehicle detection error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/armed-person
 * Handle armed person detection from AI service
 */
router.post('/armed-person', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const personData = req.body;
    console.log('[Scenario] Armed person detected:', personData);

    const thresholdReached = await manager.handleArmedPerson(personData);

    res.json({
      success: true,
      thresholdReached,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Armed person error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/transcription
 * Handle transcription from radio (for keyword detection)
 */
router.post('/transcription', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const { text } = req.body;
    console.log('[Scenario] Transcription received:', text);

    const keywordMatched = await manager.handleTranscription(text);

    res.json({
      success: true,
      keywordMatched,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Transcription error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/soldier-video
 * Upload soldier video
 */
router.post('/soldier-video', upload.single('video'), async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'No video file provided' });
    }

    const videoPath = `/uploads/scenario/${req.file.filename}`;
    console.log('[Scenario] Soldier video uploaded:', videoPath);

    const handled = await manager.handleSoldierVideo(videoPath);

    res.json({
      success: true,
      handled,
      videoPath,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Soldier video error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/video-panel-closed
 * Handle soldier video panel closed
 */
router.post('/video-panel-closed', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const handled = await manager.handleVideoPanelClosed();

    res.json({
      success: true,
      handled,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Video panel closed error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/new-camera
 * Handle new camera connection
 */
router.post('/new-camera', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const { url } = req.body;
    if (!url) {
      return res.status(400).json({ error: 'Camera URL is required' });
    }

    const handled = await manager.handleNewCameraConnected(url);

    res.json({
      success: true,
      handled,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] New camera error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/acknowledge
 * User acknowledges emergency (clicked "מטפל בזה")
 */
router.post('/acknowledge', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const acknowledged = await manager.acknowledgeEmergency();

    res.json({
      success: true,
      acknowledged,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Acknowledge error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/false-alarm
 * User declares false alarm (clicked "אזעקת שווא")
 */
router.post('/false-alarm', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const ended = await manager.falseAlarm();

    res.json({
      success: true,
      ended,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] False alarm error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/end
 * Manually end the scenario
 */
router.post('/end', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const { reason } = req.body;
    const ended = await manager.endScenario(reason || 'manual');

    res.json({
      success: true,
      ended,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] End error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/reset
 * Reset scenario to idle (for testing)
 */
router.post('/reset', (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    manager.reset();

    res.json({
      success: true,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Reset error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ===========================================================================
// SCENARIO RULE ENGINE INTEGRATION
// ===========================================================================

/**
 * POST /api/scenario/engine-event
 * Receive action events from AI service ScenarioRuleEngine
 * Routes actions to appropriate Socket.IO emissions
 */
router.post('/engine-event', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const { action, params, scenarioId, stage, context } = req.body;

    console.log(`[Scenario Engine] Received action: ${action}`, { scenarioId, stage, params });

    // Get Socket.IO instance from manager
    const io = manager.io;

    // Route action to appropriate handler
    switch (action) {
      case 'alert_popup':
        io.emit('scenario:alert', {
          type: params.type || 'warning',
          alertType: params.alertType,
          title: params.title,
          vehicle: context?.vehicle,
          autoDismiss: params.autoDismiss ?? false
        });
        break;

      case 'danger_mode':
        io.emit('scenario:danger-mode', { active: params.active });
        break;

      case 'emergency_modal':
        io.emit('scenario:emergency-modal', {
          title: params.title,
          subtitle: params.subtitle,
          vehicle: context?.vehicle,
          persons: context?.persons || [],
          showVehicle: params.showVehicle,
          showPersons: params.showPersons
        });
        break;

      case 'close_modal':
        io.emit('scenario:emergency-modal', { close: true });
        break;

      case 'journal':
        io.emit('event:new', {
          type: 'alert',
          severity: params.severity || 'info',
          title: params.title,
          description: params.description,
          timestamp: new Date().toISOString(),
          scenarioId,
          stage,
          cameraId: context?.vehicle?.cameraId,
          metadata: {
            vehicle: context?.vehicle,
            persons: context?.persons,
            armedCount: context?.armedCount
          }
        });
        break;

      case 'tts':
        io.emit('scenario:tts', {
          message: params.message,
          priority: params.priority || 'normal'
        });
        // Also trigger AI service TTS generation
        try {
          const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
          await fetch(`${AI_SERVICE_URL}/tts/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: params.message, language: 'he' })
          });
        } catch (ttsErr) {
          console.error('[Scenario Engine] TTS generation error:', ttsErr.message);
        }
        break;

      case 'play_sound':
        io.emit('scenario:play-sound', {
          sound: params.sound,
          loop: params.loop ?? false,
          repeatCount: params.repeatCount ?? 1
        });
        break;

      case 'stop_all_sounds':
        io.emit('scenario:stop-all-sounds', {});
        break;

      case 'camera_focus':
        io.emit('camera:auto-focus', {
          cameraId: params.cameraId,
          reason: params.reason,
          priority: 'critical',
          scenarioFocus: true
        });
        break;

      case 'simulation':
        io.emit('scenario:simulation', {
          type: params.type,
          title: params.title
        });
        break;

      case 'soldier_video_panel':
        io.emit('scenario:soldier-video', {
          open: params.open,
          videoPath: params.videoPath
        });
        break;

      case 'new_camera_dialog':
        io.emit('scenario:new-camera-dialog', { open: params.open });
        break;

      case 'summary_popup':
        io.emit('scenario:summary', {
          title: params.title,
          duration: context?.duration,
          reason: context?.endReason,
          vehicle: context?.vehicle,
          persons: context?.persons,
          armedCount: context?.armedCount
        });
        break;

      case 'start_recording':
        io.emit('scenario:recording', {
          action: 'start',
          allCameras: params.allCameras,
          preBuffer: params.preBuffer,
          scenarioId
        });
        break;

      case 'stop_recording':
        io.emit('scenario:recording', {
          action: 'stop',
          scenarioId
        });
        break;

      case 'cleanup':
        // Delayed cleanup
        setTimeout(() => {
          io.emit('scenario:ended', {});
        }, params.delay || 5000);
        break;

      case 'store_context':
        // Context storage is handled by the rule engine, just acknowledge
        break;

      case 'track_armed_persons':
        // Tracking is handled by the rule engine
        break;

      case 'delay':
        // Delays are handled by the rule engine
        break;

      default:
        console.warn(`[Scenario Engine] Unknown action: ${action}`);
    }

    // Emit stage change for UI tracking
    io.emit('scenario:engine-stage', {
      scenarioId,
      stage,
      action,
      context
    });

    res.json({
      success: true,
      action,
      stage
    });
  } catch (error) {
    console.error('[Scenario Engine] Event error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/engine-transition
 * Receive stage transition events from AI service ScenarioRuleEngine
 */
router.post('/engine-transition', async (req, res) => {
  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const { scenarioId, fromStage, toStage, trigger, context } = req.body;

    console.log(`[Scenario Engine] Stage transition: ${fromStage} -> ${toStage}`, { scenarioId, trigger });

    const io = manager.io;

    // Emit stage transition for UI
    io.emit('scenario:engine-transition', {
      scenarioId,
      fromStage,
      toStage,
      trigger,
      context,
      timestamp: new Date().toISOString()
    });

    res.json({
      success: true,
      fromStage,
      toStage
    });
  } catch (error) {
    console.error('[Scenario Engine] Transition error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/scenario/engine-status
 * Get current scenario rule engine status from AI service
 */
router.get('/engine-status', async (req, res) => {
  try {
    const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

    const response = await fetch(`${AI_SERVICE_URL}/scenario-rules/active/status`);

    if (response.ok) {
      const status = await response.json();
      res.json(status);
    } else {
      res.status(response.status).json({ error: 'Failed to get engine status' });
    }
  } catch (error) {
    console.error('[Scenario Engine] Status fetch error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ===========================================================================
// TEST ENDPOINTS (only available in demo mode)
// ===========================================================================

/**
 * POST /api/scenario/test/start
 * Manually start scenario with test data
 */
router.post('/test/start', async (req, res) => {
  if (!SCENARIO_CONFIG.demo.enabled) {
    return res.status(403).json({ error: 'Demo mode is disabled' });
  }

  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    // Use test vehicle data
    const testVehicle = {
      licensePlate: 'DEMO-123',
      color: 'שחור',
      make: 'טויוטה',
      model: 'קורולה',
      cameraId: req.body.cameraId || 'cam-1',
      trackId: 999,
      confidence: 0.95
    };

    const triggered = await manager.handleStolenVehicle(testVehicle);

    res.json({
      success: true,
      triggered,
      message: 'Test scenario started',
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Test start error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/test/add-armed-person
 * Add test armed person
 */
router.post('/test/add-armed-person', async (req, res) => {
  if (!SCENARIO_CONFIG.demo.enabled) {
    return res.status(403).json({ error: 'Demo mode is disabled' });
  }

  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const testPerson = {
      trackId: Date.now(),
      clothing: req.body.clothing || 'חולצה שחורה',
      clothingColor: req.body.clothingColor || 'שחור',
      weaponType: req.body.weaponType || 'רובה',
      armed: true,
      confidence: 0.9
    };

    const thresholdReached = await manager.handleArmedPerson(testPerson);

    res.json({
      success: true,
      thresholdReached,
      message: 'Test armed person added',
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Test add armed person error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/test/advance
 * Force advance to next stage
 */
router.post('/test/advance', async (req, res) => {
  if (!SCENARIO_CONFIG.demo.enabled) {
    return res.status(403).json({ error: 'Demo mode is disabled' });
  }

  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const advanced = await manager.advanceStage();

    res.json({
      success: true,
      advanced,
      message: advanced ? 'Advanced to next stage' : 'Could not advance',
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Test advance error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/test/transcription
 * Simulate transcription with test text
 */
router.post('/test/transcription', async (req, res) => {
  if (!SCENARIO_CONFIG.demo.enabled) {
    return res.status(403).json({ error: 'Demo mode is disabled' });
  }

  try {
    const manager = getScenarioManager();
    if (!manager) {
      return res.status(503).json({
        error: 'Scenario manager not initialized'
      });
    }

    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }

    const keywordMatched = await manager.handleTranscription(text);

    res.json({
      success: true,
      keywordMatched,
      message: keywordMatched ? 'Keyword detected' : 'No keyword match',
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Test transcription error:', error);
    res.status(500).json({ error: error.message });
  }
});

export default router;
