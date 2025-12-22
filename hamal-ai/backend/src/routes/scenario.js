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

// ===========================================================================
// MANUAL DEMO TRIGGERS WITH REAL DATA
// ===========================================================================

/**
 * POST /api/scenario/demo/full-scenario
 * Trigger full demo scenario with realistic data
 * This simulates detecting a stolen vehicle followed by armed persons
 */
router.post('/demo/full-scenario', async (req, res) => {
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

    const io = manager.io;
    const cameraId = req.body.cameraId || 'cam-1';

    // Realistic stolen vehicle data
    const vehicleData = {
      licensePlate: '12-345-67',  // Matches stolen vehicles in config
      color: 'לבנה',
      make: 'מזדה',
      model: '3',
      vehicleType: 'sedan',
      cameraId,
      trackId: `v_${Date.now()}`,
      confidence: 0.92,
      bbox: [120, 200, 400, 450],
      timestamp: new Date().toISOString()
    };

    console.log('[Demo] Starting full scenario with vehicle:', vehicleData);

    // Step 1: Trigger stolen vehicle detection
    const vehicleTriggered = await manager.handleStolenVehicle(vehicleData);

    // Step 2: Add armed persons (delayed to simulate detection over time)
    const armedPersons = [];
    const weaponTypes = ['רובה', 'אקדח', 'רובה קצר'];
    const clothingColors = ['שחור', 'ירוק כהה', 'חאקי'];
    const clothingTypes = ['אפוד טקטי', 'מדים צבאיים', 'בגדים כהים'];

    // Add 3 armed persons progressively
    for (let i = 0; i < 3; i++) {
      setTimeout(async () => {
        const personData = {
          trackId: `t_${Date.now()}_${i}`,
          clothing: clothingTypes[i % clothingTypes.length],
          clothingColor: clothingColors[i % clothingColors.length],
          weaponType: weaponTypes[i % weaponTypes.length],
          armed: true,
          confidence: 0.85 + Math.random() * 0.1,
          cameraId,
          bbox: [300 + i * 80, 150, 380 + i * 80, 450],
          timestamp: new Date().toISOString()
        };

        armedPersons.push(personData);
        console.log(`[Demo] Adding armed person ${i + 1}:`, personData);

        try {
          await manager.handleArmedPerson(personData);
        } catch (err) {
          console.error('[Demo] Error adding armed person:', err);
        }
      }, (i + 1) * 2000); // Add each person 2 seconds apart
    }

    res.json({
      success: true,
      vehicleTriggered,
      message: 'Full demo scenario initiated - 3 armed persons will be added progressively',
      vehicle: vehicleData,
      note: 'Watch the UI for scenario progression',
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Demo full scenario error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/demo/stolen-vehicle
 * Trigger just the stolen vehicle detection with realistic data
 */
router.post('/demo/stolen-vehicle', async (req, res) => {
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

    const cameraId = req.body.cameraId || 'cam-1';

    // Use custom or default vehicle data
    const vehicleData = {
      licensePlate: req.body.licensePlate || '12-345-67',
      color: req.body.color || 'לבנה',
      make: req.body.make || 'מזדה',
      model: req.body.model || '3',
      vehicleType: req.body.vehicleType || 'sedan',
      cameraId,
      trackId: `v_${Date.now()}`,
      confidence: req.body.confidence || 0.92,
      bbox: req.body.bbox || [120, 200, 400, 450],
      timestamp: new Date().toISOString()
    };

    console.log('[Demo] Triggering stolen vehicle:', vehicleData);

    const triggered = await manager.handleStolenVehicle(vehicleData);

    res.json({
      success: true,
      triggered,
      vehicle: vehicleData,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Demo stolen vehicle error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/demo/armed-persons
 * Add multiple armed persons with realistic data
 */
router.post('/demo/armed-persons', async (req, res) => {
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

    const count = Math.min(req.body.count || 3, 10); // Max 10 at once
    const cameraId = req.body.cameraId || 'cam-1';

    const weaponTypes = ['רובה', 'אקדח', 'רובה קצר', 'סכין', 'רובה צלפים'];
    const clothingColors = ['שחור', 'ירוק כהה', 'חאקי', 'כחול כהה', 'אפור'];
    const clothingTypes = ['אפוד טקטי', 'מדים צבאיים', 'בגדים כהים', 'חולצה ארוכה', 'ז\'קט'];

    const results = [];

    for (let i = 0; i < count; i++) {
      const personData = {
        trackId: `t_${Date.now()}_${i}`,
        clothing: clothingTypes[i % clothingTypes.length],
        clothingColor: clothingColors[i % clothingColors.length],
        weaponType: weaponTypes[i % weaponTypes.length],
        armed: true,
        confidence: 0.8 + Math.random() * 0.15,
        cameraId,
        bbox: [100 + i * 100, 100, 200 + i * 100, 500],
        timestamp: new Date().toISOString()
      };

      try {
        const thresholdReached = await manager.handleArmedPerson(personData);
        results.push({ person: personData, thresholdReached });
        console.log(`[Demo] Added armed person ${i + 1}:`, personData);
      } catch (err) {
        results.push({ person: personData, error: err.message });
      }
    }

    res.json({
      success: true,
      count: results.length,
      results,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Demo armed persons error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenario/demo/keyword
 * Simulate radio transcription with specific keyword
 */
router.post('/demo/keyword', async (req, res) => {
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

    // Predefined keyword triggers
    const keywordPresets = {
      'drone': 'שלחו רחפן לאזור האירוע',
      'code': 'קוד צפרדע, חוזר, קוד צפרדע',
      'end': 'חדל חדל חדל, האירוע הסתיים',
      'custom': req.body.text
    };

    const keyword = req.body.keyword || 'drone';
    const text = keywordPresets[keyword] || keywordPresets['custom'];

    if (!text) {
      return res.status(400).json({
        error: 'No text provided. Use keyword preset (drone, code, end) or provide custom text.',
        availablePresets: Object.keys(keywordPresets).filter(k => k !== 'custom')
      });
    }

    console.log(`[Demo] Triggering keyword "${keyword}":`, text);

    const keywordMatched = await manager.handleTranscription(text);

    res.json({
      success: true,
      keyword,
      text,
      keywordMatched,
      state: manager.getState()
    });
  } catch (error) {
    console.error('[Scenario] Demo keyword error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/scenario/demo/presets
 * Get available demo presets and their descriptions
 */
router.get('/demo/presets', (req, res) => {
  res.json({
    endpoints: {
      'POST /api/scenario/demo/full-scenario': {
        description: 'Trigger full demo: stolen vehicle + 3 armed persons progressively',
        params: { cameraId: 'optional, defaults to cam-1' }
      },
      'POST /api/scenario/demo/stolen-vehicle': {
        description: 'Trigger stolen vehicle detection',
        params: {
          cameraId: 'optional, defaults to cam-1',
          licensePlate: 'optional, defaults to 12-345-67',
          color: 'optional, defaults to לבנה',
          make: 'optional, defaults to מזדה',
          model: 'optional, defaults to 3'
        }
      },
      'POST /api/scenario/demo/armed-persons': {
        description: 'Add armed persons',
        params: {
          count: 'number of persons (1-10), defaults to 3',
          cameraId: 'optional, defaults to cam-1'
        }
      },
      'POST /api/scenario/demo/keyword': {
        description: 'Trigger keyword detection',
        params: {
          keyword: 'preset name (drone, code, end) or provide text',
          text: 'custom text for keyword=custom'
        }
      },
      'POST /api/scenario/reset': {
        description: 'Reset scenario to idle state'
      }
    },
    stolenVehiclePlates: SCENARIO_CONFIG.stolenVehicles.map(v => v.licensePlate),
    keywords: SCENARIO_CONFIG.keywords
  });
});

export default router;
