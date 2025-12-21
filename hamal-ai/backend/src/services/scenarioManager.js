/**
 * Scenario Manager - State Machine for Armed Attack Demo
 *
 * Manages the lifecycle of the Armed Attack scenario through various stages.
 * Ensures only one scenario instance runs at a time and handles all state
 * transitions, actions, and event coordination.
 */

import { EventEmitter } from 'events';
import {
  SCENARIO_CONFIG,
  STAGE_FLOW,
  isPlateStolen,
  containsKeyword,
  getNextStage,
  isValidStage
} from '../config/scenarioConfig.js';

// Scenario states
const STAGES = SCENARIO_CONFIG.stages;

class ScenarioManager extends EventEmitter {
  constructor(io) {
    super();
    this.io = io;
    this.config = SCENARIO_CONFIG;

    // Current scenario state
    this.scenario = null;

    // Timeouts for stage progression
    this.timeouts = {};

    // Armed persons buffer (for counting before trigger)
    this.armedPersonsBuffer = [];

    // Initialization
    this._setupSocketHandlers();

    console.log('[ScenarioManager] Initialized with config:', this.config.displayName);
  }

  // ===========================================================================
  // SCENARIO STATE MANAGEMENT
  // ===========================================================================

  /**
   * Check if scenario is currently active
   */
  isActive() {
    return this.scenario !== null && this.scenario.stage !== STAGES.IDLE;
  }

  /**
   * Get current scenario state
   */
  getState() {
    if (!this.scenario) {
      return {
        active: false,
        stage: STAGES.IDLE,
        context: null
      };
    }

    return {
      active: this.isActive(),
      stage: this.scenario.stage,
      scenarioId: this.scenario.id,
      startedAt: this.scenario.startedAt,
      vehicle: this.scenario.vehicle,
      persons: this.scenario.persons,
      armedCount: this.scenario.armedCount,
      acknowledged: this.scenario.acknowledged,
      recording: this.scenario.recording,
      stageHistory: this.scenario.stageHistory,
      currentStageData: this.scenario.currentStageData
    };
  }

  /**
   * Initialize a new scenario
   */
  _initScenario(vehicleData) {
    const id = `armed-attack-${Date.now()}`;

    this.scenario = {
      id,
      stage: STAGES.VEHICLE_DETECTED,
      startedAt: new Date().toISOString(),
      vehicle: vehicleData,
      persons: [],
      armedCount: 0,
      acknowledged: false,
      recording: null,
      stageHistory: [{
        stage: STAGES.VEHICLE_DETECTED,
        enteredAt: new Date().toISOString(),
        data: { vehicle: vehicleData }
      }],
      currentStageData: {},
      soldierVideoPath: null,
      newCameraUrl: null,
      transcription: null
    };

    console.log(`[ScenarioManager] New scenario initialized: ${id}`);
    return this.scenario;
  }

  /**
   * Transition to a new stage
   */
  async _transitionTo(newStage, data = {}) {
    if (!this.scenario) {
      console.warn('[ScenarioManager] No active scenario for transition');
      return false;
    }

    const oldStage = this.scenario.stage;

    // Validate stage transition
    if (!isValidStage(newStage)) {
      console.error(`[ScenarioManager] Invalid stage: ${newStage}`);
      return false;
    }

    // Clear any pending timeouts
    this._clearTimeouts();

    // Update state
    this.scenario.stage = newStage;
    this.scenario.currentStageData = data;
    this.scenario.stageHistory.push({
      stage: newStage,
      enteredAt: new Date().toISOString(),
      previousStage: oldStage,
      data
    });

    console.log(`[ScenarioManager] Stage transition: ${oldStage} -> ${newStage}`, data);

    // Emit state change to all clients
    this._emitStateChange(newStage, data);

    // Execute stage-specific actions
    await this._executeStageActions(newStage, data);

    return true;
  }

  /**
   * Execute actions for a specific stage
   */
  async _executeStageActions(stage, data) {
    switch (stage) {
      case STAGES.VEHICLE_ALERT:
        await this._handleVehicleAlert(data);
        break;

      case STAGES.ARMED_PERSONS_DETECTED:
        // Just tracking, will transition when count reaches threshold
        break;

      case STAGES.EMERGENCY_MODE:
        await this._handleEmergencyMode(data);
        break;

      case STAGES.RESPONSE_INITIATED:
        await this._handleResponseInitiated(data);
        break;

      case STAGES.DRONE_DISPATCHED:
        await this._handleDroneDispatched(data);
        break;

      case STAGES.CIVILIAN_ALERT:
        await this._handleCivilianAlert(data);
        break;

      case STAGES.CODE_BROADCAST:
        await this._handleCodeBroadcast(data);
        break;

      case STAGES.SOLDIER_VIDEO:
        await this._handleSoldierVideo(data);
        break;

      case STAGES.NEW_CAMERA:
        await this._handleNewCamera(data);
        break;

      case STAGES.SITUATION_END:
        await this._handleSituationEnd(data);
        break;
    }
  }

  // ===========================================================================
  // STAGE HANDLERS
  // ===========================================================================

  /**
   * VEHICLE_ALERT stage handler
   */
  async _handleVehicleAlert(data) {
    const vehicle = this.scenario.vehicle;

    // 1. Emit UI alert popup (full-screen modal for vehicle)
    this._emitAlert({
      type: 'warning',
      alertType: 'vehicle',  // This triggers VehicleAlertModal
      title: this.config.ui.stolenVehicleTitle,
      vehicle: {
        licensePlate: vehicle.licensePlate,
        color: vehicle.color,
        make: vehicle.make,
        model: vehicle.model,
        cameraId: vehicle.cameraId
      },
      autoDismiss: false
    });

    // 2. Play alert sound (once, not looping)
    this._emitSound('alert', { loop: false });

    // 3. Create journal entry with detailed description
    const vehicleDetails = [
      `לוחית רישוי: ${vehicle.licensePlate}`,
      vehicle.color ? `צבע: ${vehicle.color}` : null,
      (vehicle.make || vehicle.model) ? `יצרן/דגם: ${vehicle.make || ''} ${vehicle.model || ''}`.trim() : null,
      vehicle.cameraId ? `מצלמה: ${vehicle.cameraId}` : null
    ].filter(Boolean).join('\n');

    await this._createEvent({
      type: 'alert',
      severity: 'warning',
      title: 'רכב גנוב זוהה',
      description: vehicleDetails,
      cameraId: vehicle.cameraId,
      metadata: { vehicle, scenarioId: this.scenario.id }
    });

    // 3. TTS to radio
    await this._sendTTS(this.config.messages.stolenVehicle);

    // 4. Auto-focus camera
    this._emitCameraFocus(vehicle.cameraId, 'זיהוי רכב גנוב', 'critical');

    // 5. Set timeout for armed persons detection
    this._setTimeout('vehicleAlert', this.config.timeouts.vehicleAlertTimeout, () => {
      console.log('[ScenarioManager] Vehicle alert timeout - waiting for manual intervention');
    });
  }

  /**
   * EMERGENCY_MODE stage handler
   */
  async _handleEmergencyMode(data) {
    const persons = this.scenario.persons;
    const count = this.scenario.armedCount;

    // 1. Activate danger mode UI
    this._emitDangerMode(true);

    // 2. Show emergency modal
    this._emitEmergencyModal({
      title: this.config.ui.emergencyTitle,
      subtitle: this.config.ui.emergencySubtitle.replace('{count}', count),
      vehicle: this.scenario.vehicle,
      persons,
      cameraId: this.scenario.vehicle.cameraId
    });

    // 3. Journal entry with detailed description
    const vehicle = this.scenario.vehicle;
    const personDetails = persons.map((p, i) => {
      const parts = [`חמוש #${i + 1}`];
      if (p.clothing) parts.push(`לבוש: ${p.clothing}`);
      if (p.weaponType) parts.push(`נשק: ${p.weaponType}`);
      return parts.join(' | ');
    }).join('\n');

    const emergencyDetails = [
      `זוהו ${count} חמושים`,
      '',
      'פרטי רכב:',
      `לוחית רישוי: ${vehicle.licensePlate}`,
      vehicle.color ? `צבע: ${vehicle.color}` : null,
      (vehicle.make || vehicle.model) ? `יצרן/דגם: ${vehicle.make || ''} ${vehicle.model || ''}`.trim() : null,
      '',
      'חמושים:',
      personDetails
    ].filter(line => line !== null).join('\n');

    await this._createEvent({
      type: 'alert',
      severity: 'critical',
      title: 'חדירה ודאית',
      description: emergencyDetails,
      cameraId: this.scenario.vehicle.cameraId,
      metadata: {
        scenarioId: this.scenario.id,
        vehicle: this.scenario.vehicle,
        persons,
        armedCount: count
      }
    });

    // 4. TTS with person descriptions
    const descriptions = this._buildPersonDescriptions(persons);
    const message = this.config.messages.armedPersonsWithDetails
      .replace('{count}', count)
      .replace('{descriptions}', descriptions);
    await this._sendTTS(message);

    // 5. Start recording
    await this._startRecording();

    // 6. Play alarm sound (twice, not infinite)
    this._emitSound('alarm', { loop: false, repeatCount: 2 });

    // 7. Set auto-progress timeout (if user doesn't click)
    this._setTimeout('emergencyAutoProgress', this.config.timeouts.emergencyAutoProgress, () => {
      if (!this.scenario.acknowledged) {
        console.log('[ScenarioManager] Emergency auto-progress - user did not acknowledge');
        this.acknowledgeEmergency();
      }
    });
  }

  /**
   * RESPONSE_INITIATED stage handler
   */
  async _handleResponseInitiated(data) {
    // 1. TTS to response team
    await this._sendTTS(this.config.messages.responseTeam);

    // 2. Start phone call simulation
    this._emitSimulation('phone_call', {
      title: this.config.ui.callingCommander,
      sound: 'phoneDial'
    });

    // After 3 seconds, play phone ring (3 times, not infinite)
    this._setTimeout('phoneRing', 3000, () => {
      this._emitSound('phoneRing', { loop: false, repeatCount: 3 });
    });

    this._setTimeout('phoneConnected', 6000, () => {
      this._stopSound('phoneRing');
      this._emitSimulation('phone_connected', {
        title: this.config.ui.commanderOnLine
      });
    });

    // 3. Journal entry
    await this._createEvent({
      type: 'simulation',
      severity: 'info',
      title: 'התקשרות למפקד תורן',
      description: 'הופעלה התקשרות אוטומטית למפקד תורן',
      metadata: { scenarioId: this.scenario.id }
    });

    // 4. Set timeout for drone keyword
    this._setTimeout('droneKeyword', this.config.timeouts.droneKeywordTimeout, () => {
      console.log('[ScenarioManager] Drone keyword timeout - waiting for "רחפן"');
    });
  }

  /**
   * DRONE_DISPATCHED stage handler
   */
  async _handleDroneDispatched(data) {
    // 1. Show drone simulation
    this._emitSimulation('drone_dispatch', {
      title: this.config.ui.droneDispatching,
      sound: 'droneTakeoff'
    });

    // 2. Play drone sound
    this._emitSound('droneTakeoff');

    // 3. Journal entry
    await this._createEvent({
      type: 'simulation',
      severity: 'info',
      title: 'רחפן הוקפץ',
      description: 'רחפן הוקפץ לאזור האירוע',
      metadata: { scenarioId: this.scenario.id }
    });

    // 4. After delay, show "drone en route" and auto-progress
    this._setTimeout('droneEnRoute', 3000, () => {
      this._emitSimulation('drone_enroute', {
        title: this.config.ui.droneEnRoute
      });
    });

    this._setTimeout('civilianAlert', this.config.timeouts.stageProgressDelay, async () => {
      await this._transitionTo(STAGES.CIVILIAN_ALERT);
    });
  }

  /**
   * CIVILIAN_ALERT stage handler
   */
  async _handleCivilianAlert(data) {
    // 1. TTS to residential area
    await this._sendTTS(this.config.messages.civilianAlert);

    // 2. Show UI indicator
    this._emitSimulation('pa_announcement', {
      title: this.config.ui.announcementSent
    });

    // 3. Journal entry
    await this._createEvent({
      type: 'simulation',
      severity: 'warning',
      title: 'כריזה למגורים',
      description: 'בוצעה כריזת התרעה למגורים',
      metadata: { scenarioId: this.scenario.id }
    });

    // 4. Set timeout for code keyword
    this._setTimeout('codeKeyword', this.config.timeouts.codeKeywordTimeout, () => {
      console.log('[ScenarioManager] Code keyword timeout - waiting for "צפרדע"');
    });
  }

  /**
   * CODE_BROADCAST stage handler
   */
  async _handleCodeBroadcast(data) {
    const codeWord = this.config.keywords.codeBroadcast;

    // 1. TTS code word 3 times
    await this._sendTTS(`${codeWord} ${codeWord} ${codeWord}`);

    // 2. Show UI indicator
    this._emitSimulation('code_broadcast', {
      title: `${this.config.ui.codeBroadcast}: ${codeWord}`
    });

    // 3. Journal entry
    await this._createEvent({
      type: 'simulation',
      severity: 'info',
      title: 'קוד שודר',
      description: `קוד "${codeWord}" שודר 3 פעמים`,
      metadata: { scenarioId: this.scenario.id }
    });

    // 4. Set timeout for soldier video
    this._setTimeout('soldierVideo', this.config.timeouts.soldierVideoTimeout, () => {
      console.log('[ScenarioManager] Soldier video timeout - waiting for video upload');
    });
  }

  /**
   * SOLDIER_VIDEO stage handler
   */
  async _handleSoldierVideo(data) {
    // 1. Open soldier video panel
    this._emitSoldierVideoPanel({
      videoPath: data.videoPath,
      open: true
    });

    // 2. Journal entry
    await this._createEvent({
      type: 'system',
      severity: 'info',
      title: 'סרטון מלוחם התקבל',
      description: 'סרטון מלוחם התקבל ומוצג',
      metadata: { scenarioId: this.scenario.id, videoPath: data.videoPath }
    });
  }

  /**
   * NEW_CAMERA stage handler
   */
  async _handleNewCamera(data) {
    // 1. Show new camera dialog
    this._emitNewCameraDialog(true);

    // 2. Set timeout for end keyword
    this._setTimeout('endKeyword', this.config.timeouts.endKeywordTimeout, () => {
      console.log('[ScenarioManager] End keyword timeout - waiting for "חדל"');
    });
  }

  /**
   * SITUATION_END stage handler
   */
  async _handleSituationEnd(data) {
    const reason = data.reason || 'normal';
    const duration = Date.now() - new Date(this.scenario.startedAt).getTime();

    // 1. Deactivate danger mode
    this._emitDangerMode(false);

    // 2. Stop alarm
    this._stopSound('alarm');

    // 3. Stop recording
    await this._stopRecording();

    // 4. Show summary popup
    this._emitSummaryPopup({
      title: this.config.ui.situationEndedTitle,
      duration,
      reason,
      vehicle: this.scenario.vehicle,
      persons: this.scenario.persons,
      armedCount: this.scenario.armedCount
    });

    // 5. Journal entry
    await this._createEvent({
      type: 'system',
      severity: 'info',
      title: 'אירוע הסתיים',
      description: `אירוע הסתיים - סיבה: ${reason === 'false_alarm' ? 'אזעקת שווא' : 'נוטרל'}`,
      metadata: {
        scenarioId: this.scenario.id,
        duration,
        reason,
        stageHistory: this.scenario.stageHistory
      }
    });

    // 6. Clear scenario after delay
    this._setTimeout('cleanup', 5000, () => {
      this._cleanup();
    });
  }

  // ===========================================================================
  // PUBLIC API - Called by routes and other services
  // ===========================================================================

  /**
   * Handle stolen vehicle detection (from AI service)
   */
  async handleStolenVehicle(vehicleData) {
    // Prevent multiple scenarios
    if (this.isActive()) {
      console.log('[ScenarioManager] Scenario already active, ignoring vehicle detection');
      return false;
    }

    // Validate vehicle data
    if (!vehicleData.licensePlate) {
      console.log('[ScenarioManager] No license plate in vehicle data');
      return false;
    }

    // Check if plate is stolen
    const stolenInfo = isPlateStolen(vehicleData.licensePlate);
    if (!stolenInfo) {
      console.log(`[ScenarioManager] Plate ${vehicleData.licensePlate} not in stolen list`);
      return false;
    }

    console.log(`[ScenarioManager] Stolen vehicle detected: ${vehicleData.licensePlate}`);

    // Initialize scenario
    this._initScenario({
      ...vehicleData,
      stolenInfo
    });

    // Transition to alert stage
    await this._transitionTo(STAGES.VEHICLE_ALERT, { vehicle: vehicleData });

    return true;
  }

  /**
   * Handle armed person detection (from AI service)
   */
  async handleArmedPerson(personData) {
    // Must be in VEHICLE_ALERT stage
    if (!this.scenario || this.scenario.stage !== STAGES.VEHICLE_ALERT) {
      console.log('[ScenarioManager] Not in VEHICLE_ALERT stage, ignoring armed person');
      return false;
    }

    // Add person to list
    this.scenario.persons.push(personData);
    this.scenario.armedCount++;

    console.log(`[ScenarioManager] Armed person added: ${this.scenario.armedCount} total`);

    // Emit update
    this._emitStateChange(this.scenario.stage, {
      personAdded: personData,
      armedCount: this.scenario.armedCount
    });

    // Check threshold
    if (this.scenario.armedCount >= this.config.thresholds.armedPersonsRequired) {
      console.log('[ScenarioManager] Armed persons threshold reached, triggering emergency');
      await this._transitionTo(STAGES.EMERGENCY_MODE, {
        armedCount: this.scenario.armedCount,
        persons: this.scenario.persons
      });
      return true;
    }

    return false;
  }

  /**
   * Handle transcription from radio (for keyword detection)
   */
  async handleTranscription(text) {
    if (!this.scenario) return false;

    console.log(`[ScenarioManager] Processing transcription: "${text}"`);

    const stage = this.scenario.stage;

    // Check for stage-specific keywords
    switch (stage) {
      case STAGES.RESPONSE_INITIATED:
        // Looking for "רחפן"
        if (containsKeyword(text, this.config.keywords.drone)) {
          console.log('[ScenarioManager] Drone keyword detected');
          await this._transitionTo(STAGES.DRONE_DISPATCHED, { keyword: text });
          return true;
        }
        break;

      case STAGES.CIVILIAN_ALERT:
        // Looking for code keyword
        if (containsKeyword(text, this.config.keywords.codeKeyword)) {
          console.log('[ScenarioManager] Code keyword detected');
          await this._transitionTo(STAGES.CODE_BROADCAST, { keyword: text });
          return true;
        }
        break;

      case STAGES.NEW_CAMERA:
        // Looking for end keywords
        if (containsKeyword(text, this.config.keywords.end)) {
          console.log('[ScenarioManager] End keyword detected');
          await this._transitionTo(STAGES.SITUATION_END, { reason: 'neutralized', keyword: text });
          return true;
        }
        break;
    }

    return false;
  }

  /**
   * Handle soldier video upload
   */
  async handleSoldierVideo(videoPath) {
    if (!this.scenario || this.scenario.stage !== STAGES.CODE_BROADCAST) {
      console.log('[ScenarioManager] Not expecting soldier video at this stage');
      return false;
    }

    this.scenario.soldierVideoPath = videoPath;
    await this._transitionTo(STAGES.SOLDIER_VIDEO, { videoPath });
    return true;
  }

  /**
   * Handle soldier video panel closed
   */
  async handleVideoPanelClosed() {
    if (!this.scenario || this.scenario.stage !== STAGES.SOLDIER_VIDEO) {
      return false;
    }

    await this._transitionTo(STAGES.NEW_CAMERA);
    return true;
  }

  /**
   * Handle new camera connected
   */
  async handleNewCameraConnected(cameraUrl) {
    if (!this.scenario || this.scenario.stage !== STAGES.NEW_CAMERA) {
      return false;
    }

    this.scenario.newCameraUrl = cameraUrl;

    // Emit camera added
    this.io.emit('camera:added', { url: cameraUrl, scenarioCamera: true });

    await this._createEvent({
      type: 'system',
      severity: 'info',
      title: 'מצלמה חדשה התחברה',
      description: `מצלמה חדשה התחברה: ${cameraUrl}`,
      metadata: { scenarioId: this.scenario.id }
    });

    return true;
  }

  /**
   * Acknowledge emergency (user clicked "מטפל בזה")
   */
  async acknowledgeEmergency() {
    if (!this.scenario || this.scenario.stage !== STAGES.EMERGENCY_MODE) {
      return false;
    }

    this.scenario.acknowledged = true;
    this._clearTimeout('emergencyAutoProgress');

    // Close emergency modal
    this._emitEmergencyModal({ close: true });

    // Progress to response initiated
    await this._transitionTo(STAGES.RESPONSE_INITIATED);
    return true;
  }

  /**
   * False alarm (user clicked "אזעקת שווא")
   */
  async falseAlarm() {
    if (!this.scenario) {
      return false;
    }

    console.log('[ScenarioManager] False alarm triggered');

    // Close any modals
    this._emitEmergencyModal({ close: true });

    // End scenario
    await this._transitionTo(STAGES.SITUATION_END, { reason: 'false_alarm' });
    return true;
  }

  /**
   * Manually end scenario
   */
  async endScenario(reason = 'manual') {
    if (!this.scenario) {
      return false;
    }

    await this._transitionTo(STAGES.SITUATION_END, { reason });
    return true;
  }

  /**
   * Reset to idle (for testing)
   */
  reset() {
    this._clearTimeouts();
    this._cleanup();
    console.log('[ScenarioManager] Reset to idle');
  }

  /**
   * Force advance to next stage (for testing)
   */
  async advanceStage() {
    if (!this.scenario) {
      return false;
    }

    const nextStage = getNextStage(this.scenario.stage);
    if (!nextStage) {
      console.log('[ScenarioManager] No next stage available');
      return false;
    }

    await this._transitionTo(nextStage, { forced: true });
    return true;
  }

  // ===========================================================================
  // HELPERS
  // ===========================================================================

  /**
   * Emit state change to all clients
   */
  _emitStateChange(stage, data) {
    this.io.emit('scenario:stage-changed', {
      stage,
      data,
      state: this.getState()
    });
  }

  /**
   * Emit alert popup
   */
  _emitAlert(alertData) {
    this.io.emit('scenario:alert', alertData);
  }

  /**
   * Emit danger mode
   */
  _emitDangerMode(active) {
    this.io.emit('scenario:danger-mode', { active });
  }

  /**
   * Emit emergency modal
   */
  _emitEmergencyModal(data) {
    this.io.emit('scenario:emergency-modal', data);
  }

  /**
   * Emit simulation indicator
   */
  _emitSimulation(type, data) {
    this.io.emit('scenario:simulation', { type, ...data });
  }

  /**
   * Emit camera focus
   */
  _emitCameraFocus(cameraId, reason, priority) {
    this.io.emit('camera:auto-focus', {
      cameraId,
      reason,
      priority,
      scenarioFocus: true
    });
  }

  /**
   * Emit sound
   */
  _emitSound(soundName, options = {}) {
    const soundConfig = this.config.sounds[soundName];
    if (soundConfig) {
      this.io.emit('scenario:play-sound', {
        sound: soundConfig.file || soundName,
        loop: options.loop ?? soundConfig.loop ?? false,
        volume: soundConfig.volume,
        repeatCount: options.repeatCount ?? 1
      });
    } else {
      // Fallback for sounds not in config
      this.io.emit('scenario:play-sound', {
        sound: soundName,
        loop: options.loop ?? false,
        repeatCount: options.repeatCount ?? 1
      });
    }
  }

  /**
   * Stop sound
   */
  _stopSound(soundName) {
    this.io.emit('scenario:stop-sound', { sound: soundName });
  }

  /**
   * Emit soldier video panel
   */
  _emitSoldierVideoPanel(data) {
    this.io.emit('scenario:soldier-video', data);
  }

  /**
   * Emit new camera dialog
   */
  _emitNewCameraDialog(open) {
    this.io.emit('scenario:new-camera-dialog', { open });
  }

  /**
   * Emit summary popup
   */
  _emitSummaryPopup(data) {
    this.io.emit('scenario:summary', data);
  }

  /**
   * Create event in backend
   */
  async _createEvent(eventData) {
    try {
      // Import Event model if using MongoDB
      // For now, just emit via socket
      this.io.emit('event:new', {
        ...eventData,
        timestamp: new Date().toISOString(),
        scenarioId: this.scenario?.id
      });
    } catch (error) {
      console.error('[ScenarioManager] Failed to create event:', error);
    }
  }

  /**
   * Send TTS to radio
   */
  async _sendTTS(message) {
    console.log(`[ScenarioManager] TTS: ${message}`);

    // Emit TTS message to frontend for display
    this.io.emit('scenario:tts', { message, priority: 'high' });

    try {
      const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

      // Generate TTS audio file
      const response = await fetch(`${AI_SERVICE_URL}/tts/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: message, language: 'he' })
      });

      if (response.ok) {
        const ttsData = await response.json();
        console.log(`[ScenarioManager] TTS audio generated: ${ttsData.audio_path || 'success'}`);
      }
    } catch (error) {
      console.error('[ScenarioManager] TTS error:', error.message);
    }
  }

  /**
   * Start recording
   * Note: Recording is handled by the detection loop's event-based recording system.
   * This method emits a socket event that the frontend/recording system can listen to.
   */
  async _startRecording() {
    const cameraId = this.scenario.vehicle.cameraId;

    // Emit recording start event
    this.io.emit('scenario:recording', {
      action: 'start',
      cameraId,
      scenarioId: this.scenario.id,
      reason: `Scenario: ${this.scenario.id}`
    });

    this.scenario.recording = `scenario-${this.scenario.id}`;
    console.log(`[ScenarioManager] Recording started for camera: ${cameraId}`);
  }

  /**
   * Stop recording
   */
  async _stopRecording() {
    if (!this.scenario?.recording) return;

    // Emit recording stop event
    this.io.emit('scenario:recording', {
      action: 'stop',
      scenarioId: this.scenario.id
    });

    console.log(`[ScenarioManager] Recording stopped: ${this.scenario.recording}`);
  }

  /**
   * Build person descriptions for TTS
   */
  _buildPersonDescriptions(persons) {
    return persons.map((p, i) => {
      const parts = [];
      if (p.clothing) parts.push(`בלבוש ${p.clothing}`);
      if (p.weaponType) parts.push(`עם ${p.weaponType}`);
      return `אדם ${i + 1} ${parts.join(' ')}`;
    }).join('. ');
  }

  /**
   * Set timeout
   */
  _setTimeout(name, ms, callback) {
    this._clearTimeout(name);
    this.timeouts[name] = setTimeout(callback, ms);
  }

  /**
   * Clear specific timeout
   */
  _clearTimeout(name) {
    if (this.timeouts[name]) {
      clearTimeout(this.timeouts[name]);
      delete this.timeouts[name];
    }
  }

  /**
   * Clear all timeouts
   */
  _clearTimeouts() {
    Object.keys(this.timeouts).forEach(name => {
      clearTimeout(this.timeouts[name]);
    });
    this.timeouts = {};
  }

  /**
   * Cleanup after scenario ends
   */
  _cleanup() {
    this._clearTimeouts();
    this.scenario = null;
    this.armedPersonsBuffer = [];

    this.io.emit('scenario:ended', {});
    console.log('[ScenarioManager] Cleanup complete');
  }

  /**
   * Setup socket event handlers
   */
  _setupSocketHandlers() {
    // These will be called from socket handler setup in index.js
  }
}

// Singleton instance
let scenarioManagerInstance = null;

export function initScenarioManager(io) {
  if (!scenarioManagerInstance) {
    scenarioManagerInstance = new ScenarioManager(io);
  }
  return scenarioManagerInstance;
}

export function getScenarioManager() {
  return scenarioManagerInstance;
}

export { ScenarioManager };
