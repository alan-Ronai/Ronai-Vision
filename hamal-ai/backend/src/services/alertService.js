/**
 * Alert Service - Handles alert logic and notifications
 */
import Event from '../models/Event.js';

class AlertService {
  constructor(io) {
    this.io = io;
    this.activeEmergency = null;
    this.alertThresholds = {
      minPeopleForAlert: 2,
      weaponConfidenceThreshold: 0.5,
      vehicleAlertClasses: ['truck', 'bus']
    };
  }

  /**
   * Process detection and determine if alert is needed
   */
  async processDetection(detection, cameraId) {
    const { people, vehicles, weapons } = detection;

    let severity = 'info';
    let title = '';
    let details = {};

    // Critical: Weapon detected
    if (weapons && weapons.length > 0) {
      const confirmedWeapons = weapons.filter(
        w => w.confidence >= this.alertThresholds.weaponConfidenceThreshold
      );
      if (confirmedWeapons.length > 0) {
        severity = 'critical';
        title = 'זיהוי נשק!';
        details.weapons = confirmedWeapons;
        details.people = { count: people?.length || 0, armed: true };
      }
    }

    // Warning: Multiple people
    if (severity !== 'critical' && people && people.length >= this.alertThresholds.minPeopleForAlert) {
      severity = 'warning';
      title = `זיהוי ${people.length} אנשים`;
      details.people = { count: people.length, armed: false };
    }

    // Warning: Suspicious vehicle
    if (severity === 'info' && vehicles && vehicles.length > 0) {
      const suspiciousVehicle = vehicles.find(
        v => this.alertThresholds.vehicleAlertClasses.includes(v.type)
      );
      if (suspiciousVehicle) {
        severity = 'warning';
        title = `זיהוי רכב חשוד: ${suspiciousVehicle.type}`;
        details.vehicle = suspiciousVehicle;
      }
    }

    // If alert needed, create event
    if (severity !== 'info') {
      await this.createAlert({
        severity,
        title,
        cameraId,
        details: {
          ...details,
          detection
        }
      });
    }

    return { severity, title, details };
  }

  /**
   * Create alert event
   */
  async createAlert({ severity, title, cameraId, details }) {
    try {
      const event = new Event({
        type: 'detection',
        severity,
        title,
        cameraId,
        source: `camera-${cameraId}`,
        details
      });

      await event.save();

      // Emit to all clients
      this.io.emit('event:new', event.toObject());

      // Start emergency mode if critical
      if (severity === 'critical') {
        this.startEmergency(event);
      }

      return event;
    } catch (error) {
      console.error('Error creating alert:', error);
      throw error;
    }
  }

  /**
   * Start emergency mode
   */
  startEmergency(event) {
    if (this.activeEmergency) {
      console.log('Emergency already active');
      return;
    }

    console.log('🚨 STARTING EMERGENCY MODE');
    this.activeEmergency = {
      eventId: event._id,
      startedAt: new Date(),
      title: event.title,
      details: event.details
    };

    this.io.emit('emergency:start', {
      eventId: event._id,
      title: event.title,
      details: event.details,
      cameraId: event.cameraId,
      timestamp: new Date()
    });

    // Auto-trigger simulations
    this.triggerEmergencyActions(event);
  }

  /**
   * End emergency mode
   */
  endEmergency(operator = 'system') {
    if (!this.activeEmergency) {
      return;
    }

    console.log('🏁 ENDING EMERGENCY MODE');
    const emergency = this.activeEmergency;
    this.activeEmergency = null;

    this.io.emit('emergency:end', {
      eventId: emergency.eventId,
      endedAt: new Date(),
      duration: Date.now() - emergency.startedAt.getTime(),
      endedBy: operator
    });
  }

  /**
   * Trigger automatic emergency actions
   */
  async triggerEmergencyActions(event) {
    // Simulate phone call to duty officer
    setTimeout(() => {
      this.io.emit('event:new', {
        type: 'simulation',
        severity: 'warning',
        title: 'חיוג למפקד תורן',
        details: { simulation: 'phone_call' },
        createdAt: new Date()
      });
    }, 2000);

    // Simulate drone dispatch
    setTimeout(() => {
      this.io.emit('event:new', {
        type: 'simulation',
        severity: 'warning',
        title: 'רחפן הוקפץ',
        details: { simulation: 'drone_dispatch' },
        createdAt: new Date()
      });
    }, 4000);
  }

  /**
   * Get current emergency status
   */
  getEmergencyStatus() {
    return {
      active: this.activeEmergency !== null,
      emergency: this.activeEmergency
    };
  }

  /**
   * Update alert thresholds
   */
  updateThresholds(thresholds) {
    this.alertThresholds = {
      ...this.alertThresholds,
      ...thresholds
    };
  }
}

export default AlertService;
