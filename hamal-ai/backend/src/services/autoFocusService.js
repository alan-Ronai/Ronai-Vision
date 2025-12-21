/**
 * Auto Focus Service
 *
 * Manages automatic camera switching when events occur.
 * Tracks priority levels and handles returning to original camera after timeout.
 */

// Priority levels (higher number = higher priority)
const PRIORITY_LEVELS = {
  low: 1,
  medium: 2,
  high: 3,
  critical: 4
};

class AutoFocusService {
  constructor() {
    // Current auto-focus state
    this.isActive = false;
    this.currentPriority = 0;
    this.originalCameraId = null;
    this.focusedCameraId = null;
    this.focusReason = null;
    this.focusEventId = null;

    // Timeout for returning to original camera
    this.returnTimeout = null;
    this.returnTimeoutSeconds = 0;

    // Socket.IO instance (will be set by main app)
    this.io = null;

    // Configuration
    this.enabled = process.env.AUTO_FOCUS_ENABLED !== 'false';
    this.minSeverity = process.env.AUTO_FOCUS_MIN_SEVERITY || 'warning';
    this.debounceMs = 500;
    this.lastSwitchTime = 0;

    // Settings from environment
    this.defaultReturnTimeout = parseInt(process.env.AUTO_FOCUS_TIMEOUT || '30', 10);
    this.sticky = process.env.AUTO_FOCUS_STICKY === 'true';

    console.log(`AutoFocusService initialized (enabled: ${this.enabled})`);
  }

  /**
   * Set the Socket.IO instance
   */
  setIO(io) {
    this.io = io;
  }

  /**
   * Set the current main camera (called when user manually selects)
   */
  setCurrentCamera(cameraId) {
    if (!this.isActive) {
      this.originalCameraId = cameraId;
    }
  }

  /**
   * Cancel auto-focus and return to original camera
   */
  cancel(reason = 'manual_override') {
    if (!this.isActive) return;

    const previousCamera = this.focusedCameraId;
    const originalCamera = this.originalCameraId;

    this._clearState();

    if (this.io) {
      this.io.emit('camera:auto-focus-cancelled', {
        previousCameraId: previousCamera,
        returnToCameraId: originalCamera,
        reason
      });

      // Switch back to original camera
      if (originalCamera) {
        this.io.emit('camera:selected', originalCamera);
      }
    }

    console.log(`AutoFocus cancelled: ${reason}`);
  }

  /**
   * Evaluate if we should switch camera for an event
   */
  evaluateEvent(event, cameraId, severity = 'warning', options = {}) {
    if (!this.enabled) return false;

    const {
      priority = this._severityToPriority(severity),
      returnTimeout = this.defaultReturnTimeout,
      showIndicator = true,
      reason = event?.title || 'אירוע',
      eventId = event?._id || Date.now().toString()
    } = options;

    const priorityValue = PRIORITY_LEVELS[priority] || PRIORITY_LEVELS.medium;

    // Check if we should switch
    if (!this._shouldSwitch(cameraId, priorityValue)) {
      return false;
    }

    // Debounce rapid switches
    const now = Date.now();
    if (now - this.lastSwitchTime < this.debounceMs) {
      console.log('AutoFocus debounced');
      return false;
    }
    this.lastSwitchTime = now;

    // Perform the switch
    return this._doSwitch(cameraId, priority, priorityValue, returnTimeout, reason, eventId, showIndicator);
  }

  /**
   * Check if we should switch to this camera
   */
  _shouldSwitch(cameraId, priorityValue) {
    // Already focused on this camera
    if (this.isActive && this.focusedCameraId === cameraId) {
      return false;
    }

    // Current focus has higher priority
    if (this.isActive && this.currentPriority >= priorityValue) {
      return false;
    }

    // Same as original camera, no need to switch
    if (!this.isActive && this.originalCameraId === cameraId) {
      return false;
    }

    return true;
  }

  /**
   * Perform the camera switch
   */
  _doSwitch(cameraId, priority, priorityValue, returnTimeout, reason, eventId, showIndicator) {
    // Store original camera if not already in auto-focus mode
    if (!this.isActive) {
      // originalCameraId should already be set by setCurrentCamera
      // but if not, we can't return anywhere
      if (!this.originalCameraId) {
        console.log('AutoFocus: No original camera set, cannot switch');
        return false;
      }
    }

    // Clear existing return timeout
    if (this.returnTimeout) {
      clearTimeout(this.returnTimeout);
      this.returnTimeout = null;
    }

    // Update state
    const previousCameraId = this.isActive ? this.focusedCameraId : this.originalCameraId;
    this.isActive = true;
    this.focusedCameraId = cameraId;
    this.currentPriority = priorityValue;
    this.focusReason = reason;
    this.focusEventId = eventId;
    this.returnTimeoutSeconds = returnTimeout;

    // Emit socket event
    if (this.io) {
      this.io.emit('camera:auto-focus', {
        newCameraId: cameraId,
        previousCameraId,
        reason,
        eventId,
        severity: this._priorityToSeverity(priority),
        priority,
        autoReturn: returnTimeout > 0 && !this.sticky,
        returnTimeout,
        showIndicator
      });

      // Also emit camera selection
      this.io.emit('camera:selected', cameraId);
    }

    console.log(`AutoFocus: Switched to ${cameraId} (priority: ${priority}, reason: ${reason})`);

    // Set return timeout if configured
    if (returnTimeout > 0 && !this.sticky) {
      this.returnTimeout = setTimeout(() => {
        this._returnToOriginal();
      }, returnTimeout * 1000);
    }

    return true;
  }

  /**
   * Return to original camera after timeout
   */
  _returnToOriginal() {
    if (!this.isActive) return;

    const focusedCamera = this.focusedCameraId;
    const originalCamera = this.originalCameraId;

    this._clearState();

    if (this.io && originalCamera) {
      this.io.emit('camera:auto-focus-return', {
        previousCameraId: focusedCamera,
        returnToCameraId: originalCamera
      });

      this.io.emit('camera:selected', originalCamera);
    }

    console.log(`AutoFocus: Returned to original camera ${originalCamera}`);
  }

  /**
   * Clear auto-focus state
   */
  _clearState() {
    if (this.returnTimeout) {
      clearTimeout(this.returnTimeout);
      this.returnTimeout = null;
    }

    this.isActive = false;
    this.currentPriority = 0;
    this.focusedCameraId = null;
    this.focusReason = null;
    this.focusEventId = null;
    this.returnTimeoutSeconds = 0;
  }

  /**
   * Convert severity to priority
   */
  _severityToPriority(severity) {
    switch (severity) {
      case 'critical': return 'critical';
      case 'warning': return 'high';
      case 'info': return 'medium';
      default: return 'medium';
    }
  }

  /**
   * Convert priority to severity
   */
  _priorityToSeverity(priority) {
    switch (priority) {
      case 'critical': return 'critical';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'info';
      default: return 'info';
    }
  }

  /**
   * Get current auto-focus state
   */
  getState() {
    return {
      enabled: this.enabled,
      isActive: this.isActive,
      originalCameraId: this.originalCameraId,
      focusedCameraId: this.focusedCameraId,
      currentPriority: this.currentPriority,
      focusReason: this.focusReason,
      focusEventId: this.focusEventId,
      returnTimeoutSeconds: this.returnTimeoutSeconds,
      sticky: this.sticky
    };
  }

  /**
   * Update configuration
   */
  updateConfig(config) {
    if (config.enabled !== undefined) {
      this.enabled = config.enabled;
    }
    if (config.minSeverity !== undefined) {
      this.minSeverity = config.minSeverity;
    }
    if (config.defaultReturnTimeout !== undefined) {
      this.defaultReturnTimeout = config.defaultReturnTimeout;
    }
    if (config.sticky !== undefined) {
      this.sticky = config.sticky;
    }
  }
}

// Singleton instance
const autoFocusService = new AutoFocusService();

export default autoFocusService;
export { AutoFocusService, PRIORITY_LEVELS };
