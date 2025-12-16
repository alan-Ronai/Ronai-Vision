/**
 * Camera Service - Manages camera connections and frame processing
 */
import axios from 'axios';
import Camera from '../models/Camera.js';

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

class CameraService {
  constructor(io) {
    this.io = io;
    this.frameIntervals = new Map();
    this.lastFrames = new Map();
  }

  /**
   * Start frame capture for a camera
   */
  async startCapture(cameraId, frameRate = 2) {
    if (this.frameIntervals.has(cameraId)) {
      console.log(`Frame capture already running for ${cameraId}`);
      return;
    }

    const camera = await Camera.findOne({ cameraId }).lean();
    if (!camera) {
      throw new Error(`Camera ${cameraId} not found`);
    }

    console.log(`Starting frame capture for ${cameraId} at ${frameRate} FPS`);

    const interval = setInterval(async () => {
      try {
        // Get frame from existing API if available
        const existingApiUrl = process.env.EXISTING_API_URL || 'http://localhost:8000';
        const response = await axios.get(
          `${existingApiUrl}/api/stream/snapshot/${cameraId}`,
          { responseType: 'arraybuffer', timeout: 5000 }
        );

        const frameBuffer = Buffer.from(response.data);
        this.lastFrames.set(cameraId, {
          buffer: frameBuffer,
          timestamp: new Date()
        });

        // Send to AI service for detection
        if (camera.aiEnabled) {
          await this.processFrame(cameraId, frameBuffer);
        }

        // Update camera status
        await Camera.findOneAndUpdate(
          { cameraId },
          { status: 'online', lastSeen: new Date() }
        );
      } catch (error) {
        console.error(`Error capturing frame from ${cameraId}:`, error.message);

        await Camera.findOneAndUpdate(
          { cameraId },
          { status: 'error', lastError: error.message }
        );
      }
    }, 1000 / frameRate);

    this.frameIntervals.set(cameraId, interval);
  }

  /**
   * Stop frame capture for a camera
   */
  stopCapture(cameraId) {
    const interval = this.frameIntervals.get(cameraId);
    if (interval) {
      clearInterval(interval);
      this.frameIntervals.delete(cameraId);
      console.log(`Stopped frame capture for ${cameraId}`);
    }
  }

  /**
   * Process frame through AI service
   */
  async processFrame(cameraId, frameBuffer) {
    try {
      const formData = new FormData();
      formData.append('file', new Blob([frameBuffer]), 'frame.jpg');
      formData.append('camera_id', cameraId);

      const response = await axios.post(
        `${AI_SERVICE_URL}/detect`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 10000
        }
      );

      const detection = response.data;

      // If significant detection, emit event
      if (detection.hasDetection) {
        this.io.emit('detection', {
          cameraId,
          detection,
          timestamp: new Date()
        });
      }

      return detection;
    } catch (error) {
      console.error(`AI detection error for ${cameraId}:`, error.message);
      return null;
    }
  }

  /**
   * Get last frame for a camera
   */
  getLastFrame(cameraId) {
    return this.lastFrames.get(cameraId);
  }

  /**
   * Get all active cameras
   */
  getActiveCameras() {
    return Array.from(this.frameIntervals.keys());
  }

  /**
   * Stop all captures
   */
  stopAll() {
    for (const cameraId of this.frameIntervals.keys()) {
      this.stopCapture(cameraId);
    }
  }
}

export default CameraService;
