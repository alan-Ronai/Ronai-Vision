/**
 * go2rtc Service for HAMAL-AI
 * Manages communication with go2rtc streaming server
 */

import axios from 'axios';

const GO2RTC_URL = process.env.GO2RTC_URL || 'http://localhost:1984';
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

class Go2rtcService {
  constructor() {
    this.baseUrl = GO2RTC_URL;
    this.streams = new Map();
  }

  /**
   * Check if a source is a local file path
   * @param {string} source - Source URL or file path
   * @returns {boolean}
   */
  isLocalFile(source) {
    if (!source) return false;
    // Check if it's a file path (not a URL scheme)
    if (source.startsWith('/') || source.startsWith('./') || source.startsWith('../')) {
      return true;
    }
    // Check for common video extensions without URL scheme
    if (/\.(mp4|mkv|avi|mov|webm|ts|m3u8)$/i.test(source) && !source.includes('://')) {
      return true;
    }
    return false;
  }

  /**
   * Convert a local file to a go2rtc source by using AI service's MJPEG stream
   * This is optimal because:
   * 1. AI service already reads and processes the file
   * 2. go2rtc uses ffmpeg to transcode MJPEG to H264 for WebRTC
   * @param {string} streamId - Stream identifier
   * @returns {string} - go2rtc source URL
   */
  fileToSource(streamId) {
    // Use ffmpeg to transcode MJPEG from AI service to H264 for WebRTC
    // Format: ffmpeg:<input>#video=h264
    const mjpegUrl = `${AI_SERVICE_URL}/api/stream/mjpeg/${streamId}`;
    return `ffmpeg:${mjpegUrl}#video=h264`;
  }

  /**
   * Add a stream to go2rtc
   * @param {string} streamId - Unique stream identifier
   * @param {string} sourceUrl - RTSP URL, file path, or other source
   * @param {boolean} useTcp - Use TCP transport for RTSP streams (default: true)
   * @returns {Promise<object>} - Stream info
   */
  async addStream(streamId, sourceUrl, useTcp = true) {
    try {
      let source = sourceUrl;

      // Handle local file paths - use AI service's MJPEG stream
      if (this.isLocalFile(sourceUrl)) {
        source = this.fileToSource(streamId);
        console.log(`[go2rtc] Local file detected, using AI service MJPEG: ${source}`);
      }
      // Handle RTSP URLs with TCP transport
      else if (useTcp && sourceUrl.startsWith('rtsp://')) {
        // Add TCP transport hint if not already present
        if (!sourceUrl.includes('#')) {
          source = `${sourceUrl}#transport=tcp`;
        } else if (!sourceUrl.includes('transport=')) {
          source = `${sourceUrl}&transport=tcp`;
        }
      }

      // go2rtc API: PUT /api/streams?src=...&name=streamId
      const response = await axios.put(
        `${this.baseUrl}/api/streams`,
        null,
        {
          params: {
            name: streamId,
            src: source
          }
        }
      );

      this.streams.set(streamId, {
        rtspUrl,
        addedAt: new Date(),
        status: 'active'
      });

      console.log(`[go2rtc] Added stream: ${streamId}`);

      return {
        success: true,
        streamId,
        webrtcUrl: `${this.baseUrl}/api/webrtc?src=${streamId}`,
        mseUrl: `${this.baseUrl}/api/ws?src=${streamId}`,
        hlsUrl: `${this.baseUrl}/api/stream.m3u8?src=${streamId}`,
        rtspUrl: `rtsp://localhost:8554/${streamId}`
      };
    } catch (error) {
      console.error(`[go2rtc] Error adding stream ${streamId}:`, error.message);
      throw error;
    }
  }

  /**
   * Remove a stream from go2rtc
   * @param {string} streamId - Stream identifier to remove
   */
  async removeStream(streamId) {
    try {
      await axios.delete(`${this.baseUrl}/api/streams`, {
        params: { name: streamId }
      });

      this.streams.delete(streamId);
      console.log(`[go2rtc] Removed stream: ${streamId}`);

      return { success: true, streamId };
    } catch (error) {
      console.error(`[go2rtc] Error removing stream ${streamId}:`, error.message);
      throw error;
    }
  }

  /**
   * Get all streams from go2rtc
   * @returns {Promise<object>} - All stream configurations
   */
  async getStreams() {
    try {
      const response = await axios.get(`${this.baseUrl}/api/streams`);
      return response.data;
    } catch (error) {
      console.error('[go2rtc] Error getting streams:', error.message);
      throw error;
    }
  }

  /**
   * Get stream info for a specific stream
   * @param {string} streamId - Stream identifier
   * @returns {Promise<object>} - Stream info with producers/consumers
   */
  async getStreamInfo(streamId) {
    try {
      const response = await axios.get(`${this.baseUrl}/api/streams`, {
        params: { name: streamId }
      });
      return response.data;
    } catch (error) {
      console.error(`[go2rtc] Error getting stream info for ${streamId}:`, error.message);
      throw error;
    }
  }

  /**
   * Check if go2rtc is running and accessible
   * @returns {Promise<boolean>}
   */
  async isHealthy() {
    try {
      const response = await axios.get(`${this.baseUrl}/api`, { timeout: 2000 });
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get WebRTC connection URLs for a stream
   * @param {string} streamId - Stream identifier
   * @returns {object} - WebRTC connection info
   */
  getWebRTCInfo(streamId) {
    return {
      // WebRTC WHEP endpoint (browser-compatible)
      whep: `${this.baseUrl}/api/webrtc?src=${streamId}`,
      // MSE WebSocket endpoint (fallback)
      mse: `${this.baseUrl}/api/ws?src=${streamId}`,
      // go2rtc's internal RTSP output
      rtsp: `rtsp://localhost:8554/${streamId}`,
      // HLS fallback
      hls: `${this.baseUrl}/api/stream.m3u8?src=${streamId}`
    };
  }

  /**
   * Get WHIP URL for browser webcam ingestion
   * WHIP = WebRTC-HTTP Ingestion Protocol (browser pushes stream TO server)
   * @param {string} streamId - Stream identifier for the browser webcam
   * @returns {object} - WHIP connection info
   */
  getWHIPInfo(streamId) {
    return {
      // WHIP endpoint - browser sends its webcam stream here
      whip: `${this.baseUrl}/api/webrtc?dst=${streamId}`,
      // Once published, the stream can be consumed via these URLs
      whep: `${this.baseUrl}/api/webrtc?src=${streamId}`,
      mse: `${this.baseUrl}/api/ws?src=${streamId}`,
      rtsp: `rtsp://localhost:8554/${streamId}`,
      hls: `${this.baseUrl}/api/stream.m3u8?src=${streamId}`
    };
  }

  /**
   * Get WHIP info for the pre-configured browser webcam stream
   * The stream is defined in go2rtc.yaml as a self-referencing stream
   * @returns {object} - WHIP connection info
   */
  getBrowserWebcamStreamInfo() {
    // Use the fixed stream name from go2rtc.yaml
    const streamId = 'browser-webcam';

    console.log(`[go2rtc] Using pre-configured browser-webcam stream for WHIP`);

    const whipInfo = this.getWHIPInfo(streamId);

    this.streams.set(streamId, {
      type: 'browser-webcam',
      addedAt: new Date(),
      status: 'waiting'
    });

    return {
      success: true,
      streamId,
      ...whipInfo
    };
  }

  /**
   * Clear all streams from go2rtc
   * Called on startup to ensure clean state
   */
  async clearAllStreams() {
    try {
      const streams = await this.getStreams();
      const streamNames = Object.keys(streams || {});

      if (streamNames.length === 0) {
        console.log('[go2rtc] No streams to clear');
        return { success: true, cleared: 0 };
      }

      console.log(`[go2rtc] Clearing ${streamNames.length} existing streams...`);

      for (const name of streamNames) {
        try {
          await this.removeStream(name);
        } catch (e) {
          console.warn(`[go2rtc] Failed to remove stream ${name}: ${e.message}`);
        }
      }

      this.streams.clear();
      console.log(`[go2rtc] Cleared ${streamNames.length} streams`);
      return { success: true, cleared: streamNames.length };
    } catch (error) {
      console.error('[go2rtc] Error clearing streams:', error.message);
      return { success: false, error: error.message };
    }
  }

  /**
   * Sync cameras from database to go2rtc
   * @param {Array} cameras - Array of camera objects with rtspUrl, filePath, or sourceUrl
   */
  async syncCameras(cameras) {
    const results = [];

    for (const camera of cameras) {
      // Use cameraId (frontend uses this), fallback to _id
      const streamId = camera.cameraId || camera._id;

      // Skip webcam type cameras - they're handled by AI service with FFmpeg device capture
      if (camera.type === 'webcam') {
        console.log(`[go2rtc] Skipping ${streamId}: webcam (handled by AI service)`);
        results.push({
          camera: streamId,
          success: true,
          skipped: true,
          reason: 'webcam handled by AI service'
        });
        continue;
      }

      // Get source URL - can be rtspUrl, filePath, or sourceUrl
      const sourceUrl = camera.rtspUrl || camera.filePath || camera.sourceUrl;

      if (sourceUrl) {
        try {
          const result = await this.addStream(streamId, sourceUrl);
          results.push({ camera: streamId, ...result });
        } catch (error) {
          results.push({
            camera: streamId,
            success: false,
            error: error.message
          });
        }
      } else {
        console.log(`[go2rtc] Skipping ${streamId}: no source URL`);
      }
    }

    return results;
  }
}

// Singleton instance
const go2rtcService = new Go2rtcService();
export default go2rtcService;
