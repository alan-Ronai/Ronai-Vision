import express from 'express';
import http from 'http';
import https from 'https';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const router = express.Router();

// Store active HLS processes
const hlsProcesses = new Map();

/**
 * POST /api/stream/start-hls
 * Start RTSP to HLS conversion for a camera
 */
router.post('/start-hls', async (req, res) => {
  try {
    const { cameraId, rtspUrl, username, password } = req.body;

    if (!cameraId || !rtspUrl) {
      return res.status(400).json({ error: 'cameraId and rtspUrl are required' });
    }

    // Stop existing process for this camera if any
    if (hlsProcesses.has(cameraId)) {
      const existingProcess = hlsProcesses.get(cameraId);
      existingProcess.kill('SIGTERM');
      hlsProcesses.delete(cameraId);
    }

    // Create HLS output directory
    const hlsPath = process.env.HLS_PATH || path.join(__dirname, '../../../data/hls');
    const cameraHlsPath = path.join(hlsPath, cameraId);

    if (!fs.existsSync(cameraHlsPath)) {
      fs.mkdirSync(cameraHlsPath, { recursive: true });
    }

    // Build RTSP URL with credentials if provided
    let fullRtspUrl = rtspUrl;
    if (username && password) {
      const url = new URL(rtspUrl);
      url.username = username;
      url.password = password;
      fullRtspUrl = url.toString();
    }

    // Start FFmpeg process
    const ffmpegArgs = [
      '-rtsp_transport', 'tcp',
      '-i', fullRtspUrl,
      '-c:v', 'copy',
      '-c:a', 'aac',
      '-f', 'hls',
      '-hls_time', '2',
      '-hls_list_size', '5',
      '-hls_flags', 'delete_segments+append_list',
      '-hls_segment_filename', path.join(cameraHlsPath, 'segment_%03d.ts'),
      path.join(cameraHlsPath, 'stream.m3u8')
    ];

    console.log(`Starting HLS stream for ${cameraId}:`, ffmpegArgs.join(' '));

    const ffmpegProcess = spawn('ffmpeg', ffmpegArgs);
    hlsProcesses.set(cameraId, ffmpegProcess);

    ffmpegProcess.stderr.on('data', (data) => {
      const output = data.toString();
      if (output.includes('error') || output.includes('Error')) {
        console.error(`FFmpeg error for ${cameraId}:`, output);
      }
    });

    ffmpegProcess.on('exit', (code) => {
      console.log(`FFmpeg process for ${cameraId} exited with code ${code}`);
      hlsProcesses.delete(cameraId);

      const io = req.app.get('io');
      io.emit('stream:stopped', { cameraId, code });
    });

    res.json({
      message: 'HLS stream started',
      cameraId,
      hlsUrl: `/hls/${cameraId}/stream.m3u8`
    });
  } catch (error) {
    console.error('Error starting HLS stream:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/stream/stop-hls
 * Stop HLS conversion for a camera
 */
router.post('/stop-hls', async (req, res) => {
  try {
    const { cameraId } = req.body;

    if (!cameraId) {
      return res.status(400).json({ error: 'cameraId is required' });
    }

    if (!hlsProcesses.has(cameraId)) {
      return res.status(404).json({ error: 'No active stream for this camera' });
    }

    const process = hlsProcesses.get(cameraId);
    process.kill('SIGTERM');
    hlsProcesses.delete(cameraId);

    res.json({
      message: 'HLS stream stopped',
      cameraId
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/stream/status
 * Get status of all streams
 */
router.get('/status', (req, res) => {
  const streams = Array.from(hlsProcesses.keys()).map(cameraId => ({
    cameraId,
    hlsUrl: `/hls/${cameraId}/stream.m3u8`,
    active: true
  }));

  res.json({ streams });
});

/**
 * GET /api/stream/mjpeg/:cameraId
 * Proxy MJPEG stream from AI service using native http module
 */
router.get('/mjpeg/:cameraId', (req, res) => {
  const { cameraId } = req.params;
  const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000';
  const targetUrl = `${aiServiceUrl}/api/stream/mjpeg/${encodeURIComponent(cameraId)}`;

  console.log(`[Stream] Proxying MJPEG for camera: ${cameraId}`);

  try {
    const url = new URL(targetUrl);
    const client = url.protocol === 'https:' ? https : http;

    const proxyReq = client.get(targetUrl, (proxyRes) => {
      res.setHeader('Content-Type', proxyRes.headers['content-type'] || 'multipart/x-mixed-replace; boundary=frame');
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.setHeader('Connection', 'keep-alive');

      proxyRes.pipe(res);

      proxyRes.on('error', (err) => {
        console.error(`[Stream] Proxy response error for ${cameraId}:`, err.message);
      });
    });

    proxyReq.on('error', (err) => {
      console.error(`[Stream] Proxy request error for ${cameraId}:`, err.message);
      if (!res.headersSent) {
        res.status(502).json({ error: 'Failed to connect to AI service', message: err.message });
      }
    });

    req.on('close', () => {
      proxyReq.destroy();
    });

  } catch (error) {
    console.error(`[Stream] Error proxying stream for ${cameraId}:`, error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: error.message });
    }
  }
});

/**
 * GET /api/stream/sse/:cameraId
 * Proxy SSE stream from AI service
 */
router.get('/sse/:cameraId', (req, res) => {
  const { cameraId } = req.params;
  const fps = req.query.fps || 5;
  const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000';
  const targetUrl = `${aiServiceUrl}/api/stream/sse/${encodeURIComponent(cameraId)}?fps=${fps}`;

  console.log(`[Stream] Proxying SSE for camera: ${cameraId}`);

  try {
    const url = new URL(targetUrl);
    const client = url.protocol === 'https:' ? https : http;

    const proxyReq = client.get(targetUrl, (proxyRes) => {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      proxyRes.pipe(res);

      proxyRes.on('error', (err) => {
        console.error(`[Stream] SSE proxy error for ${cameraId}:`, err.message);
      });
    });

    proxyReq.on('error', (err) => {
      console.error(`[Stream] SSE request error for ${cameraId}:`, err.message);
      if (!res.headersSent) {
        res.status(502).json({ error: 'Failed to connect to AI service' });
      }
    });

    req.on('close', () => {
      proxyReq.destroy();
    });

  } catch (error) {
    console.error(`[Stream] SSE error for ${cameraId}:`, error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: error.message });
    }
  }
});

/**
 * POST /api/stream/frame
 * Receive frame for broadcasting (from AI service)
 */
router.post('/frame', express.raw({ type: 'image/*', limit: '10mb' }), (req, res) => {
  try {
    const { cameraId } = req.query;

    if (!cameraId) {
      return res.status(400).json({ error: 'cameraId query param is required' });
    }

    const io = req.app.get('io');
    const frameData = req.body.toString('base64');

    io.emit(`frame:${cameraId}`, {
      cameraId,
      frame: frameData,
      timestamp: new Date()
    });

    res.json({ message: 'Frame broadcasted' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Cleanup on process exit
process.on('SIGINT', () => {
  console.log('Stopping all HLS streams...');
  for (const [cameraId, proc] of hlsProcesses.entries()) {
    console.log(`Stopping stream for ${cameraId}`);
    proc.kill('SIGTERM');
  }
  process.exit();
});

export default router;
