import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const backendUrl = env.VITE_BACKEND_URL || 'http://localhost:3000';
  const aiServiceUrl = env.VITE_AI_SERVICE_URL || 'http://localhost:8000';
  const go2rtcUrl = env.VITE_GO2RTC_URL || 'http://localhost:1984';

  return {
    plugins: [
      react(),
      basicSsl()  // Generates proper self-signed certificate
    ],
    server: {
      port: 5173,
      host: '0.0.0.0',  // Allow external connections
      proxy: {
        // Backend API
        '/api': {
          target: backendUrl,
          changeOrigin: true
        },
        '/socket.io': {
          target: backendUrl,
          ws: true
        },
        '/hls': {
          target: backendUrl,
          changeOrigin: true
        },
        '/clips': {
          target: backendUrl,
          changeOrigin: true
        },
        // AI Service endpoints
        '/detection': {
          target: aiServiceUrl,
          changeOrigin: true
        },
        '/tts': {
          target: aiServiceUrl,
          changeOrigin: true
        },
        '/scenario-rules': {
          target: aiServiceUrl,
          changeOrigin: true
        },
        '/transcription': {
          target: aiServiceUrl,
          changeOrigin: true
        },
        // go2rtc endpoints (for WebRTC/WHIP)
        '/go2rtc': {
          target: go2rtcUrl,
          changeOrigin: true,
          ws: true,  // Enable WebSocket proxying
          rewrite: (path) => path.replace(/^\/go2rtc/, '')
        }
      }
    }
  };
});
