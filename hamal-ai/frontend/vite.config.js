import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const backendUrl = env.VITE_BACKEND_URL || 'http://localhost:3000';

  return {
    plugins: [
      react(),
      basicSsl()  // Generates proper self-signed certificate
    ],
    server: {
      port: 5173,
      host: '0.0.0.0',  // Allow external connections
      proxy: {
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
        }
      }
    }
  };
});
