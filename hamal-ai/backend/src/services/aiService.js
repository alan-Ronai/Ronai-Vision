/**
 * AI Service Client - Communicates with Python AI service
 */
import axios from 'axios';
import FormData from 'form-data';

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

class AIService {
  constructor() {
    this.client = axios.create({
      baseURL: AI_SERVICE_URL,
      timeout: 30000
    });
  }

  /**
   * Check AI service health
   */
  async checkHealth() {
    try {
      const response = await this.client.get('/health');
      return {
        healthy: true,
        ...response.data
      };
    } catch (error) {
      return {
        healthy: false,
        error: error.message
      };
    }
  }

  /**
   * Run detection on a frame
   */
  async detect(frameBuffer, cameraId = 'unknown') {
    try {
      const formData = new FormData();
      formData.append('file', frameBuffer, {
        filename: 'frame.jpg',
        contentType: 'image/jpeg'
      });

      const response = await this.client.post('/detect', formData, {
        params: { camera_id: cameraId },
        headers: formData.getHeaders()
      });

      return response.data;
    } catch (error) {
      console.error('Detection error:', error.message);
      throw error;
    }
  }

  /**
   * Verify vehicle with Gemini (2 frames)
   */
  async verifyVehicle(frames) {
    try {
      const formData = new FormData();

      frames.forEach((frame, i) => {
        formData.append('files', frame, {
          filename: `frame_${i}.jpg`,
          contentType: 'image/jpeg'
        });
      });

      const response = await this.client.post('/verify-vehicle', formData, {
        headers: formData.getHeaders()
      });

      return response.data;
    } catch (error) {
      console.error('Vehicle verification error:', error.message);
      throw error;
    }
  }

  /**
   * Analyze people in frame
   */
  async analyzePeople(frameBuffer) {
    try {
      const formData = new FormData();
      formData.append('file', frameBuffer, {
        filename: 'frame.jpg',
        contentType: 'image/jpeg'
      });

      const response = await this.client.post('/analyze-people', formData, {
        headers: formData.getHeaders()
      });

      return response.data;
    } catch (error) {
      console.error('People analysis error:', error.message);
      throw error;
    }
  }

  /**
   * Generate Hebrew TTS audio
   */
  async textToSpeech(text) {
    try {
      const response = await this.client.post('/tts', null, {
        params: { text }
      });

      return response.data;
    } catch (error) {
      console.error('TTS error:', error.message);
      throw error;
    }
  }

  /**
   * Analyze scene with Gemini
   */
  async analyzeScene(frameBuffer, prompt = null) {
    try {
      const formData = new FormData();
      formData.append('file', frameBuffer, {
        filename: 'frame.jpg',
        contentType: 'image/jpeg'
      });

      const params = {};
      if (prompt) {
        params.prompt = prompt;
      }

      const response = await this.client.post('/analyze-scene', formData, {
        params,
        headers: formData.getHeaders()
      });

      return response.data;
    } catch (error) {
      console.error('Scene analysis error:', error.message);
      throw error;
    }
  }
}

// Export singleton instance
export default new AIService();
