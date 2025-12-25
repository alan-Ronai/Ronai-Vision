/**
 * AudioTransmitter Component
 *
 * Provides audio transmission to the radio system via RTP.
 * Features:
 * - Live microphone recording with VAD (voice activity detection)
 * - Audio file upload
 * - Text-to-Speech transmission
 * - Transmission status display
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';

// Use relative URLs to leverage Vite proxy (avoids mixed content issues with HTTPS)
const BACKEND_URL = '';
const AI_SERVICE_PROXY = '';  // /tts endpoint is proxied to AI service

// Audio processing constants
const SAMPLE_RATE = 16000;
const VAD_SILENCE_THRESHOLD = 0.01; // RMS threshold for silence detection
const VAD_SILENCE_DURATION = 1500; // ms of silence before auto-stop

export default function AudioTransmitter({ isOpen, onClose }) {
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);

  // Transmission state
  const [isTransmitting, setIsTransmitting] = useState(false);
  const [transmitStatus, setTransmitStatus] = useState(null);
  const [txStats, setTxStats] = useState(null);

  // TTS state
  const [ttsText, setTtsText] = useState('');
  const [isTTSProcessing, setIsTTSProcessing] = useState(false);

  // Settings
  const [useVAD, setUseVAD] = useState(true);

  // Refs for audio processing
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const recordingTimerRef = useRef(null);
  const silenceTimerRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Fetch TX stats when modal opens (not polling)
  useEffect(() => {
    if (!isOpen) return;

    const fetchStats = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/radio/transmit/stats`, {
          signal: AbortSignal.timeout(3000) // 3 second timeout
        });
        if (response.ok) {
          const data = await response.json();
          setTxStats(data);
        } else {
          // Service returned error
          setTxStats({ connected: false, error: `HTTP ${response.status}` });
        }
      } catch (error) {
        // Service unavailable - don't spam console
        setTxStats({ connected: false, error: 'שירות AI לא זמין' });
      }
    };

    fetchStats();
    // Only poll every 30 seconds instead of 5 to reduce noise
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, [isOpen]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  /**
   * Start recording from microphone
   */
  const startRecording = async () => {
    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      streamRef.current = stream;

      // Set up audio context for level metering and VAD
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE
      });

      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);

      // Start level metering
      startLevelMetering();

      // Set up MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Convert recorded audio to PCM and transmit
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        await processAndTransmit(blob);
      };

      // Start recording
      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);
      setRecordingTime(0);

      // Start recording timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Failed to start recording:', error);
      setTransmitStatus({ error: 'שגיאה בגישה למיקרופון' });
    }
  };

  /**
   * Stop recording
   */
  const stopRecording = useCallback(() => {
    // Stop timers
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Stop media recorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    // Stop stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setIsRecording(false);
    setAudioLevel(0);
  }, []);

  /**
   * Start level metering with VAD
   */
  const startLevelMetering = () => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const checkLevel = () => {
      if (!isRecording && !analyserRef.current) return;

      analyser.getByteFrequencyData(dataArray);

      // Calculate RMS level
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const normalized = dataArray[i] / 255;
        sum += normalized * normalized;
      }
      const rms = Math.sqrt(sum / dataArray.length);
      setAudioLevel(rms);

      // VAD: Check for silence
      if (useVAD && isRecording) {
        if (rms < VAD_SILENCE_THRESHOLD) {
          // Below threshold - start silence timer
          if (!silenceTimerRef.current) {
            silenceTimerRef.current = setTimeout(() => {
              console.log('VAD: Silence detected, stopping recording');
              stopRecording();
            }, VAD_SILENCE_DURATION);
          }
        } else {
          // Above threshold - reset silence timer
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = null;
          }
        }
      }

      animationFrameRef.current = requestAnimationFrame(checkLevel);
    };

    checkLevel();
  };

  /**
   * Process recorded audio and transmit
   */
  const processAndTransmit = async (blob) => {
    setIsTransmitting(true);
    setTransmitStatus({ message: 'ממיר אודיו...' });

    try {
      // Convert WebM to PCM using Web Audio API
      const arrayBuffer = await blob.arrayBuffer();
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE
      });

      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      // Get mono channel data
      const channelData = audioBuffer.getChannelData(0);

      // Convert float32 to int16
      const pcmData = new Int16Array(channelData.length);
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      // Convert to base64
      const base64 = arrayBufferToBase64(pcmData.buffer);

      // Send to backend
      setTransmitStatus({ message: 'משדר...' });

      const response = await fetch(`${BACKEND_URL}/api/radio/transmit/audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_base64: base64,
          sample_rate: SAMPLE_RATE,
          auto_ptt: true
        })
      });

      const result = await response.json();

      if (result.success) {
        setTransmitStatus({
          success: true,
          message: `שודר בהצלחה - ${result.packets_sent || 0} חבילות`
        });
      } else {
        setTransmitStatus({
          error: result.message || 'שגיאה בשידור'
        });
      }

      audioContext.close();

    } catch (error) {
      console.error('Transmission error:', error);
      setTransmitStatus({ error: 'שגיאה בעיבוד האודיו' });
    } finally {
      setIsTransmitting(false);
    }
  };

  /**
   * Handle file upload
   */
  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsTransmitting(true);
    setTransmitStatus({ message: 'מעלה קובץ...' });

    try {
      // Read file as array buffer
      const arrayBuffer = await file.arrayBuffer();

      // Check if it's a WAV file
      const isWav = file.name.toLowerCase().endsWith('.wav') ||
                    file.type === 'audio/wav';

      let pcmData;
      let sampleRate = SAMPLE_RATE;

      if (isWav) {
        // Parse WAV header
        const dataView = new DataView(arrayBuffer);

        // Check RIFF header
        if (dataView.getUint32(0, false) !== 0x52494646) { // 'RIFF'
          throw new Error('Invalid WAV file');
        }

        // Get sample rate from header
        sampleRate = dataView.getUint32(24, true);

        // Find data chunk
        let offset = 12;
        while (offset < arrayBuffer.byteLength - 8) {
          const chunkId = dataView.getUint32(offset, false);
          const chunkSize = dataView.getUint32(offset + 4, true);

          if (chunkId === 0x64617461) { // 'data'
            pcmData = new Uint8Array(arrayBuffer, offset + 8, chunkSize);
            break;
          }
          offset += 8 + chunkSize;
          if (chunkSize % 2 === 1) offset++; // Word alignment
        }

        if (!pcmData) {
          throw new Error('No data chunk in WAV file');
        }
      } else {
        // Decode other audio formats using Web Audio API
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        sampleRate = audioBuffer.sampleRate;
        const channelData = audioBuffer.getChannelData(0);

        // Convert to int16
        const int16Data = new Int16Array(channelData.length);
        for (let i = 0; i < channelData.length; i++) {
          const s = Math.max(-1, Math.min(1, channelData[i]));
          int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        pcmData = new Uint8Array(int16Data.buffer);

        audioContext.close();
      }

      // Convert to base64
      const base64 = arrayBufferToBase64(pcmData.buffer);

      setTransmitStatus({ message: 'משדר...' });

      const response = await fetch(`${BACKEND_URL}/api/radio/transmit/audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_base64: base64,
          sample_rate: sampleRate,
          auto_ptt: true
        })
      });

      const result = await response.json();

      if (result.success) {
        setTransmitStatus({
          success: true,
          message: `קובץ שודר בהצלחה - ${result.packets_sent || 0} חבילות`
        });
      } else {
        setTransmitStatus({
          error: result.message || 'שגיאה בשידור'
        });
      }

    } catch (error) {
      console.error('File upload error:', error);
      setTransmitStatus({ error: 'שגיאה בעיבוד הקובץ' });
    } finally {
      setIsTransmitting(false);
      event.target.value = ''; // Reset file input
    }
  };

  /**
   * Send TTS text to radio
   */
  const handleTTSSend = async () => {
    if (!ttsText.trim()) return;

    setIsTTSProcessing(true);
    setTransmitStatus({ message: 'ממיר טקסט לדיבור...' });

    try {
      // Call TTS service to convert text to audio
      const ttsResponse = await fetch(`${AI_SERVICE_PROXY}/tts/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: ttsText,
          language: 'he'  // Hebrew
        })
      });

      if (!ttsResponse.ok) {
        throw new Error(`TTS failed: ${ttsResponse.status}`);
      }

      const ttsResult = await ttsResponse.json();

      if (!ttsResult.audio_base64) {
        throw new Error('No audio returned from TTS');
      }

      setTransmitStatus({ message: 'משדר...' });

      // Send the audio to radio
      const transmitResponse = await fetch(`${BACKEND_URL}/api/radio/transmit/audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_base64: ttsResult.audio_base64,
          sample_rate: ttsResult.sample_rate || 16000,
          auto_ptt: true
        })
      });

      const transmitResult = await transmitResponse.json();

      if (transmitResult.success) {
        setTransmitStatus({
          success: true,
          message: `הודעה שודרה בהצלחה - ${transmitResult.packets_sent || 0} חבילות`
        });
        setTtsText(''); // Clear the text after successful send
      } else {
        setTransmitStatus({
          error: transmitResult.message || 'שגיאה בשידור'
        });
      }

    } catch (error) {
      console.error('TTS error:', error);
      setTransmitStatus({ error: 'שגיאה בהמרת טקסט לדיבור' });
    } finally {
      setIsTTSProcessing(false);
    }
  };

  /**
   * Convert ArrayBuffer to base64
   */
  const arrayBufferToBase64 = (buffer) => {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  };

  /**
   * Format time as MM:SS
   */
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-md mx-4 border border-gray-700">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span>📻</span>
            <span>שידור לקשר</span>
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white p-1"
          >
            ✕
          </button>
        </div>

        {/* Connection Status */}
        <div className={`mb-4 p-3 rounded-lg ${
          txStats?.connected
            ? 'bg-green-900/30 border border-green-700'
            : txStats?.error
            ? 'bg-yellow-900/30 border border-yellow-700'
            : 'bg-red-900/30 border border-red-700'
        }`}>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-300">
              {txStats?.connected
                ? '🟢 מחובר לשרת TX'
                : txStats?.error
                ? `🟡 ${txStats.error}`
                : '🔴 לא מחובר'}
            </span>
            {txStats?.packets_sent !== undefined && (
              <span className="text-gray-400">
                {txStats.packets_sent} חבילות נשלחו
              </span>
            )}
          </div>
        </div>

        {/* TTS Section */}
        <div className="mb-6">
          <h3 className="text-gray-300 mb-3 font-medium">הודעה קולית (טקסט לדיבור)</h3>
          <div className="space-y-3">
            <textarea
              value={ttsText}
              onChange={(e) => setTtsText(e.target.value)}
              placeholder="הקלד הודעה בעברית לשידור..."
              className="w-full h-24 bg-gray-700 border border-gray-600 rounded-lg p-3 text-white placeholder-gray-500 resize-none focus:outline-none focus:border-blue-500"
              dir="rtl"
              disabled={isTTSProcessing || isTransmitting}
            />
            <button
              onClick={handleTTSSend}
              disabled={!ttsText.trim() || isTTSProcessing || isTransmitting || isRecording}
              className={`w-full py-3 px-4 rounded-lg font-medium flex items-center justify-center gap-2 ${
                !ttsText.trim() || isTTSProcessing || isTransmitting || isRecording
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isTTSProcessing ? (
                <>
                  <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>ממיר ומשדר...</span>
                </>
              ) : (
                <>
                  <span>🔊</span>
                  <span>שדר הודעה</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Recording Section */}
        <div className="mb-6">
          <h3 className="text-gray-300 mb-3 font-medium">הקלטה מהמיקרופון</h3>

          {/* Audio Level Meter */}
          <div className="mb-3">
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-75 ${
                  audioLevel > 0.3 ? 'bg-red-500' :
                  audioLevel > 0.1 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${Math.min(100, audioLevel * 300)}%` }}
              />
            </div>
          </div>

          {/* Recording Controls */}
          <div className="flex items-center gap-3">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isTransmitting || isTTSProcessing}
              className={`flex-1 py-3 px-4 rounded-lg font-medium flex items-center justify-center gap-2 ${
                isRecording
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              } disabled:opacity-50`}
            >
              {isRecording ? (
                <>
                  <span className="w-3 h-3 bg-white rounded-sm" />
                  <span>עצור ({formatTime(recordingTime)})</span>
                </>
              ) : (
                <>
                  <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                  <span>הקלט</span>
                </>
              )}
            </button>
          </div>

          {/* VAD Toggle */}
          <label className="flex items-center gap-2 mt-3 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={useVAD}
              onChange={(e) => setUseVAD(e.target.checked)}
              className="rounded border-gray-600 bg-gray-700"
            />
            <span>עצירה אוטומטית בשקט (VAD)</span>
          </label>
        </div>

        {/* File Upload Section */}
        <div className="mb-6">
          <h3 className="text-gray-300 mb-3 font-medium">העלאת קובץ אודיו</h3>
          <label className={`block w-full py-3 px-4 rounded-lg border-2 border-dashed border-gray-600 hover:border-gray-500 text-center cursor-pointer ${
            isTransmitting || isTTSProcessing ? 'opacity-50 cursor-not-allowed' : ''
          }`}>
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              disabled={isTransmitting || isRecording || isTTSProcessing}
              className="hidden"
            />
            <span className="text-gray-400">
              📁 לחץ לבחירת קובץ (WAV, MP3, OGG)
            </span>
          </label>
        </div>

        {/* Status Display */}
        {transmitStatus && (
          <div className={`p-3 rounded-lg text-sm ${
            transmitStatus.error
              ? 'bg-red-900/30 border border-red-700 text-red-300'
              : transmitStatus.success
              ? 'bg-green-900/30 border border-green-700 text-green-300'
              : 'bg-blue-900/30 border border-blue-700 text-blue-300'
          }`}>
            {transmitStatus.error || transmitStatus.message}
          </div>
        )}

        {/* Transmitting Indicator */}
        {(isTransmitting || isTTSProcessing) && (
          <div className="mt-4 flex items-center justify-center gap-2 text-yellow-400">
            <div className="w-2 h-2 bg-yellow-400 rounded-full animate-ping" />
            <span>משדר...</span>
          </div>
        )}
      </div>
    </div>
  );
}
