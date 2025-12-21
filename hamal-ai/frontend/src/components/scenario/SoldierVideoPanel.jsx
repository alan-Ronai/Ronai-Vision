/**
 * SoldierVideoPanel - Split-screen video player with transcription
 *
 * Shows soldier video on the left, live transcription on the right.
 */

import { useState, useRef, useEffect } from 'react';
import { useScenario } from '../../context/ScenarioContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

export default function SoldierVideoPanel() {
  const { soldierVideo, closeSoldierVideoPanel, config } = useScenario();
  const [transcription, setTranscription] = useState([]);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const videoRef = useRef(null);
  const transcriptionRef = useRef(null);

  if (!soldierVideo || !soldierVideo.open) {
    return null;
  }

  // Handle missing video path - show placeholder
  const hasVideo = soldierVideo.videoPath && soldierVideo.videoPath.length > 0;
  const videoUrl = hasVideo
    ? (soldierVideo.videoPath.startsWith('http')
        ? soldierVideo.videoPath
        : `${API_URL}${soldierVideo.videoPath}`)
    : null;

  const handleClose = async () => {
    await closeSoldierVideoPanel();
  };

  // Simulate live transcription (in real implementation, this would use audio extraction + transcription)
  useEffect(() => {
    if (!soldierVideo) return;

    setIsTranscribing(true);

    // Simulate transcription appearing over time
    const mockTranscription = [
      { time: 0, text: 'אני באזור המטרה.' },
      { time: 2, text: 'רואה שלושה חשודים לובשים שחור.' },
      { time: 5, text: 'הם חמושים ברובים.' },
      { time: 8, text: 'ממתין להוראות.' },
      { time: 11, text: 'האם להתקדם?' },
    ];

    const timers = mockTranscription.map((item, index) => {
      return setTimeout(() => {
        setTranscription(prev => [...prev, item]);

        // Auto-scroll transcription
        if (transcriptionRef.current) {
          transcriptionRef.current.scrollTop = transcriptionRef.current.scrollHeight;
        }
      }, item.time * 1000);
    });

    // Mark transcription complete
    const completeTimer = setTimeout(() => {
      setIsTranscribing(false);
    }, 15000);

    return () => {
      timers.forEach(t => clearTimeout(t));
      clearTimeout(completeTimer);
      setTranscription([]);
    };
  }, [soldierVideo]);

  return (
    <div className="fixed inset-0 z-[100] bg-black/95 flex flex-col">
      {/* Header */}
      <div className="bg-gray-900 px-4 py-3 flex items-center justify-between border-b border-gray-700">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span>&#127909;</span>
          סרטון מלוחם בשטח
        </h2>
        <button
          onClick={handleClose}
          className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded flex items-center gap-2"
        >
          <span>{config?.ui?.closeButton || 'סגור'}</span>
          <span>&times;</span>
        </button>
      </div>

      {/* Main content - split view */}
      <div className="flex-1 flex">
        {/* Video player - left side (60%) */}
        <div className="w-3/5 p-4 flex items-center justify-center bg-black">
          {hasVideo ? (
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              autoPlay
              className="max-h-full max-w-full rounded shadow-lg"
              onError={(e) => {
                console.error('Video error:', e);
              }}
            >
              הדפדפן שלך לא תומך בהצגת וידאו
            </video>
          ) : (
            <div className="flex flex-col items-center justify-center text-gray-400">
              <div className="text-8xl mb-4">📹</div>
              <p className="text-xl mb-2">ממתין לסרטון מלוחם</p>
              <p className="text-sm text-gray-500">
                יש להעלות סרטון דרך האפליקציה או לחכות לקבלת שידור
              </p>
              <div className="mt-4 flex items-center gap-2">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                <span className="text-green-400">מחכה לסרטון...</span>
              </div>
            </div>
          )}
        </div>

        {/* Transcription - right side (40%) */}
        <div className="w-2/5 bg-gray-900 border-r border-gray-700 flex flex-col">
          {/* Transcription header */}
          <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">תמלול אודיו</h3>
            {isTranscribing && (
              <span className="flex items-center gap-2 text-green-400 text-sm">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                מתמלל...
              </span>
            )}
          </div>

          {/* Transcription content */}
          <div
            ref={transcriptionRef}
            className="flex-1 p-4 overflow-y-auto text-right"
            dir="rtl"
          >
            {transcription.length === 0 ? (
              <div className="text-gray-500 text-center py-8">
                ממתין לתמלול...
              </div>
            ) : (
              <div className="space-y-3">
                {transcription.map((item, i) => (
                  <div
                    key={i}
                    className="bg-gray-800 rounded p-3 animate-fade-in"
                  >
                    <div className="text-xs text-gray-500 mb-1">
                      {formatTime(item.time)}
                    </div>
                    <div className="text-white">{item.text}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Download button */}
          <div className="px-4 py-3 border-t border-gray-700">
            <button
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded flex items-center justify-center gap-2"
              onClick={() => {
                // Download transcription as text
                const text = transcription.map(t => `[${formatTime(t.time)}] ${t.text}`).join('\n');
                const blob = new Blob([text], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'transcription.txt';
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              <span>&#128190;</span>
              הורד תמלול
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
