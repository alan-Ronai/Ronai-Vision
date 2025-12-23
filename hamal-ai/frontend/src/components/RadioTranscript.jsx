import { useApp } from '../context/AppContext';
import { useRef, useEffect, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

export default function RadioTranscript() {
  const { radioTranscript, clearRadioTranscript, isEmergency } = useApp();
  const scrollRef = useRef(null);
  const fileInputRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // Auto-scroll to bottom when new transcription arrives
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft = scrollRef.current.scrollWidth;
    }
  }, [radioTranscript]);

  // Clear error after 5 seconds
  useEffect(() => {
    if (uploadError) {
      const timer = setTimeout(() => setUploadError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [uploadError]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.wav')) {
      setUploadError('יש להעלות קובץ WAV בלבד');
      return;
    }

    // Validate file size (25MB max)
    if (file.size > 25 * 1024 * 1024) {
      setUploadError('הקובץ גדול מדי. גודל מקסימלי: 25MB');
      return;
    }

    setIsUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/api/radio/transcribe-file`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.detail || 'שגיאה בתמלול');
      }

      if (!data.text) {
        setUploadError('לא זוהה דיבור בקובץ');
      }
      // Success - transcription will be added via socket event
    } catch (error) {
      console.error('File transcription error:', error);
      setUploadError(error.message || 'שגיאה בהעלאת הקובץ');
    } finally {
      setIsUploading(false);
      // Clear the file input so the same file can be selected again
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('he-IL', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className={`
      h-full bg-gray-800 rounded-lg overflow-hidden flex flex-col
      ${isEmergency ? 'border-2 border-red-500' : 'border border-gray-700'}
    `}>
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".wav,audio/wav,audio/x-wav"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Header */}
      <div className="bg-gray-700 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-lg">📻</span>
          <span className="font-bold">תמלול קשר</span>
          <div className="flex items-center gap-1 mr-2 px-2 py-0.5 bg-green-600 rounded text-xs">
            <div className="w-2 h-2 bg-green-300 rounded-full animate-pulse"></div>
            <span>חי</span>
          </div>
          {/* Upload file button */}
          <button
            onClick={handleUploadClick}
            disabled={isUploading}
            className={`
              flex items-center gap-1 mr-1 px-2 py-0.5 rounded text-xs font-medium transition-colors
              ${isUploading
                ? 'bg-gray-600 cursor-wait'
                : 'bg-blue-600 hover:bg-blue-700 cursor-pointer'}
            `}
            title="העלה קובץ WAV לתמלול"
          >
            {isUploading ? (
              <>
                <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>מתמלל...</span>
              </>
            ) : (
              <>
                <span>📁</span>
                <span>העלה קובץ</span>
              </>
            )}
          </button>
        </div>
        <div className="flex items-center gap-3">
          {/* Error message */}
          {uploadError && (
            <span className="text-sm text-red-400 bg-red-900/50 px-2 py-0.5 rounded">
              {uploadError}
            </span>
          )}
          <span className="text-sm text-gray-400">{radioTranscript.length} הודעות</span>
          {radioTranscript.length > 0 && (
            <button
              onClick={clearRadioTranscript}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
              title="נקה תמלולים"
            >
              🗑️ נקה
            </button>
          )}
        </div>
      </div>

      {/* Transcript content - horizontal scrolling ticker style */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-x-auto overflow-y-hidden p-3 min-h-0"
      >
        {radioTranscript.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-500">
            <div className="text-center">
              <div className="text-4xl mb-2">📻</div>
              <p>ממתין לשידורים...</p>
            </div>
          </div>
        ) : (
          <div className="flex gap-4 h-full items-center">
            {radioTranscript.map((item, i) => (
              <TranscriptItem
                key={i}
                item={item}
                formatTime={formatTime}
                isLatest={i === radioTranscript.length - 1}
              />
            ))}
          </div>
        )}
      </div>

      {/* Latest transcription highlight */}
      {radioTranscript.length > 0 && (
        <div className="bg-gray-700/50 px-4 py-2 border-t border-gray-600 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-yellow-400">⚡</span>
            <span className="text-sm text-gray-300">אחרון:</span>
            <span className="text-sm font-medium truncate">
              {radioTranscript[radioTranscript.length - 1]?.text}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function TranscriptItem({ item, formatTime, isLatest }) {
  // Check for keywords that might indicate importance
  const isImportant = item.text &&
    (item.text.includes('חדירה') ||
     item.text.includes('חשוד') ||
     item.text.includes('נשק') ||
     item.text.includes('צפרדע') ||
     item.text.includes('רחפן'));

  return (
    <div
      className={`
        flex-shrink-0 min-w-[200px] max-w-[400px] p-3 rounded-lg
        ${isLatest ? 'bg-blue-900/50 border border-blue-500' : 'bg-gray-700'}
        ${isImportant ? 'border-2 border-yellow-500' : ''}
      `}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs text-gray-400">
          {formatTime(item.timestamp)}
        </span>
        {item.source && (
          <span className="text-xs bg-gray-600 px-1 rounded">
            {item.source}
          </span>
        )}
        {isImportant && (
          <span className="text-yellow-400">⚠️</span>
        )}
      </div>
      <p className={`text-sm ${isImportant ? 'text-yellow-300 font-bold' : 'text-gray-200'}`}>
        {item.text}
      </p>
      {item.confidence && (
        <div className="mt-1 text-xs text-gray-500">
          דיוק: {Math.round(item.confidence * 100)}%
        </div>
      )}
    </div>
  );
}
