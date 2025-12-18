import { useApp } from '../context/AppContext';
import { useRef, useEffect } from 'react';

export default function RadioTranscript() {
  const { radioTranscript, clearRadioTranscript, isEmergency } = useApp();
  const scrollRef = useRef(null);

  // Auto-scroll to bottom when new transcription arrives
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft = scrollRef.current.scrollWidth;
    }
  }, [radioTranscript]);

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
      {/* Header */}
      <div className="bg-gray-700 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-lg">📻</span>
          <span className="font-bold">תמלול קשר</span>
          <div className="flex items-center gap-1 mr-2 px-2 py-0.5 bg-green-600 rounded text-xs">
            <div className="w-2 h-2 bg-green-300 rounded-full animate-pulse"></div>
            <span>חי</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
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
