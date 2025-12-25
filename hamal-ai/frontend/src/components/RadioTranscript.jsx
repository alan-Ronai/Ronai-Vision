import { useApp } from "../context/AppContext";
import { useRef, useEffect, useState } from "react";

// Use relative URLs to leverage Vite proxy (avoids mixed content issues with HTTPS)
const API_URL = '';
const AI_SERVICE_PROXY = '';  // /radio endpoint is proxied to AI service

export default function RadioTranscript() {
    const {
        isEmergency,
        whisperTranscript,
        clearWhisperTranscript,
        geminiTranscript,
        clearGeminiTranscript,
    } = useApp();

    const scrollRef = useRef(null);
    const fileInputRef = useRef(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const [activeTab, setActiveTab] = useState(null); // Will be set based on available transcribers
    const [availableTranscribers, setAvailableTranscribers] = useState([
        "whisper",
        "gemini",
    ]); // Default to both
    const [transcribersLoading, setTranscribersLoading] = useState(true);

    // Fetch available transcribers on mount
    useEffect(() => {
        const fetchTranscribers = async () => {
            try {
                const response = await fetch(
                    `${AI_SERVICE_PROXY}/radio/transcribers`
                );
                if (response.ok) {
                    const data = await response.json();
                    const available = data.available_transcribers || [];
                    setAvailableTranscribers(available);
                    // Set default tab to first available transcriber
                    if (available.length > 0 && !activeTab) {
                        setActiveTab(available[0]);
                    }
                }
            } catch (error) {
                console.warn("Failed to fetch transcriber status:", error);
                // Keep defaults if fetch fails
            } finally {
                setTranscribersLoading(false);
            }
        };
        fetchTranscribers();
    }, []);

    // Set default tab once transcribers are loaded
    useEffect(() => {
        if (
            !transcribersLoading &&
            !activeTab &&
            availableTranscribers.length > 0
        ) {
            setActiveTab(availableTranscribers[0]);
        }
    }, [transcribersLoading, availableTranscribers, activeTab]);

    // Get current transcript based on active tab
    const currentTranscript =
        activeTab === "whisper" ? whisperTranscript : geminiTranscript;
    const clearCurrentTranscript =
        activeTab === "whisper"
            ? clearWhisperTranscript
            : clearGeminiTranscript;

    // Auto-scroll to right when new transcription arrives
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollLeft = scrollRef.current.scrollWidth;
        }
    }, [currentTranscript]);

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
        if (!file.name.toLowerCase().endsWith(".wav")) {
            setUploadError("יש להעלות קובץ WAV בלבד");
            return;
        }

        // Validate file size (25MB max)
        if (file.size > 25 * 1024 * 1024) {
            setUploadError("הקובץ גדול מדי. גודל מקסימלי: 25MB");
            return;
        }

        setIsUploading(true);
        setUploadError(null);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch(
                `${API_URL}/api/radio/transcribe-file`,
                {
                    method: "POST",
                    body: formData,
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || data.detail || "שגיאה בתמלול");
            }

            // Check if at least one transcriber returned results
            const hasWhisper = data.whisper && data.whisper.text;
            const hasGemini = data.gemini && data.gemini.text;

            if (!hasWhisper && !hasGemini) {
                setUploadError("לא זוהה דיבור בקובץ");
            }
            // Success - transcriptions will be added via socket events
        } catch (error) {
            console.error("File transcription error:", error);
            setUploadError(error.message || "שגיאה בהעלאת הקובץ");
        } finally {
            setIsUploading(false);
            // Clear the file input so the same file can be selected again
            if (fileInputRef.current) {
                fileInputRef.current.value = "";
            }
        }
    };

    const formatTime = (timestamp) => {
        return new Date(timestamp).toLocaleTimeString("he-IL", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
    };

    return (
        <div
            className={`
      h-full bg-gray-800 rounded-lg overflow-hidden flex flex-col
      ${isEmergency ? "border-2 border-red-500" : "border border-gray-700"}
    `}
        >
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
              ${
                  isUploading
                      ? "bg-gray-600 cursor-wait"
                      : "bg-blue-600 hover:bg-blue-700 cursor-pointer"
              }
            `}
                        title="העלה קובץ WAV לתמלול (שני המתמללים)"
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
                    <span className="text-sm text-gray-400">
                        {currentTranscript.length} הודעות
                    </span>
                    {currentTranscript.length > 0 && (
                        <button
                            onClick={clearCurrentTranscript}
                            className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
                            title="נקה תמלולים של הטאב הנוכחי"
                        >
                            🗑️ נקה
                        </button>
                    )}
                </div>
            </div>

            {/* Main content area with vertical tabs */}
            <div className="flex-1 flex min-h-0">
                {/* Vertical tabs on the left (appears on right in RTL) - only show if multiple transcribers */}
                {availableTranscribers.length > 1 && (
                    <div className="flex flex-col border-l border-gray-600 bg-gray-750">
                        {availableTranscribers.includes("whisper") && (
                            <button
                                onClick={() => setActiveTab("whisper")}
                                className={`
              flex-1 px-2 text-xs font-medium transition-colors flex items-center justify-center
              ${
                  activeTab === "whisper"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-gray-200"
              }
            `}
                                style={{
                                    writingMode: "vertical-rl",
                                    textOrientation: "mixed",
                                }}
                                title="מתמלל 1 (Whisper)"
                            >
                                <div className="flex items-center gap-1">
                                    <span>מתמלל 1</span>
                                    {whisperTranscript.length > 0 && (
                                        <span className="bg-blue-500 text-white text-xs px-1 rounded">
                                            {whisperTranscript.length}
                                        </span>
                                    )}
                                </div>
                            </button>
                        )}
                        {availableTranscribers.includes("gemini") && (
                            <button
                                onClick={() => setActiveTab("gemini")}
                                className={`
              flex-1 px-2 text-xs font-medium transition-colors flex items-center justify-center
              ${
                  activeTab === "gemini"
                      ? "bg-purple-600 text-white"
                      : "bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-gray-200"
              }
            `}
                                style={{
                                    writingMode: "vertical-rl",
                                    textOrientation: "mixed",
                                }}
                                title="מתמלל 2 (Gemini)"
                            >
                                <div className="flex items-center gap-1">
                                    <span>מתמלל 2</span>
                                    {geminiTranscript.length > 0 && (
                                        <span className="bg-purple-500 text-white text-xs px-1 rounded">
                                            {geminiTranscript.length}
                                        </span>
                                    )}
                                </div>
                            </button>
                        )}
                    </div>
                )}

                {/* Transcript content */}
                <div
                    ref={scrollRef}
                    className="flex-1 overflow-x-auto overflow-y-hidden p-3"
                >
                    {availableTranscribers.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-gray-500">
                            <div className="text-center">
                                <div className="text-4xl mb-2">🔇</div>
                                <p>אין מתמללים זמינים</p>
                                <p className="text-xs mt-1 text-gray-600">
                                    כל המתמללים מושבתים
                                </p>
                            </div>
                        </div>
                    ) : currentTranscript.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-gray-500">
                            <div className="text-center">
                                <div className="text-4xl mb-2">📻</div>
                                <p>ממתין לשידורים...</p>
                                <p className="text-xs mt-1 text-gray-600">
                                    {availableTranscribers.length === 1
                                        ? activeTab === "whisper"
                                            ? "מתמלל"
                                            : "מתמלל"
                                        : activeTab === "whisper"
                                        ? "מתמלל 1"
                                        : "מתמלל 2"}
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="flex gap-4 h-full items-center">
                            {currentTranscript.map((item, i) => (
                                <TranscriptItem
                                    key={i}
                                    item={item}
                                    formatTime={formatTime}
                                    isLatest={
                                        i === currentTranscript.length - 1
                                    }
                                />
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Latest transcription highlight */}
            {currentTranscript.length > 0 &&
                availableTranscribers.length > 0 && (
                    <div className="bg-gray-700/50 px-4 py-2 border-t border-gray-600 flex-shrink-0">
                        <div className="flex items-center gap-2">
                            <span className="text-yellow-400">⚡</span>
                            <span className="text-sm text-gray-300">
                                אחרון:
                            </span>
                            <span className="text-sm font-medium truncate">
                                {
                                    currentTranscript[
                                        currentTranscript.length - 1
                                    ]?.text
                                }
                            </span>
                            {/* Only show transcriber badge if multiple transcribers available */}
                            {availableTranscribers.length > 1 && (
                                <span
                                    className={`text-xs px-1 rounded ${
                                        activeTab === "whisper"
                                            ? "bg-blue-600"
                                            : "bg-purple-600"
                                    }`}
                                >
                                    {activeTab === "whisper"
                                        ? "מתמלל 1"
                                        : "מתמלל 2"}
                                </span>
                            )}
                        </div>
                    </div>
                )}
        </div>
    );
}

function TranscriptItem({ item, formatTime, isLatest }) {
    // Check for keywords that might indicate importance
    const isImportant =
        item.text &&
        (item.text.includes("חדירה") ||
            item.text.includes("חשוד") ||
            item.text.includes("נשק") ||
            item.text.includes("צפרדע") ||
            item.text.includes("רחפן"));

    // Determine transcriber from item data
    const transcriber =
        item.transcriber ||
        (item.source?.includes("whisper") ? "whisper" : "gemini");

    return (
        <div
            className={`
        flex-shrink-0 min-w-[200px] max-w-[400px] p-3 rounded-lg
        ${isLatest ? "bg-blue-900/50 border border-blue-500" : "bg-gray-700"}
        ${isImportant ? "border-2 border-yellow-500" : ""}
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
                {/* Show transcriber badge */}
                <span
                    className={`text-xs px-1 rounded ${
                        transcriber === "whisper"
                            ? "bg-blue-600"
                            : "bg-purple-600"
                    }`}
                >
                    {transcriber === "whisper" ? "W" : "G"}
                </span>
                {isImportant && <span className="text-yellow-400">⚠️</span>}
            </div>
            <p
                className={`text-sm ${
                    isImportant ? "text-yellow-300 font-bold" : "text-gray-200"
                }`}
            >
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
