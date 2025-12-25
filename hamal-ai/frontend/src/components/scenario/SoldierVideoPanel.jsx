/**
 * SoldierVideoPanel - Split-screen video player with transcription
 *
 * Shows soldier video on the right (in RTL), live transcription on the left.
 * Receives real transcriptions via Socket.IO from available transcribers (Whisper/Gemini).
 * Shows tabs if multiple transcribers are available.
 */

import { useState, useRef, useEffect } from "react";
import { useScenario } from "../../context/ScenarioContext";
import { useApp } from "../../context/AppContext";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:3000";
const AI_SERVICE_URL =
    import.meta.env.VITE_AI_SERVICE_URL || "http://localhost:8000";

export default function SoldierVideoPanel() {
    const { soldierVideo, closeSoldierVideoPanel, config } = useScenario();
    const { socket } = useApp();
    const [transcriptions, setTranscriptions] = useState({
        whisper: [],
        gemini: [],
    });
    const [activeTab, setActiveTab] = useState("whisper");
    const [availableTranscribers, setAvailableTranscribers] = useState([]);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const videoRef = useRef(null);
    const transcriptionRef = useRef(null);

    // Handle missing video path - show placeholder
    const hasVideo =
        soldierVideo?.videoPath && soldierVideo.videoPath.length > 0;
    const videoUrl = hasVideo
        ? soldierVideo.videoPath.startsWith("http")
            ? soldierVideo.videoPath
            : `${API_URL}${soldierVideo.videoPath}`
        : null;

    const handleClose = async () => {
        await closeSoldierVideoPanel();
    };

    // Fetch available transcribers on mount
    useEffect(() => {
        if (!soldierVideo?.open) return;

        const fetchTranscribers = async () => {
            try {
                const response = await fetch(
                    `${AI_SERVICE_URL}/radio/transcribers`
                );
                if (response.ok) {
                    const data = await response.json();
                    const available = data.available_transcribers || [];
                    setAvailableTranscribers(available);
                    if (available.length > 0) {
                        setActiveTab(available[0]);
                    }
                }
            } catch (error) {
                console.warn("Failed to fetch transcriber status:", error);
                setAvailableTranscribers(["whisper", "gemini"]); // Default
            }
        };
        fetchTranscribers();
    }, [soldierVideo?.open]);

    // Listen for transcription events from Socket.IO
    // We listen to BOTH the soldier-video specific event AND the regular radio transcription events
    // This way we get transcriptions immediately as each transcriber finishes
    useEffect(() => {
        if (!socket || !soldierVideo?.open) return;

        // Start transcription indicator when video opens
        setIsTranscribing(true);
        setTranscriptions({ whisper: [], gemini: [] });

        // Handler for adding a single transcription
        const addTranscription = (source, text, timestamp) => {
            console.log(
                `[SoldierVideoPanel] Received ${source} transcription:`,
                text?.substring(0, 50)
            );

            setTranscriptions((prev) => {
                const updated = { ...prev };
                if (!updated[source]) updated[source] = [];
                updated[source] = [
                    ...updated[source],
                    {
                        text: text,
                        source: source,
                        timestamp: timestamp || new Date().toISOString(),
                        time: updated[source].length,
                    },
                ];
                return updated;
            });
            setIsTranscribing(false);

            // Auto-scroll transcription
            setTimeout(() => {
                if (transcriptionRef.current) {
                    transcriptionRef.current.scrollTop =
                        transcriptionRef.current.scrollHeight;
                }
            }, 100);
        };

        // Handler for soldier-video specific event (combined results)
        const handleSoldierVideoTranscription = (data) => {
            console.log(
                "[SoldierVideoPanel] Received soldier-video transcription:",
                data
            );
            if (data.transcriptions && data.transcriptions.length > 0) {
                data.transcriptions.forEach((t) => {
                    addTranscription(
                        t.source || "whisper",
                        t.text,
                        t.timestamp
                    );
                });
            }
        };

        // Handler for Whisper radio transcription (immediate)
        const handleWhisperTranscription = (data) => {
            if (data.text) {
                addTranscription("whisper", data.text, data.timestamp);
            }
        };

        // Handler for Gemini radio transcription (immediate)
        const handleGeminiTranscription = (data) => {
            if (data.text) {
                addTranscription("gemini", data.text, data.timestamp);
            }
        };

        // Listen to all transcription events
        socket.on(
            "scenario:soldier-video-transcription",
            handleSoldierVideoTranscription
        );
        socket.on("radio:transcription:whisper", handleWhisperTranscription);
        socket.on("radio:transcription:gemini", handleGeminiTranscription);

        // Set timeout to stop "transcribing" indicator if no transcription received
        const timeout = setTimeout(() => {
            setIsTranscribing(false);
        }, 60000); // 1 minute timeout

        return () => {
            socket.off(
                "scenario:soldier-video-transcription",
                handleSoldierVideoTranscription
            );
            socket.off(
                "radio:transcription:whisper",
                handleWhisperTranscription
            );
            socket.off("radio:transcription:gemini", handleGeminiTranscription);
            clearTimeout(timeout);
        };
    }, [socket, soldierVideo?.open]);

    // Clear transcription when panel closes
    useEffect(() => {
        if (!soldierVideo?.open) {
            setTranscriptions({ whisper: [], gemini: [] });
            setIsTranscribing(false);
        }
    }, [soldierVideo?.open]);

    if (!soldierVideo || !soldierVideo.open) {
        return null;
    }

    // Get current tab's transcription
    const currentTranscription = transcriptions[activeTab] || [];

    // Check if we have multiple transcribers with content
    const hasMultipleTranscribers = availableTranscribers.length > 1;

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
                    <span>{config?.ui?.closeButton || "סגור"}</span>
                    <span>&times;</span>
                </button>
            </div>

            {/* Main content - split view (flex-row-reverse for RTL: video on right, transcription on left) */}
            <div className="flex-1 flex flex-row-reverse">
                {/* Video player - right side in RTL (60%) */}
                <div className="w-3/5 h-full p-4 flex items-center justify-center bg-black overflow-hidden">
                    {hasVideo ? (
                        <video
                            ref={videoRef}
                            src={videoUrl}
                            controls
                            autoPlay
                            playsInline
                            preload="metadata"
                            className="w-full h-auto max-h-full object-contain bg-black rounded shadow-lg"
                            style={{ maxHeight: "calc(100vh - 140px)" }}
                            onError={(e) => {
                                console.error("Video error:", e);
                            }}
                        >
                            הדפדפן שלך לא תומך בהצגת וידאו
                        </video>
                    ) : (
                        <div className="flex flex-col items-center justify-center text-gray-400">
                            <div className="text-8xl mb-4">📹</div>
                            <p className="text-xl mb-2">ממתין לסרטון מלוחם</p>
                            <p className="text-sm text-gray-500">
                                יש להעלות סרטון דרך האפליקציה או לחכות לקבלת
                                שידור
                            </p>
                            <div className="mt-4 flex items-center gap-2">
                                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                                <span className="text-green-400">
                                    מחכה לסרטון...
                                </span>
                            </div>
                        </div>
                    )}
                </div>

                {/* Transcription - left side in RTL (40%) */}
                <div className="w-2/5 bg-gray-900 border-l border-gray-700 flex flex-col">
                    {/* Transcription header with tabs */}
                    <div className="px-4 py-3 border-b border-gray-700">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-lg font-semibold text-white">
                                תמלול אודיו
                            </h3>
                            {isTranscribing && (
                                <span className="flex items-center gap-2 text-green-400 text-sm">
                                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                                    מתמלל...
                                </span>
                            )}
                        </div>

                        {/* Tabs - only show if multiple transcribers */}
                        {hasMultipleTranscribers && (
                            <div className="flex gap-2">
                                {availableTranscribers.includes("whisper") && (
                                    <button
                                        onClick={() => setActiveTab("whisper")}
                                        className={`px-3 py-1 rounded text-sm transition-colors ${
                                            activeTab === "whisper"
                                                ? "bg-blue-600 text-white"
                                                : "bg-gray-700 text-gray-400 hover:bg-gray-600"
                                        }`}
                                    >
                                        Whisper
                                        {transcriptions.whisper.length > 0 && (
                                            <span className="mr-1 bg-blue-500 text-white text-xs px-1 rounded">
                                                {transcriptions.whisper.length}
                                            </span>
                                        )}
                                    </button>
                                )}
                                {availableTranscribers.includes("gemini") && (
                                    <button
                                        onClick={() => setActiveTab("gemini")}
                                        className={`px-3 py-1 rounded text-sm transition-colors ${
                                            activeTab === "gemini"
                                                ? "bg-purple-600 text-white"
                                                : "bg-gray-700 text-gray-400 hover:bg-gray-600"
                                        }`}
                                    >
                                        Gemini
                                        {transcriptions.gemini.length > 0 && (
                                            <span className="mr-1 bg-purple-500 text-white text-xs px-1 rounded">
                                                {transcriptions.gemini.length}
                                            </span>
                                        )}
                                    </button>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Transcription content */}
                    <div
                        ref={transcriptionRef}
                        className="flex-1 p-4 overflow-y-auto text-right"
                        dir="rtl"
                    >
                        {currentTranscription.length === 0 ? (
                            <div className="text-gray-500 text-center py-8">
                                {isTranscribing
                                    ? "מתמלל..."
                                    : "ממתין לתמלול..."}
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {currentTranscription.map((item, i) => (
                                    <div
                                        key={i}
                                        className={`rounded p-3 animate-fade-in ${
                                            item.source === "whisper"
                                                ? "bg-blue-900/30 border border-blue-700"
                                                : "bg-purple-900/30 border border-purple-700"
                                        }`}
                                    >
                                        <div className="flex items-center gap-2 text-xs text-gray-500 mb-1">
                                            <span>{formatTime(item.time)}</span>
                                            <span
                                                className={`px-1 rounded ${
                                                    item.source === "whisper"
                                                        ? "bg-blue-600"
                                                        : "bg-purple-600"
                                                }`}
                                            >
                                                {item.source === "whisper"
                                                    ? "W"
                                                    : "G"}
                                            </span>
                                        </div>
                                        <div className="text-white">
                                            {item.text}
                                        </div>
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
                                // Download all transcriptions as text
                                let text = "";
                                if (transcriptions.whisper.length > 0) {
                                    text += "=== Whisper ===\n";
                                    text += transcriptions.whisper
                                        .map(
                                            (t) =>
                                                `[${formatTime(t.time)}] ${
                                                    t.text
                                                }`
                                        )
                                        .join("\n");
                                    text += "\n\n";
                                }
                                if (transcriptions.gemini.length > 0) {
                                    text += "=== Gemini ===\n";
                                    text += transcriptions.gemini
                                        .map(
                                            (t) =>
                                                `[${formatTime(t.time)}] ${
                                                    t.text
                                                }`
                                        )
                                        .join("\n");
                                }
                                const blob = new Blob([text], {
                                    type: "text/plain",
                                });
                                const url = URL.createObjectURL(blob);
                                const a = document.createElement("a");
                                a.href = url;
                                a.download = "transcription.txt";
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
    return `${mins}:${secs.toString().padStart(2, "0")}`;
}
