import { useState, useRef, useEffect } from "react";
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Alert,
    ActivityIndicator,
    Dimensions,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImagePicker from "expo-image-picker";
import * as FileSystem from "expo-file-system/legacy";
import { Video, ResizeMode } from "expo-av";
import { router } from "expo-router";

// Server URL - change this to your server IP
const SERVER_URL =
    process.env.EXPO_PUBLIC_SERVER_URL || "http://192.168.1.100:3000";

export default function CameraScreen() {
    const [permission, requestPermission] = useCameraPermissions();
    const [mode, setMode] = useState("select"); // 'select', 'camera', 'preview'
    const [recording, setRecording] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [facing, setFacing] = useState("back");
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [recordingDuration, setRecordingDuration] = useState(0);
    const cameraRef = useRef(null);
    const recordingTimer = useRef(null);

    // Clean up timer on unmount
    useEffect(() => {
        return () => {
            if (recordingTimer.current) {
                clearInterval(recordingTimer.current);
            }
        };
    }, []);

    // Upload video to server
    const uploadVideo = async (uri) => {
        setUploading(true);
        setUploadProgress(0);

        try {
            console.log(
                "Uploading video to:",
                `${SERVER_URL}/api/scenario/soldier-video`
            );

            // Ensure the URI is a local file path with file:// prefix
            const fileUri = uri.startsWith("file://") ? uri : `file://${uri}`;

            const fileName = fileUri.split("/").pop() || "video.mov";
            const mimeType = fileName.toLowerCase().endsWith(".mov")
                ? "video/quicktime"
                : "video/mp4";

            const response = await FileSystem.uploadAsync(
                `${SERVER_URL}/api/scenario/soldier-video`,
                fileUri,
                {
                    fieldName: "video",
                    httpMethod: "POST",
                    uploadType: FileSystem.FileSystemUploadType.MULTIPART,
                    mimeType,
                    parameters: {
                        soldierId: "soldier-1",
                        location: "field",
                        timestamp: new Date().toISOString(),
                        filename: fileName,
                    },
                }
            );

            console.log("Upload response:", response);

            if (response.status === 200 || response.status === 201) {
                Alert.alert("נשלח בהצלחה", 'הסרטון נשלח לחמ"ל', [
                    { text: "אישור", onPress: () => router.back() },
                ]);
            } else {
                const errorBody = JSON.parse(response.body || "{}");
                throw new Error(errorBody.error || "Upload failed");
            }
        } catch (error) {
            console.error("Upload error:", error);
            Alert.alert("שגיאת העלאה", error.message);
        }
        setUploading(false);
        setUploadProgress(0);
    };

    // Pick video from gallery
    const pickVideo = async () => {
        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Videos,
                allowsEditing: true,
                quality: 1,
                videoMaxDuration: 60,
            });

            if (!result.canceled && result.assets[0]) {
                setSelectedVideo(result.assets[0]);
                setMode("preview");
            }
        } catch (error) {
            console.error("Error picking video:", error);
            Alert.alert("שגיאה", "לא ניתן לבחור סרטון");
        }
    };

    // Start camera recording
    const startRecording = async () => {
        if (!cameraRef.current) return;

        setRecording(true);
        setRecordingDuration(0);

        // Start duration timer
        recordingTimer.current = setInterval(() => {
            setRecordingDuration((prev) => {
                if (prev >= 29) {
                    // Auto-stop at 30 seconds
                    stopRecording();
                    return prev;
                }
                return prev + 1;
            });
        }, 1000);

        try {
            const video = await cameraRef.current.recordAsync({
                maxDuration: 30,
                quality: "720p",
            });

            if (video?.uri) {
                setSelectedVideo({ uri: video.uri });
                setMode("preview");
            }
        } catch (error) {
            console.error("Recording error:", error);
            Alert.alert("שגיאה", "הצילום נכשל: " + error.message);
        }

        if (recordingTimer.current) {
            clearInterval(recordingTimer.current);
        }
        setRecording(false);
    };

    // Stop camera recording
    const stopRecording = () => {
        if (cameraRef.current && recording) {
            cameraRef.current.stopRecording();
        }
        if (recordingTimer.current) {
            clearInterval(recordingTimer.current);
        }
    };

    // Toggle camera facing
    const toggleFacing = () => {
        setFacing((current) => (current === "back" ? "front" : "back"));
    };

    // Send the selected/recorded video
    const sendVideo = () => {
        if (selectedVideo?.uri) {
            uploadVideo(selectedVideo.uri);
        }
    };

    // Cancel and go back to selection
    const cancelPreview = () => {
        setSelectedVideo(null);
        setMode("select");
    };

    // Go to camera mode
    const goToCamera = () => {
        if (!permission?.granted) {
            requestPermission();
            return;
        }
        setMode("camera");
    };

    // Format duration as MM:SS
    const formatDuration = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    // ========== RENDER MODES ==========

    // Selection mode - choose between camera or file upload
    if (mode === "select") {
        return (
            <View style={styles.container}>
                <View style={styles.selectContainer}>
                    <Text style={styles.selectTitle}>שלח סרטון לחמ"ל</Text>
                    <Text style={styles.selectSubtitle}>
                        בחר את אופן השליחה
                    </Text>

                    {/* Record Video Button */}
                    <TouchableOpacity
                        style={styles.selectButton}
                        onPress={goToCamera}
                    >
                        <View style={styles.selectButtonIcon}>
                            <Text style={styles.selectButtonEmoji}>📹</Text>
                        </View>
                        <View style={styles.selectButtonText}>
                            <Text style={styles.selectButtonTitle}>
                                צלם עכשיו
                            </Text>
                            <Text style={styles.selectButtonDesc}>
                                הקלטה חדשה מהמצלמה
                            </Text>
                        </View>
                    </TouchableOpacity>

                    {/* Upload Video Button */}
                    <TouchableOpacity
                        style={styles.selectButton}
                        onPress={pickVideo}
                    >
                        <View style={styles.selectButtonIcon}>
                            <Text style={styles.selectButtonEmoji}>📁</Text>
                        </View>
                        <View style={styles.selectButtonText}>
                            <Text style={styles.selectButtonTitle}>
                                בחר מהגלריה
                            </Text>
                            <Text style={styles.selectButtonDesc}>
                                העלה סרטון קיים
                            </Text>
                        </View>
                    </TouchableOpacity>

                    {/* Cancel Button */}
                    <TouchableOpacity
                        style={styles.cancelButton}
                        onPress={() => router.back()}
                    >
                        <Text style={styles.cancelButtonText}>ביטול</Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    }

    // Preview mode - show selected/recorded video before sending
    if (mode === "preview") {
        return (
            <View style={styles.container}>
                {/* Video Preview */}
                <View style={styles.previewContainer}>
                    {selectedVideo?.uri && (
                        <Video
                            source={{ uri: selectedVideo.uri }}
                            style={styles.videoPreview}
                            useNativeControls
                            resizeMode="contain"
                            shouldPlay
                            isLooping
                        />
                    )}
                </View>

                {/* Upload Overlay */}
                {uploading && (
                    <View style={styles.uploadingOverlay}>
                        <ActivityIndicator size="large" color="#fff" />
                        <Text style={styles.uploadingText}>מעלה לחמ"ל...</Text>
                    </View>
                )}

                {/* Action Buttons */}
                {!uploading && (
                    <View style={styles.previewActions}>
                        <TouchableOpacity
                            style={styles.previewCancelButton}
                            onPress={cancelPreview}
                        >
                            <Text style={styles.previewCancelIcon}>✕</Text>
                            <Text style={styles.previewButtonText}>
                                בחר אחר
                            </Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.previewSendButton}
                            onPress={sendVideo}
                        >
                            <Text style={styles.previewSendIcon}>✓</Text>
                            <Text style={styles.previewButtonText}>
                                שלח לחמ"ל
                            </Text>
                        </TouchableOpacity>
                    </View>
                )}
            </View>
        );
    }

    // Camera mode - live recording
    // Check permissions first
    if (!permission) {
        return (
            <View style={styles.container}>
                <ActivityIndicator size="large" color="#3b82f6" />
            </View>
        );
    }

    if (!permission.granted) {
        return (
            <View style={styles.container}>
                <View style={styles.permissionContainer}>
                    <Text style={styles.permissionIcon}>📷</Text>
                    <Text style={styles.permissionTitle}>
                        נדרשת הרשאה למצלמה
                    </Text>
                    <Text style={styles.permissionText}>
                        אפליקציה זו זקוקה לגישה למצלמה כדי לצלם ולשלוח סרטונים
                        לחמ"ל
                    </Text>
                    <TouchableOpacity
                        style={styles.permissionButton}
                        onPress={requestPermission}
                    >
                        <Text style={styles.permissionButtonText}>
                            אשר הרשאה
                        </Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={styles.backButton}
                        onPress={() => setMode("select")}
                    >
                        <Text style={styles.backButtonText}>חזרה</Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <CameraView
                ref={cameraRef}
                style={styles.camera}
                facing={facing}
                mode="video"
            >
                {/* Top controls */}
                <View style={styles.topControls}>
                    <TouchableOpacity
                        style={styles.backToSelectButton}
                        onPress={() => setMode("select")}
                        disabled={recording}
                    >
                        <Text style={styles.backToSelectText}>← חזרה</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={styles.flipButton}
                        onPress={toggleFacing}
                        disabled={recording}
                    >
                        <Text style={styles.flipButtonText}>🔄</Text>
                    </TouchableOpacity>
                </View>

                {/* Recording indicator */}
                {recording && (
                    <View style={styles.recordingBanner}>
                        <View style={styles.recordingDot} />
                        <Text style={styles.recordingText}>
                            מקליט {formatDuration(recordingDuration)} / 0:30
                        </Text>
                    </View>
                )}

                {/* Upload overlay */}
                {uploading && (
                    <View style={styles.uploadingOverlay}>
                        <ActivityIndicator size="large" color="#fff" />
                        <Text style={styles.uploadingText}>מעלה לחמ"ל...</Text>
                    </View>
                )}

                {/* Bottom controls */}
                <View style={styles.bottomControls}>
                    <TouchableOpacity
                        style={[
                            styles.recordButton,
                            recording && styles.recordingButton,
                        ]}
                        onPress={recording ? stopRecording : startRecording}
                        disabled={uploading}
                    >
                        {recording ? (
                            <View style={styles.stopIcon} />
                        ) : (
                            <View style={styles.recordIcon} />
                        )}
                    </TouchableOpacity>

                    <Text style={styles.hint}>
                        {recording ? "לחץ לעצירה" : "לחץ להקלטה (עד 30 שניות)"}
                    </Text>
                </View>
            </CameraView>
        </View>
    );
}

const { width, height } = Dimensions.get("window");

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: "#111827",
    },

    // ===== SELECT MODE STYLES =====
    selectContainer: {
        flex: 1,
        padding: 24,
        justifyContent: "center",
    },
    selectTitle: {
        color: "#fff",
        fontSize: 28,
        fontWeight: "bold",
        textAlign: "center",
        marginBottom: 8,
    },
    selectSubtitle: {
        color: "#9ca3af",
        fontSize: 16,
        textAlign: "center",
        marginBottom: 40,
    },
    selectButton: {
        backgroundColor: "#1f2937",
        borderRadius: 16,
        padding: 20,
        flexDirection: "row",
        alignItems: "center",
        marginBottom: 16,
        borderWidth: 2,
        borderColor: "#374151",
    },
    selectButtonIcon: {
        width: 60,
        height: 60,
        borderRadius: 30,
        backgroundColor: "#3b82f6",
        justifyContent: "center",
        alignItems: "center",
        marginRight: 16,
    },
    selectButtonEmoji: {
        fontSize: 28,
    },
    selectButtonText: {
        flex: 1,
    },
    selectButtonTitle: {
        color: "#fff",
        fontSize: 20,
        fontWeight: "bold",
        marginBottom: 4,
    },
    selectButtonDesc: {
        color: "#9ca3af",
        fontSize: 14,
    },
    cancelButton: {
        marginTop: 24,
        padding: 16,
        alignItems: "center",
    },
    cancelButtonText: {
        color: "#9ca3af",
        fontSize: 16,
    },

    // ===== PREVIEW MODE STYLES =====
    previewContainer: {
        flex: 1,
        backgroundColor: "#000",
        alignItems: "center",
        justifyContent: "center",
    },
    videoPreview: {
        width: "100%",
        maxHeight: height * 0.8,
        aspectRatio: 9 / 16,
        backgroundColor: "#000",
    },
    previewActions: {
        flexDirection: "row",
        padding: 20,
        backgroundColor: "#1f2937",
        gap: 16,
    },
    previewCancelButton: {
        flex: 1,
        backgroundColor: "#374151",
        borderRadius: 16,
        padding: 20,
        alignItems: "center",
    },
    previewSendButton: {
        flex: 1,
        backgroundColor: "#22c55e",
        borderRadius: 16,
        padding: 20,
        alignItems: "center",
    },
    previewCancelIcon: {
        fontSize: 32,
        color: "#fff",
        marginBottom: 8,
    },
    previewSendIcon: {
        fontSize: 32,
        color: "#fff",
        marginBottom: 8,
    },
    previewButtonText: {
        color: "#fff",
        fontSize: 16,
        fontWeight: "bold",
    },

    // ===== CAMERA MODE STYLES =====
    camera: {
        flex: 1,
    },
    topControls: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        padding: 20,
        paddingTop: 50,
    },
    backToSelectButton: {
        backgroundColor: "rgba(0,0,0,0.5)",
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 20,
    },
    backToSelectText: {
        color: "#fff",
        fontSize: 16,
        fontWeight: "bold",
    },
    flipButton: {
        backgroundColor: "rgba(0,0,0,0.5)",
        padding: 12,
        borderRadius: 30,
    },
    flipButtonText: {
        fontSize: 24,
    },
    recordingBanner: {
        position: "absolute",
        top: 110,
        left: 20,
        right: 20,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "rgba(220, 38, 38, 0.9)",
        paddingHorizontal: 20,
        paddingVertical: 12,
        borderRadius: 30,
        gap: 10,
    },
    recordingDot: {
        width: 12,
        height: 12,
        borderRadius: 6,
        backgroundColor: "#fff",
    },
    recordingText: {
        color: "#fff",
        fontWeight: "bold",
        fontSize: 18,
    },
    uploadingOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: "rgba(0,0,0,0.8)",
        justifyContent: "center",
        alignItems: "center",
        gap: 20,
    },
    uploadingText: {
        color: "#fff",
        fontSize: 20,
        fontWeight: "bold",
    },
    bottomControls: {
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        alignItems: "center",
        paddingBottom: 60,
    },
    recordButton: {
        width: 90,
        height: 90,
        borderRadius: 45,
        backgroundColor: "rgba(255,255,255,0.3)",
        justifyContent: "center",
        alignItems: "center",
        borderWidth: 5,
        borderColor: "#fff",
    },
    recordingButton: {
        backgroundColor: "rgba(220, 38, 38, 0.5)",
    },
    recordIcon: {
        width: 70,
        height: 70,
        borderRadius: 35,
        backgroundColor: "#dc2626",
    },
    stopIcon: {
        width: 35,
        height: 35,
        backgroundColor: "#dc2626",
        borderRadius: 6,
    },
    hint: {
        color: "#fff",
        marginTop: 20,
        fontSize: 16,
        fontWeight: "500",
    },

    // ===== PERMISSION STYLES =====
    permissionContainer: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        padding: 40,
    },
    permissionIcon: {
        fontSize: 64,
        marginBottom: 24,
    },
    permissionTitle: {
        color: "#fff",
        fontSize: 24,
        fontWeight: "bold",
        marginBottom: 12,
        textAlign: "center",
    },
    permissionText: {
        color: "#9ca3af",
        fontSize: 16,
        textAlign: "center",
        marginBottom: 32,
        lineHeight: 24,
    },
    permissionButton: {
        backgroundColor: "#3b82f6",
        paddingHorizontal: 32,
        paddingVertical: 16,
        borderRadius: 12,
        marginBottom: 16,
    },
    permissionButtonText: {
        color: "#fff",
        fontSize: 18,
        fontWeight: "bold",
    },
    backButton: {
        padding: 16,
    },
    backButtonText: {
        color: "#9ca3af",
        fontSize: 16,
    },
});
