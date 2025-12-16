import { useState, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, ActivityIndicator } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import { router } from 'expo-router';

// Server URL - change this to your server IP
const SERVER_URL = process.env.EXPO_PUBLIC_SERVER_URL || 'http://192.168.1.100:3000';

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [recording, setRecording] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [facing, setFacing] = useState('back');
  const cameraRef = useRef(null);

  // Permission not determined yet
  if (!permission) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#3b82f6" />
      </View>
    );
  }

  // Permission denied
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionIcon}>📷</Text>
          <Text style={styles.permissionTitle}>נדרשת הרשאה למצלמה</Text>
          <Text style={styles.permissionText}>
            אפליקציה זו זקוקה לגישה למצלמה כדי לצלם ולשלוח סרטונים לחמ"ל
          </Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>אשר הרשאה</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const handleRecord = async () => {
    if (!cameraRef.current) return;

    if (recording) {
      // Stop recording
      cameraRef.current.stopRecording();
      return;
    }

    // Start recording
    setRecording(true);
    try {
      const video = await cameraRef.current.recordAsync({
        maxDuration: 30, // Max 30 seconds
        quality: '720p'
      });

      if (video?.uri) {
        // Upload the video
        await uploadVideo(video.uri);
      }
    } catch (error) {
      console.error('Recording error:', error);
      Alert.alert('שגיאה', 'הצילום נכשל: ' + error.message);
    }
    setRecording(false);
  };

  const uploadVideo = async (uri) => {
    setUploading(true);
    try {
      console.log('Uploading video to:', `${SERVER_URL}/api/uploads/soldier-video`);

      const response = await FileSystem.uploadAsync(
        `${SERVER_URL}/api/uploads/soldier-video`,
        uri,
        {
          fieldName: 'video',
          httpMethod: 'POST',
          uploadType: FileSystem.FileSystemUploadType.MULTIPART,
          parameters: {
            soldierId: 'soldier-1', // TODO: Get from app state/login
            location: 'field', // TODO: Get from GPS
            timestamp: new Date().toISOString()
          }
        }
      );

      console.log('Upload response:', response);

      if (response.status === 200 || response.status === 201) {
        Alert.alert(
          '✅ נשלח בהצלחה',
          'הסרטון נשלח לחמ"ל',
          [{ text: 'אישור', onPress: () => router.back() }]
        );
      } else {
        const errorBody = JSON.parse(response.body || '{}');
        throw new Error(errorBody.error || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      Alert.alert('שגיאת העלאה', error.message);
    }
    setUploading(false);
  };

  const toggleFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

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
            style={styles.flipButton}
            onPress={toggleFacing}
            disabled={recording}
          >
            <Text style={styles.flipButtonText}>🔄</Text>
          </TouchableOpacity>

          {recording && (
            <View style={styles.recordingIndicator}>
              <View style={styles.recordingDot} />
              <Text style={styles.recordingText}>מקליט...</Text>
            </View>
          )}
        </View>

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
              recording && styles.recordingButton
            ]}
            onPress={handleRecord}
            disabled={uploading}
          >
            {recording ? (
              <View style={styles.stopIcon} />
            ) : (
              <View style={styles.recordIcon} />
            )}
          </TouchableOpacity>

          <Text style={styles.hint}>
            {recording ? 'לחץ לעצירה' : 'לחץ לצילום (עד 30 שניות)'}
          </Text>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  topControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: 40,
  },
  flipButton: {
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 12,
    borderRadius: 30,
  },
  flipButtonText: {
    fontSize: 24,
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(220, 38, 38, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    gap: 8,
  },
  recordingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#fff',
  },
  recordingText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  uploadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 16,
  },
  uploadingText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  bottomControls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingBottom: 50,
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#fff',
  },
  recordingButton: {
    backgroundColor: 'rgba(220, 38, 38, 0.5)',
  },
  recordIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#dc2626',
  },
  stopIcon: {
    width: 30,
    height: 30,
    backgroundColor: '#dc2626',
    borderRadius: 4,
  },
  hint: {
    color: '#fff',
    marginTop: 16,
    fontSize: 14,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  permissionIcon: {
    fontSize: 64,
    marginBottom: 24,
  },
  permissionTitle: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 12,
    textAlign: 'center',
  },
  permissionText: {
    color: '#9ca3af',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 24,
  },
  permissionButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  permissionButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
