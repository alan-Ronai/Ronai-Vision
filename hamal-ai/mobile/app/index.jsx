import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { Link } from 'expo-router';
import { useEffect, useState } from 'react';
import { io } from 'socket.io-client';

// Server URL - change this to your server IP
const SERVER_URL = process.env.EXPO_PUBLIC_SERVER_URL || 'http://192.168.1.100:3000';

export default function Home() {
  const [connected, setConnected] = useState(false);
  const [emergency, setEmergency] = useState(false);
  const [emergencyData, setEmergencyData] = useState(null);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    console.log('Connecting to:', SERVER_URL);
    const newSocket = io(SERVER_URL, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 5
    });

    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to HAMAL server');
      setConnected(true);
      newSocket.emit('identify', { type: 'soldier-app' });
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
      setConnected(false);
    });

    newSocket.on('connect_error', (error) => {
      console.log('Connection error:', error.message);
      setConnected(false);
    });

    newSocket.on('emergency:start', (data) => {
      console.log('Emergency started:', data);
      setEmergency(true);
      setEmergencyData(data);
    });

    newSocket.on('emergency:end', () => {
      console.log('Emergency ended');
      setEmergency(false);
      setEmergencyData(null);
    });

    return () => newSocket.close();
  }, []);

  return (
    <View style={[styles.container, emergency && styles.emergencyContainer]}>
      {/* Connection Status */}
      <View style={styles.statusContainer}>
        <View style={[styles.statusDot, connected ? styles.statusOnline : styles.statusOffline]} />
        <Text style={styles.statusText}>
          {connected ? 'מחובר לחמ"ל' : 'מנותק מהשרת'}
        </Text>
      </View>

      {/* Server info */}
      <Text style={styles.serverInfo}>שרת: {SERVER_URL}</Text>

      {/* Emergency Banner */}
      {emergency && (
        <View style={styles.emergencyBanner}>
          <Text style={styles.emergencyIcon}>🚨</Text>
          <Text style={styles.emergencyText}>אירוע חירום פעיל</Text>
          <Text style={styles.emergencyIcon}>🚨</Text>
        </View>
      )}

      {/* Emergency Details */}
      {emergencyData && (
        <View style={styles.emergencyDetails}>
          <Text style={styles.emergencyTitle}>{emergencyData.title}</Text>
          {emergencyData.details?.people && (
            <Text style={styles.emergencyInfo}>
              👥 {emergencyData.details.people.count} אנשים
              {emergencyData.details.people.armed && ' - חמושים ⚠️'}
            </Text>
          )}
          {emergencyData.cameraId && (
            <Text style={styles.emergencyInfo}>
              📹 מצלמה: {emergencyData.cameraId}
            </Text>
          )}
        </View>
      )}

      {/* Logo/Title */}
      <View style={styles.logoContainer}>
        <Text style={styles.logo}>🛡️</Text>
        <Text style={styles.title}>אפליקציית לוחם</Text>
        <Text style={styles.subtitle}>חמ"ל AI</Text>
      </View>

      {/* Main Action Button */}
      <Link href="/camera" asChild>
        <TouchableOpacity
          style={[styles.mainButton, emergency && styles.mainButtonEmergency]}
        >
          <Text style={styles.mainButtonIcon}>📹</Text>
          <Text style={styles.mainButtonText}>צלם ושלח לחמ"ל</Text>
          <Text style={styles.mainButtonSubtext}>צילום וידאו מהשטח</Text>
        </TouchableOpacity>
      </Link>

      {/* Quick Actions */}
      <View style={styles.quickActions}>
        <TouchableOpacity
          style={styles.quickButton}
          onPress={() => Alert.alert('התרעה', 'האם לשלוח התרעה לחמ"ל?', [
            { text: 'ביטול', style: 'cancel' },
            {
              text: 'שלח',
              style: 'destructive',
              onPress: () => {
                if (socket) {
                  socket.emit('soldier:alert', { type: 'manual', timestamp: new Date() });
                  Alert.alert('נשלח', 'ההתרעה נשלחה לחמ"ל');
                }
              }
            }
          ])}
        >
          <Text style={styles.quickButtonIcon}>🚨</Text>
          <Text style={styles.quickButtonText}>התרעה</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.quickButton}
          onPress={() => Alert.alert('מיקום', 'שליחת מיקום עדיין לא מיושמת')}
        >
          <Text style={styles.quickButtonIcon}>📍</Text>
          <Text style={styles.quickButtonText}>מיקום</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.quickButton}
          onPress={() => Alert.alert('דיווח', 'דיווח טקסטואלי עדיין לא מיושם')}
        >
          <Text style={styles.quickButtonIcon}>📝</Text>
          <Text style={styles.quickButtonText}>דיווח</Text>
        </TouchableOpacity>
      </View>

      {/* Footer */}
      <Text style={styles.footer}>
        גרסה 1.0.0 | חמ"ל AI
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
    padding: 20,
  },
  emergencyContainer: {
    backgroundColor: '#450a0a',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  statusOnline: {
    backgroundColor: '#22c55e',
  },
  statusOffline: {
    backgroundColor: '#ef4444',
  },
  statusText: {
    color: '#fff',
    fontSize: 16,
  },
  serverInfo: {
    color: '#6b7280',
    fontSize: 12,
    marginBottom: 20,
  },
  emergencyBanner: {
    backgroundColor: '#dc2626',
    padding: 16,
    borderRadius: 12,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 8,
    marginBottom: 16,
  },
  emergencyIcon: {
    fontSize: 24,
  },
  emergencyText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  emergencyDetails: {
    backgroundColor: '#7f1d1d',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  emergencyTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
  },
  emergencyInfo: {
    color: '#fecaca',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 4,
  },
  logoContainer: {
    alignItems: 'center',
    marginVertical: 40,
  },
  logo: {
    fontSize: 64,
    marginBottom: 16,
  },
  title: {
    color: '#fff',
    fontSize: 28,
    fontWeight: 'bold',
  },
  subtitle: {
    color: '#9ca3af',
    fontSize: 16,
    marginTop: 4,
  },
  mainButton: {
    backgroundColor: '#3b82f6',
    padding: 24,
    borderRadius: 20,
    alignItems: 'center',
    marginBottom: 24,
    shadowColor: '#3b82f6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  mainButtonEmergency: {
    backgroundColor: '#dc2626',
    shadowColor: '#dc2626',
  },
  mainButtonIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  mainButtonText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  mainButtonSubtext: {
    color: '#bfdbfe',
    fontSize: 14,
    marginTop: 4,
  },
  quickActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 24,
  },
  quickButton: {
    backgroundColor: '#374151',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    minWidth: 80,
  },
  quickButtonIcon: {
    fontSize: 28,
    marginBottom: 8,
  },
  quickButtonText: {
    color: '#fff',
    fontSize: 14,
  },
  footer: {
    color: '#6b7280',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 'auto',
  },
});
