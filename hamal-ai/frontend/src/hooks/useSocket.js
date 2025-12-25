/**
 * Custom hook for Socket.IO connection management
 * Can be used as an alternative to the context-based approach
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { io } from 'socket.io-client';

// Use relative URL to leverage Vite proxy (avoids mixed content issues with HTTPS)
const SOCKET_URL = '';

export function useSocket(options = {}) {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const socketRef = useRef(null);
  const listenersRef = useRef(new Map());

  useEffect(() => {
    // Create socket connection
    const socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: options.reconnectionAttempts || 5,
      reconnectionDelay: options.reconnectionDelay || 1000,
      ...options.socketOptions
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('Socket connected');
      setConnected(true);
      setError(null);

      // Identify client
      if (options.clientType) {
        socket.emit('identify', { type: options.clientType });
      }
    });

    socket.on('disconnect', (reason) => {
      console.log('Socket disconnected:', reason);
      setConnected(false);
    });

    socket.on('connect_error', (err) => {
      console.error('Socket connection error:', err);
      setError(err.message);
      setConnected(false);
    });

    // Re-attach any listeners that were added before connection
    for (const [event, callback] of listenersRef.current) {
      socket.on(event, callback);
    }

    return () => {
      socket.close();
      socketRef.current = null;
    };
  }, [options.clientType]);

  /**
   * Subscribe to a socket event
   */
  const on = useCallback((event, callback) => {
    listenersRef.current.set(event, callback);

    if (socketRef.current) {
      socketRef.current.on(event, callback);
    }

    // Return unsubscribe function
    return () => {
      listenersRef.current.delete(event);
      if (socketRef.current) {
        socketRef.current.off(event, callback);
      }
    };
  }, []);

  /**
   * Subscribe to event once
   */
  const once = useCallback((event, callback) => {
    if (socketRef.current) {
      socketRef.current.once(event, callback);
    }
  }, []);

  /**
   * Emit an event
   */
  const emit = useCallback((event, data) => {
    if (socketRef.current && connected) {
      socketRef.current.emit(event, data);
      return true;
    }
    return false;
  }, [connected]);

  /**
   * Get socket instance (for advanced usage)
   */
  const getSocket = useCallback(() => socketRef.current, []);

  return {
    connected,
    error,
    on,
    once,
    emit,
    getSocket
  };
}

export default useSocket;
