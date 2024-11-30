import { useEffect, useState } from 'react';
import { Alert } from 'react-native';

// Define the WebSocket service as a custom hook
const useWebSocket = (getPollySpeech: (message: string) => void) => {
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket(`ws://${process.env.EXPO_PUBLIC_API_URL}/ws-audio`);

    socket.onopen = () => {
      console.log('WebSocket connection established.');
    };

    socket.onmessage = async (event) => {
      const message = event.data;
      console.log('Received message:', message);
      getPollySpeech(message); // Process the message with Polly
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      Alert.alert('WebSocket Error', 'Could not connect to server.');
    };

    socket.onclose = () => {
      console.log('WebSocket connection closed.');
      Alert.alert('Connection Closed', 'The connection to the server was lost.');
    };

    setWs(socket);

    // Clean up the WebSocket connection on unmount
    return () => {
      socket.close();
    };
  }, [getPollySpeech]);

  return ws; // Return the WebSocket instance if needed
};

export default useWebSocket;
