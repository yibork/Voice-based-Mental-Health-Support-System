import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, Image } from 'react-native';
import { Audio } from 'expo-av';
import Icon from 'react-native-vector-icons/Ionicons';
import { useNavigation } from '@react-navigation/native';
import { AndroidAudioEncoder, AndroidOutputFormat, IOSAudioQuality, IOSOutputFormat } from 'expo-av/build/Audio';
import { Buffer } from 'buffer';
import {getPollySpeech} from '../../services/pollyService';
global.Buffer = Buffer;
import useWebSocket from '../../hooks/useWebSocket';
import { useRouter } from 'expo-router';

//.env load 


export default function ChatBotScreen() {
    global.Buffer = Buffer;
    const router = useRouter();

  const navigation = useNavigation();
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [permissionResponse, requestPermission] = Audio.usePermissions();
  

  const RecordingOptions = {
    isMeteringEnabled: true,
    android: {
      extension: '.aac',
      outputFormat: AndroidOutputFormat.AAC_ADTS,
      audioEncoder: AndroidAudioEncoder.AAC,
      sampleRate: 16000,
      numberOfChannels: 1,
      bitRate: 16000,
    },
    ios: {
      extension: '.wav',
      outputFormat: IOSOutputFormat.LINEARPCM,
      audioQuality: IOSAudioQuality.MEDIUM,
      sampleRate: 16000,
      numberOfChannels: 1,
      bitRate: 16000,
    },
    web: {
      mimeType: 'audio/webm',
      bitsPerSecond: 128000,
    },
  };

  // Establish WebSocket connection
  
  const ws = useWebSocket(getPollySpeech);


  // Function to start recording
  async function startRecording() {
    try {
      if (!permissionResponse || permissionResponse.status !== 'granted') {
        console.log('Requesting permission..');
        await requestPermission();
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      console.log('Starting recording..');
      const { recording } = await Audio.Recording.createAsync(RecordingOptions);
      setRecording(recording);
      console.log('Recording started');
    } catch (err) {
      console.error('Failed to start recording', err);
    }
  }

  // Function to stop recording and send audio via WebSocket
  const stopRecording = async () => {
    console.log('Stopping recording...');
    if (!recording) return;

    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      console.log('Recording stopped and stored at', uri);

      if (!uri) {
        throw new Error('Recording URI is null');
      }
      const response = await fetch(uri);
      console.log('Audio data fetched:', response);
      const blob = await response.blob();

      // Send the audio data to the WebSocket server
      if (ws && ws.readyState === WebSocket.OPEN) {
        const reader = new FileReader();
        reader.onloadend = () => {
          const arrayBuffer = reader.result;
          if (arrayBuffer) {
            ws.send(arrayBuffer);
            console.log('Audio sent to the WebSocket server.');
          } else {
            console.error('Failed to read audio data as ArrayBuffer.');
          }
        };
        reader.readAsArrayBuffer(blob);
      }
      setRecording(null);
    } catch (error) {
      console.error('Error stopping recording:', error);
      Alert.alert('Recording Error', String(error));
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.circleContainer}>
        <View style={styles.circle}>
          <Image
            source={require('@/assets/call-image.jpg')}
            style={styles.image}
          />
        </View>
      </View>
      <TouchableOpacity
        style={styles.micButton}
        onPress={recording ? stopRecording : startRecording}
      >
        <Icon name={recording ? 'mic-off' : 'mic'} size={30} color="#fff" />
      </TouchableOpacity>

      <TouchableOpacity style={styles.cancelButton} onPress={() => router.push('/(tabs)')}>
        <Icon name="close" size={30} color="#fff" />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  circleContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  circle: {
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: '#00f',
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: 180,
    height: 180,
    borderRadius: 90,
  },
  micButton: {
    position: 'absolute',
    bottom: 50,
    left: 30,
    backgroundColor: '#1e90ff',
    borderRadius: 50,
    padding: 20,
  },
  cancelButton: {
    position: 'absolute',
    bottom: 50,
    right: 30,
    backgroundColor: '#ff4500',
    borderRadius: 50,
    padding: 20,
  },
});
