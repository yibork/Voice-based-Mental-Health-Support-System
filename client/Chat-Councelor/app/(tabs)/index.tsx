import React, { useState, useRef, useEffect } from 'react';
import { View, TextInput, FlatList, Text, KeyboardAvoidingView, Platform, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { ThemedView } from '@/components/ThemedView';
import Icon from 'react-native-vector-icons/Ionicons';
import { styles } from '../../styles/index';
import { useRouter } from 'expo-router';

export default function ChatBotScreen() {
  interface Message {
    id: string;
    text: string;
    sender: 'user' | 'bot';
  }

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const flatListRef = useRef<FlatList<Message>>(null);
  const navigation = useNavigation(); 
  const [ws, setWs] = useState<WebSocket | null>(null); // WebSocket state
  const router = useRouter();

  useEffect(() => {
    // Initialize WebSocket connection
    console.log('WebSocket connecting... to ', process.env.EXPO_PUBLIC_API_URL);
    const websocket = new WebSocket(`ws://${process.env.EXPO_PUBLIC_API_URL}/ws`);
    console.log('WebSocket connecting...');

    websocket.onopen = () => {
      console.log('WebSocket connected');
    };

    websocket.onmessage = (e) => {
      const response = e.data; // Get the response from the server
      
      // Add bot response to messages
      setMessages((prevMessages) => [
        ...prevMessages,
        { id: Math.random().toString(), text: response, sender: 'bot' },
      ]);
      // Scroll to the end of the list after a new message
      flatListRef.current?.scrollToEnd({ animated: true });
    };

    websocket.onerror = (e) => {
      console.error('WebSocket error:', e);
    };

    websocket.onclose = () => {
      console.log('WebSocket closed');
    };

    setWs(websocket); // Store WebSocket in state

    return () => {
      websocket.close(); // Cleanup on unmount
    };
  }, []);

  const handleSendMessage = () => {
    if (inputText.trim() && ws) {
      const userMessage = inputText; // Store user message
      setMessages((prevMessages) => [
        ...prevMessages,
        { id: Math.random().toString(), text: userMessage, sender: 'user' },
      ]);
      setInputText('');

      // Send user message to the WebSocket server
      ws.send(userMessage);

      // Scroll to the end of the list after a new message
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  };

  const renderMessage = ({ item }: { item: Message }) => {
    const isUserMessage = item.sender === 'user';
    return (
      <View
        style={[
          styles.messageContainer,
          isUserMessage ? styles.userMessage : styles.botMessage,
        ]}
      >
        <Text style={styles.messageText}>{item.text}</Text>
      </View>
    );
  };

  const handleSpeakButtonPress = () => {
    router.push('/(tabs)/ChatBotScreen');
  };

  return (
    <ThemedView style={styles.container}>
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.chatContainer}
        onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
      />
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={0}
      >
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Type a message"
            placeholderTextColor="#aaa"
          />
          <TouchableOpacity
            style={[styles.sendButton, inputText.trim() === '' ? styles.disabledButton : null]} 
            onPress={handleSendMessage}
            disabled={inputText.trim() === ''}
          >
            <Text style={styles.sendButtonText}>Send</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.speakIconContainer}
            onPress={handleSpeakButtonPress}
          >
            <Icon name="mic" size={24} color="#fff" />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </ThemedView>
  );
}