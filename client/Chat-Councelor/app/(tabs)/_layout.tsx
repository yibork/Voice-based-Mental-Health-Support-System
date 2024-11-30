// StackLayout.js
import { Stack } from 'expo-router';
import React from 'react';
import { Colors } from '@/constants/Colors'; // Assuming this is for custom styling
import { useColorScheme } from '@/hooks/useColorScheme';

export default function StackLayout() {
  const colorScheme = useColorScheme();

  return (
    <Stack
      screenOptions={{
        headerShown: false, // Hide header if not needed
      }}
    >
      {/* ChatBotScreen Page */}
      <Stack.Screen
        name="index" // Maps to the corresponding route in expo-router
        options={{
          title: 'Chat',
        }}
      />

      {/* CallScreen Page */}
      <Stack.Screen
        name="CallScreen"
        options={{
          title: 'Call',
        }}
      />
    </Stack>
  );
}
