import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',  // Black background
    paddingBottom: 20,  // Add some padding for better visibility with keyboard
  },
  chatContainer: {
    flexGrow: 1,
    justifyContent: 'flex-end',
    paddingHorizontal: 10,
    paddingBottom: 80,  // Adjusted to accommodate the input field
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    backgroundColor: '#111',  // Slight contrast to the main background
  },
  input: {
    flex: 1,
    backgroundColor: '#333',
    borderRadius: 20,
    paddingHorizontal: 15,
    color: '#fff',
    height: 40,  // Input height
  },
  sendButton: {
    marginLeft: 10,
    backgroundColor: '#1e90ff',
    borderRadius: 20,
    paddingVertical: 10,
    paddingHorizontal: 20,
  },
  disabledButton: {
    backgroundColor: '#888',
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  speakIconContainer: {
    marginLeft: 10,
    backgroundColor: '#1e90ff',
    borderRadius: 20,
    padding: 10,
  },
  messageContainer: {
    marginVertical: 5,
    maxWidth: '80%',
    borderRadius: 15,
    padding: 10,
  },
  userMessage: {
    backgroundColor: '#1e90ff',
    alignSelf: 'flex-end',
  },
  botMessage: {
    backgroundColor: '#333',
    alignSelf: 'flex-start',
  },
  messageText: {
    color: '#fff',
    fontSize: 16,
  },
});

