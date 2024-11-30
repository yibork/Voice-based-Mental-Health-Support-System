// pollyService.ts
import { Polly } from 'aws-sdk';
import { Audio } from 'expo-av';
import AWS from 'aws-sdk';


AWS.config.update({
    region: process.env.EXPO_PUBLIC_AWS_REGION, // 
    accessKeyId: process.env.EXPO_PUBLIC_aws_access_key_id, //
    secretAccessKey: process.env.EXPO_PUBLIC_aws_secret_access_key, // 
  });
const polly = new AWS.Polly();

export const getPollySpeech = async (text: string) => {
    const params = {
        OutputFormat: 'mp3',
        Text: text,
        VoiceId: 'Joanna', 
      };

  try {
    console.log('Synthesizing speech...');
    const data = await polly.synthesizeSpeech(params).promise();

    if (!data.AudioStream) {
      throw new Error('Failed to synthesize speech');
    }

    
    // Convert the buffer to a base64 string
    const base64Audio = Buffer.from(data.AudioStream as Buffer).toString('base64');

    // Set the audio mode to ensure playback through the main speaker
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: false, // We are not recording
      playsInSilentModeIOS: true,
      staysActiveInBackground: false,
    });

    // Play the base64 audio using expo-av
    const { sound } = await Audio.Sound.createAsync({
      uri: `data:audio/mpeg;base64,${base64Audio}`,
    });

    // Play the sound and await for it to finish
    await sound.playAsync();

    // Set the status update to monitor the sound and unload it only when playback finishes
    sound.setOnPlaybackStatusUpdate((status) => {
      if (status.isLoaded && status.didJustFinish) {
        sound.unloadAsync();
      }
    });

  } catch (error) {
    console.error('Error synthesizing speech:', error);
  }
};
