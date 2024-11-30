import logging
import speech_recognition as sr
from speak import speak


recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True
def listen():
    with sr.Microphone() as source:
        logging.debug("Calibrating for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Calibrate
        
        logging.debug("Listening...")
        recognizer.pause_threshold = 0.8  # Reduced threshold
        
        try:
            audio = recognizer.listen(source, timeout=10)  # Wait 10 seconds max for speech
            logging.debug("Audio captured, processing...")
            
            user_input = recognizer.recognize_google(audio)  # Recognize speech
            logging.debug(f"Recognized input: {user_input}")
            
            return user_input
                
        except sr.WaitTimeoutError:
            logging.warning("Listening stopped due to inactivity.")
            speak("Listening stopped due to inactivity.")
            return None
        except sr.UnknownValueError:
            logging.warning("Could not understand audio.")
            speak("Sorry, I didn't catch that.")
            return None
        except sr.RequestError as e:
            logging.error(f"Request error: {e}")
            speak(f"Could not request results; {e}")
            return None

