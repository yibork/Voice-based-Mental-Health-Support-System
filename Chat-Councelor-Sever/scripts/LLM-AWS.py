import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
import pyttsx3
import soundfile as sf
import numpy as np
import speech_recognition as sr
import pygame
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
from playsound import playsound
import logging
import pickle
# from listen import listen
# from speak import speak
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import io
app = FastAPI()
import os
import pygame
from io import BytesIO
import boto3
from dotenv import load_dotenv
import logging


load_dotenv()

# Retrieve AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'us-west-2')

# Initialize a session using Amazon Polly
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Initialize the Polly client
polly = session.client('polly')

def speak(text):
    """Converts text to speech using Amazon Polly and returns the audio bytes."""
    try:
        # Call Amazon Polly to synthesize the speech
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Nicole'  # You can change the voice here
        )

        # Check if the response contains audio stream
        if 'AudioStream' in response:
            # Use BytesIO to handle the audio stream in memory
            audio_stream = BytesIO(response['AudioStream'].read())
            audio_bytes = audio_stream.getvalue()  # Get audio bytes
            
            return audio_bytes  
        else:
            logging.error("Error: No audio stream returned from Amazon Polly.")
            return None  # Return None if there's no audio stream
    
    except Exception as e:
        logging.error(f"Error occurred in speak function: {e}")
        return None  # Return None on error

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize pygame mixer before playing audio
try:
    if not pygame.mixer.get_init():
        pygame.mixer.init()
        logging.info("Pygame mixer initialized.")
except pygame.error as e:
    logging.error(f"Error initializing pygame mixer: {e}")

# Initialize speech recognizer

EMBEDDINGS_FILE = "embeddings/embeddings.pkl"


# Instantiate the embedding model
with open(EMBEDDINGS_FILE, "rb") as f:
    vector = pickle.load(f)


# Create the vector store 
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the prompt for the QA chain
prompt = """
1. Use the following pieces of context to answer the question at the end. Aim to provide empathetic and supportive responses.
2. If you're uncertain about something, say "I don't know," but do not fabricate an answer. 
3. Your answer should be concise and insightful, ideally 2 sentences long, offering practical suggestions. Do not make it more than 90 words long.

Context: {context}

Question: {question}

Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

# Create the LLM chain
llm = Ollama(model="llama3")
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=True)

# Create the document prompt
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

# Create the combine documents chain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)

# Create the RetrievalQA chain
qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    verbose=True,
    retriever=retriever,
    return_source_documents=True,
)

# chat_history = []
# speak("How can I help you today?")

# Endless loop for user interaction
# while True:
#     # Get user input via speech
#     user_input = listen()
    
#     if user_input is None:
#         continue

#     # Check for exit condition
#     if user_input.lower() in ['exit', 'quit']:
#         speak("Exiting the mental health assistant. Take care!")
#         break

#     # Construct the full query with chat history
#     full_query = f"Chat History: {chat_history}\n\nCurrent Question: {user_input}"

#     # Process the input through the QA chain
#     result = qa(full_query)
    
#     # Speak and print the assistant's response
#     assistant_response = result["result"]
#     speak(assistant_response)
#     print("Assistant:", assistant_response)

#     # Update chat history
#     chat_history.append((user_input, assistant_response))

#     # Limit the chat history to the last 5 interactions
#     chat_history = chat_history[-5:]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_history = []  # Keep track of chat history for the user
    try:
        while True:
            # Receive message (user input) from the WebSocket
            user_input = await websocket.receive_text()
            logging.info(f"Received: {user_input}")

            # Exit condition for quitting
            if user_input.lower() in ['exit', 'quit']:
                await websocket.send_text("Exiting the mental health assistant. Take care!")
                break

            # Construct the full query with chat history
            full_query = f"Chat History: {chat_history}\n\nCurrent Question: {user_input}"

            # Process the input through the QA chain
            result = qa(full_query)
            assistant_response = result["result"]
            logging.info(f"Assistant: {assistant_response}")

            # Send the response back to the client
            await websocket.send_text(assistant_response)

            # Update chat history
            chat_history.append((user_input, assistant_response))

            # Limit the chat history to the last 5 interactions
            chat_history = chat_history[-5:]

    except WebSocketDisconnect:
        logging.info("Client disconnected")
@app.websocket("/ws-audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_history = []  # Keep track of chat history for the user
    recognizer = sr.Recognizer()

    try:
        while True:
            # Receive message (user input) from the WebSocket
            message = await websocket.receive_bytes()  # Receive audio data
            if message:  # Check if there is audio data
                logging.info("Received audio data")

                # Convert audio data to text
                audio_data = io.BytesIO(message)
                try:
                    # Use SpeechRecognition to recognize the audio
                    with sr.AudioFile(audio_data) as source:
                        audio_recorded = recognizer.record(source)
                        user_input = recognizer.recognize_google(audio_recorded)
                        logging.info(f"Recognized text: {user_input}")
                        if user_input.lower() in ['exit', 'quit']:
                            await websocket.send_text("Exiting the mental health assistant. Take care!")
                            break

                        # Construct the full query with chat history
                        full_query = f"Chat History: {chat_history}\n\nCurrent Question: {user_input}"

                        # Process the input through the QA chain
                        result = qa(full_query)
                        assistant_response = result["result"]
                        # audio_bytes = speak(assistant_response)
                        # if audio_bytes:
                        #     print(f"Audio response size: {len(audio_bytes)} bytes")
                                                # print(audio_bytes)
                        # if audio_bytes is not None:
                        print("Sending audio response")
                        # Send the audio bytes back to the client
                        await websocket.send_text(assistant_response)  # Send audio bytes
                        # else:
                        #     await websocket.send_text("Error generating audio response.")

                        logging.info(f"Assistant: {assistant_response}")

                        # Update chat history
                        chat_history.append((user_input, assistant_response))

                        # Limit the chat history to the last 5 interactions
                        chat_history = chat_history[-5:]

                except sr.UnknownValueError:
                    logging.error("Could not understand audio")
                    await websocket.send_text("Could not understand audio")
                except sr.RequestError as e:
                    logging.error(f"Could not request results from Google Speech Recognition service; {e}")
                    await websocket.send_text("Could not request results from service")

    except WebSocketDisconnect:
        logging.info("Client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

