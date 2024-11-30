import json
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
import pyttsx3
import speech_recognition as sr
import pygame
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
from playsound import playsound

# Load environment variables from .env file
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

# Initialize pygame mixer before playing audio
try:
    pygame.mixer.init()
    print("Pygame mixer initialized.")
except pygame.error as e:
    print(f"Error initializing pygame mixer: {e}")

# Initialize speech recognizer
recognizer = sr.Recognizer()

def speak(text):
    """Converts text to speech using Amazon Polly and plays the audio from memory."""
    
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
            
            # Load the audio stream into pygame
            pygame.mixer.music.load(audio_stream, 'mp3')
            pygame.mixer.music.play()

            # Wait until the audio is finished playing
            while pygame.mixer.music.get_busy():
                continue

        else:
            print("Error: No audio stream returned from Amazon Polly.")
    
    except Exception as e:
        print(f"Error occurred: {e}")

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        # Adjust the recognizer's sensitivity to silence
        recognizer.pause_threshold = 1.5  # Shorten the threshold to detect pauses faster
        try:
            audio = recognizer.listen(source, timeout=10)  # Wait 10 seconds max for speech
            print("Recognizing...")
            user_input = recognizer.recognize_google(audio)  # Recognize speech
            return user_input
        except sr.WaitTimeoutError:
            print("Listening stopped due to inactivity.")
            return None
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None


# Load the JSON file
with open("Psych8k/Alexander_Street_shareGPT_2.0.json", "r") as file:
    json_data = json.load(file)

# Initialize an empty list to hold the documents
documents = []

# Iterate over each item in the JSON data
for item in json_data:
    # Create a document for each 'input' or 'output' with source metadata
    if 'input' in item:
        documents.append(Document(page_content=item['input'], metadata={"source": "input"}))
    if 'output' in item:
        documents.append(Document(page_content=item['output'], metadata={"source": "output"}))

# Check the number of documents loaded
print("Number of documents loaded:", len(documents))

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Create the vector store 
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the prompt for the QA chain
prompt = """
1. Use the following pieces of context to answer the question at the end. Aim to provide empathetic and supportive responses.
2. If you're uncertain about something, say "I don't know," but do not fabricate an answer. 
3. Your answer should be concise and insightful, ideally 3-4 sentences long, offering practical suggestions.

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

chat_history = []
speak("How can I help you today?")

# Endless loop for user interaction
while True:
    # Get user input via speech
    user_input = listen()
    
    if user_input is None:
        continue

    # Check for exit condition
    if user_input.lower() in ['exit', 'quit']:
        speak("Exiting the mental health assistant. Take care!")
        break

    # Construct the full query with chat history
    full_query = f"Chat History: {chat_history}\n\nCurrent Question: {user_input}"

    # Process the input through the QA chain
    result = qa(full_query)
    
    # Speak and print the assistant's response
    assistant_response = result["result"]
    speak(assistant_response)
    print("Assistant:", assistant_response)

    # Update chat history
    chat_history.append((user_input, assistant_response))

    # Limit the chat history to the last 5 interactions
    chat_history = chat_history[-5:]
