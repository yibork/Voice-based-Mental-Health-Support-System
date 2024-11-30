Certainly! Here's the README in Markdown format suitable for GitHub:

---

# Chat Counselor: Voice-based Mental Health Support System

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
   - [Technology Stack](#technology-stack)
   - [System Architecture](#system-architecture)
3. [Folder Structure](#folder-structure)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running the Server](#running-the-server)
   - [Running the Client Application](#running-the-client-application)
5. [Implementation Details](#implementation-details)
   - [Dataset Preparation and Embedding Generation](#dataset-preparation-and-embedding-generation)
   - [Audio Processing](#audio-processing)
   - [Query Construction and LLM Integration](#query-construction-and-llm-integration)
   - [Text-to-Speech Conversion with AWS Polly](#text-to-speech-conversion-with-aws-polly)
6. [Text-to-Speech (TTS) Module](#text-to-speech-tts-module)
7. [Future Work](#future-work)
8. [Conclusion](#conclusion)
9. [References](#references)

---

## Introduction

**Chat Counselor** is a voice-based mental health support system that enables users to interact with a language model (LLM) through natural speech. The system provides empathetic and insightful responses to users' mental health concerns, leveraging advanced technologies such as FastAPI, LangChain, AWS Polly, and React Native.

This project integrates several open-source tools to create an end-to-end pipeline where users can converse with a chat counselor using voice input and receive voice responses in real-time.

---

## Architecture

### Technology Stack

- **React Native**: Cross-platform mobile application framework for iOS and Android.
- **FastAPI**: High-performance web framework for building APIs with Python.
- **AWS Polly**: Amazon's cloud service that turns text into lifelike speech.
- **LangChain**: Framework for developing applications powered by language models.
- **SpeechRecognition**: Python library for performing speech recognition.
- **FAISS**: Facebook AI Similarity Search, for efficient vector similarity search.

### System Architecture

The system architecture consists of the following components:

1. **Mobile Application**:
   - Built with React Native.
   - Captures user audio input.
   - Connects to the FastAPI server via WebSocket.
   - Receives audio responses from the server.

2. **FastAPI Server**:
   - Manages WebSocket connections with the client.
   - Converts user audio input to text using SpeechRecognition.
   - Processes text queries through LangChain and the LLM.
   - Converts LLM responses from text to speech using AWS Polly.
   - Sends audio responses back to the client.

3. **Language Model (LLM) and LangChain**:
   - Uses LangChain to interface with the LLM (e.g., LLaMA).
   - Employs Retrieval-Augmented Generation (RAG) for context-aware responses.
   - Retrieves relevant information from embeddings stored in FAISS.

4. **Embeddings and Vector Store**:
   - Uses FAISS for efficient similarity search over embeddings.
   - Embeddings are generated from the Psych8k dataset.

---

## Folder Structure

The project consists of three main folders:

1. **`Chat-Counselor-Server`**:
   - Contains the FastAPI server code.
   - Handles user queries and provides insights about mental health problems.
   - Uses WebSocket API for communication with the client side.

2. **`Client Application`**:
   - Mobile application built with React Native.
   - Allows users to interact with the chat counselor via voice input.
   - Communicates with the server over WebSocket.

3. **`TTS Module`**:
   - Contains scripts for comparing different Text-to-Speech (TTS) systems.
   - Includes `test-ljspeech.py` and `test-librispeech.py` for testing TTS models on the LJ Speech and LibriSpeech datasets respectively.
   - Results are detailed in the accompanying research paper.

---

## Getting Started

### Prerequisites

- **Python 3.7+**
- **Node.js and npm**
- **Expo CLI**: Install globally using `npm install -g expo-cli`.
- **AWS Account**: For AWS Polly (ensure you have AWS credentials).
- **Docker**: For running the LLM server (e.g., Ollama in a Docker container).
- **Datasets**: Psych8k dataset for embeddings, LJ Speech and LibriSpeech datasets for TTS module.

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/chat-counselor.git
   cd chat-counselor
   ```

2. **Set Up the Server Environment**:

   ```bash
   cd Chat-Counselor-Server
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Set Up the Client Application**:

   ```bash
   cd ../ClientApplication
   npm install
   ```

4. **Set Up Environment Variables**:

   - In the server directory, create a `.env` file and add your AWS credentials:

     ```env
     AWS_ACCESS_KEY_ID=your_aws_access_key_id
     AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
     AWS_REGION=your_aws_region
     ```

   - In the client application, create a `.env` file and add the server URL:

     ```env
     EXPO_PUBLIC_API_URL=http://<your-ip-address>:8000
     ```

     Replace `<your-ip-address>` with the actual IP address of your machine.

### Running the Server

1. **Start the LLM Server**:

   - If you're using Ollama or another LLM server, ensure it's running and accessible.

2. **Run the FastAPI Server**:

   Navigate to the `Chat-Counselor-Server` directory and run:

   ```bash
   uvicorn scripts.LLM-AWS:app --host <your-ip-address> --port 8000
   ```

   Replace `<your-ip-address>` with the actual IP address of your machine.

   **Example**:

   ```bash
   uvicorn scripts.LLM-AWS:app --host 192.168.1.10 --port 8000
   ```

3. **Ensure the Server is Running**:

   - You can test the server by navigating to `http://<your-ip-address>:8000/docs` to see the FastAPI automatic documentation.

### Running the Client Application

1. **Start the Expo Development Server**:

   ```bash
   cd ClientApplication
   npx expo start
   ```

2. **Run the App on Your Mobile Device**:

   - Install the **Expo Go** app on your mobile device.
   - Scan the QR code displayed in the terminal or browser.
   - Ensure your mobile device is on the same network as your development machine.

3. **Update the `.env` File in the Client Application**:

   - Make sure the `EXPO_PUBLIC_API_URL` in your `.env` file matches the server's IP and port.

---

## Implementation Details

### Dataset Preparation and Embedding Generation

- **Dataset**: Psych8k dataset, a collection of mental health dialogues.
- **Processing**:
  - Convert the dataset into a format suitable for embedding generation.
  - Each data item is structured as a `Document` object with relevant metadata.

- **Embedding Creation**:
  - Use HuggingFace embeddings to generate vector representations of the documents.
  - Store embeddings in FAISS, a high-performance vector store.
  - Save the embeddings to a file (`embeddings/embeddings.pkl`) for quick retrieval during query processing.

**Code Snippet**:

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from datasets import load_dataset
import pickle

def generate_and_save_embeddings():
    # Load the Psych8k dataset
    dataset = load_dataset("EmoCareAI/Psych8k")

    # Process the dataset to create documents
    documents = []
    for data in dataset['train']:
        content = f"Input: {data['Input']}\nOutput: {data['Output']}"
        documents.append(Document(page_content=content, metadata={"source": "Psych8k"}))

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save the vector store
    with open("embeddings/embeddings.pkl", "wb") as f:
        pickle.dump(vector_store, f)

if __name__ == "__main__":
    generate_and_save_embeddings()
```

### Audio Processing

- **Speech Recognition**:
  - Use the `SpeechRecognition` library to convert user audio input into text.
  - Audio data is received via WebSocket and processed in-memory.

**Code Snippet**:

```python
import speech_recognition as sr
from fastapi import WebSocket
import io

recognizer = sr.Recognizer()

async def process_audio(websocket: WebSocket):
    audio_data = await websocket.receive_bytes()
    audio_stream = io.BytesIO(audio_data)

    with sr.AudioFile(audio_stream) as source:
        audio_recorded = recognizer.record(source)
        user_input = recognizer.recognize_google(audio_recorded)
    return user_input
```

### Query Construction and LLM Integration

- **Prompt Template**:
  - Create a structured prompt that includes context and chat history.
  - Use `PromptTemplate` from LangChain.

- **LLM Chain**:
  - Use LangChain's `LLMChain` to interface with the LLM.
  - Implement a `RetrievalQA` chain for context-aware responses.

**Code Snippet**:

```python
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

prompt_template = """
1. Use the following context to answer the question.
2. Provide empathetic and supportive responses.
3. If unsure, say "I don't know," but do not fabricate an answer.
4. Keep the answer concise, ideally 2 sentences.

Context: {context}

Question: {question}

Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

llm = Ollama(model="llama3")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    retriever=retriever
)
```

### Text-to-Speech Conversion with AWS Polly

- **AWS Polly Integration**:
  - Use AWS Polly to convert text responses from the LLM into speech.
  - Configure AWS credentials and specify voice parameters.

**Code Snippet**:

```python
import boto3
from io import BytesIO

def speak(text):
    polly = boto3.client('polly', region_name='your-region')
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Nicole'
    )
    audio_stream = BytesIO(response['AudioStream'].read())
    audio_bytes = audio_stream.getvalue()
    return audio_bytes
```

---

## Text-to-Speech (TTS) Module

The TTS module is responsible for evaluating different TTS systems to find the most suitable one for the application.

### Files:

- **`test-ljspeech.py`**:
  - Tests various TTS models on the LJ Speech dataset.
  - Compares performance, inference speed, and accuracy.

- **`test-librispeech.py`**:
  - Similar to `test-ljspeech.py`, but uses the LibriSpeech dataset.

### Running the TTS Tests:

1. **Prepare the Datasets**:
   - Place the LJ Speech and LibriSpeech datasets in the `tts` directory.

2. **Run the Tests**:

   ```bash
   cd tts
   python test-ljspeech.py
   python test-librispeech.py
   ```

3. **Review the Results**:
   - The results are detailed in the accompanying research paper.
   - They include metrics on accuracy, inference speed, and cost.

---

## Future Work

- **Local TTS Solution**:
  - Transition from AWS Polly to a locally hosted TTS model.
  - Evaluate models for accuracy, speed, and cost-effectiveness.

- **Enhanced Conversational AI**:
  - Implement advanced dialogue management for better multi-turn conversations.
  - Incorporate sentiment analysis to adjust responses based on user emotions.

- **Scalability and Deployment**:
  - Containerize the application for easier deployment.
  - Optimize performance for handling multiple simultaneous users.

---

## Conclusion

In summary, this project successfully demonstrates the integration of multiple technologies to create a comprehensive voice-based chat counselor application. By combining frameworks like React Native, FastAPI, AWS Polly, and LangChain, the application achieves seamless end-to-end communication between users and a language model. This setup provides real-time responses and a user-friendly experience.

While the foundational components of the system are operational, further refinements are necessary to enhance performance, particularly in the text-to-speech module. Testing alternative TTS models locally will yield insights on accuracy, speed, and cost, helping to make a data-informed decision for future iterations.

---

## References

1. **The LJ Speech Dataset**  
   Accessed: Sep. 08, 2024  
   [https://keithito.com/LJ-Speech-Dataset](https://keithito.com/LJ-Speech-Dataset)

2. **EmoCareAI/Psych8k Dataset**  
   Accessed: Sep. 08, 2024  
   [https://huggingface.co/datasets/EmoCareAI/Psych8k](https://huggingface.co/datasets/EmoCareAI/Psych8k)

3. **ChatCounselor: A Large Language Models for Mental Health Support**  
   J. M. Liu, D. Li, H. Cao, T. Ren, Z. Liao, and J. Wu, Sep. 27, 2023  
   DOI: [10.48550/arXiv.2309.15461](https://arxiv.org/abs/2309.15461)
