+++markdown
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
   - [Text-to-Speech (TTS) Module](#text-to-speech-tts-module)
6. [Future Work](#future-work)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction
Chat Counselor is a voice-based mental health support system enabling users to interact with a language model (LLM) through natural speech. It provides empathetic and insightful responses to users' mental health concerns, leveraging technologies like FastAPI, LangChain, AWS Polly, and React Native. The system enables real-time conversations using voice input and output.

## Architecture

### Technology Stack
- **React Native**: Cross-platform mobile application framework for iOS and Android.
- **FastAPI**: High-performance web framework for building APIs with Python.
- **AWS Polly**: Amazon's cloud service for text-to-speech conversion.
- **LangChain**: Framework for building applications powered by language models.
- **SpeechRecognition**: Python library for converting speech to text.
- **FAISS**: Facebook AI Similarity Search for efficient vector similarity search.

### System Architecture
1. **Mobile Application**:
   - Built with React Native.
   - Captures user audio input.
   - Communicates with the FastAPI server via WebSocket.
   - Receives audio responses from the server.

2. **FastAPI Server**:
   - Manages WebSocket connections with the client.
   - Converts user audio to text using SpeechRecognition.
   - Processes text queries through LangChain and the LLM.
   - Converts responses to speech using AWS Polly.
   - Sends audio responses back to the client.

3. **Language Model (LLM) and LangChain**:
   - Interfaces with the LLM (e.g., LLaMA).
   - Uses Retrieval-Augmented Generation (RAG) for context-aware responses.
   - Retrieves relevant information from embeddings stored in FAISS.

4. **Embeddings and Vector Store**:
   - Uses FAISS for efficient similarity search.
   - Embeddings are generated from the Psych8k dataset.

## Folder Structure
1. **Chat-Counselor-Server**:
   - Contains FastAPI server code.
   - Handles user queries.
   - Communicates with the client via WebSocket.

2. **Client Application**:
   - Built with React Native.
   - Allows voice-based interaction.
   - Connects to the server over WebSocket.

3. **TTS Module**:
   - Evaluates different TTS systems.
   - Includes scripts for testing on the LJ Speech and LibriSpeech datasets.

## Getting Started

### Prerequisites
- Python 3.7+
- Node.js and npm
- Expo CLI (`npm install -g expo-cli`)
- AWS Account (for AWS Polly)
- Docker (for running LLM server)
- Required datasets: Psych8k, LJ Speech, and LibriSpeech.

### Installation

#### Clone the Repository
```bash
git clone https://github.com/yourusername/chat-counselor.git
cd chat-counselor
```

#### Set Up the Server Environment
```bash
cd Chat-Counselor-Server
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt
```

#### Set Up the Client Application
```bash
cd ../ClientApplication
npm install
```

#### Set Up Environment Variables
- In the server directory, create a `.env` file:
  ```env
  AWS_ACCESS_KEY_ID=your_aws_access_key_id
  AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
  AWS_REGION=your_aws_region
  ```
- In the client application, create a `.env` file:
  ```env
  EXPO_PUBLIC_API_URL=http://<your-ip-address>:8000
  ```

### Running the Server

1. **Start the LLM Server**:
   Ensure your LLM server (e.g., Ollama) is running.
2. **Run the FastAPI Server**:
   ```bash
   uvicorn scripts.LLM-AWS:app --host <your-ip-address> --port 8000
   ```
   Example:
   ```bash
   uvicorn scripts.LLM-AWS:app --host 192.168.1.10 --port 8000
   ```

### Running the Client Application

1. **Start the Expo Development Server**:
   ```bash
   cd ClientApplication
   npx expo start
   ```
2. **Run the App**:
   - Install the Expo Go app.
   - Scan the QR code to launch the app.

## Implementation Details

### Dataset Preparation and Embedding Generation
- **Dataset**: Psych8k mental health dialogues.
- **Embedding Creation**:
  - Use HuggingFace embeddings.
  - Store embeddings in FAISS for efficient retrieval.

### Audio Processing
- Convert user audio input to text using the SpeechRecognition library.

### Query Construction and LLM Integration
- **Prompt Template**:
  Define structured prompts for LLM interaction.
- **RetrievalQA Chain**:
  Retrieve context-aware answers.

### Text-to-Speech Conversion with AWS Polly
- Converts LLM text responses to speech using AWS Polly.

### Text-to-Speech (TTS) Module
- Evaluates TTS systems on datasets like LJ Speech and LibriSpeech.

## Future Work
- Transition to local TTS solutions.
- Improve conversational AI capabilities.
- Scale and optimize the system.

## Conclusion
This project integrates React Native, FastAPI, AWS Polly, and LangChain to create a voice-based mental health support system. Future enhancements will improve TTS and conversational capabilities.

## References
- [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset)
- [EmoCareAI/Psych8k Dataset](https://huggingface.co/datasets/EmoCareAI/Psych8k)
- ChatCounselor: A Large Language Models for Mental Health Support  
  J. M. Liu, D. Li, H. Cao, T. Ren, Z. Liao, and J. Wu, Sep. 27, 2023  
  DOI: 10.48550/arXiv.2309.15461
+++