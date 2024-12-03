# Chat Counselor: Voice-Based Mental Health Support System

This README provides step-by-step instructions on how to set up and run the **Chat Counselor** project.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running the Server](#running-the-server)
5. [Running the Client Application](#running-the-client-application)
6. [Optional: Running the TTS Module](#optional-running-the-tts-module)
7. [Future Work](#future-work)
8. [References](#references)

---

## Introduction

**Chat Counselor** is a voice-based mental health support system that allows users to interact with a language model using natural speech. It leverages technologies like FastAPI, LangChain, AWS Polly, and React Native to provide empathetic and insightful responses to users' concerns.

---

## Prerequisites

Ensure you have the following installed before proceeding:

- **Python 3.7 or higher**
- **Node.js and npm**
- **Expo CLI**: Install globally using:

  ```bash
  npm install -g expo-cli
  ```

- **AWS Account**: For AWS Polly (with AWS credentials configured)
- **Docker**: For running the LLM server (e.g., Ollama)
- **Datasets** (if you plan to work with embeddings or TTS module):
  - **Psych8k**: For generating embeddings
  - **LJ Speech** and **LibriSpeech**: For the TTS module

---

## Installation

### 1. Clone the Repository

Open your terminal and execute:

```bash
git clone https://github.com/yourusername/chat-counselor.git
cd chat-counselor
```

### 2. Set Up the Server Environment

Navigate to the server directory:

```bash
cd Chat-Counselor-Server
```

Create and activate a virtual environment:

- On macOS/Linux:

  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

- On Windows:

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Set Up the Client Application

Navigate to the client application directory:

```bash
cd ../ClientApplication
```

Install the necessary npm packages:

```bash
npm install
```

### 4. Configure Environment Variables

#### For the Server

In the `Chat-Counselor-Server` directory, create a `.env` file and add your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=your_aws_region
```

Replace the placeholders with your actual AWS credentials and region.

#### For the Client

In the `ClientApplication` directory, create a `.env` file and set the server URL:

```env
EXPO_PUBLIC_API_URL=http://<your-ip-address>:8000
```

Replace `<your-ip-address>` with the IP address of the machine running the server.

---

## Running the Server

### 1. Start the LLM Server

If you're using an LLM server like **Ollama**, start it using Docker:

```bash
docker run -p 11434:11434 ghcr.io/jmorganca/ollama:v0.0.11
```

Ensure the LLM server is up and running.

### 2. Run the FastAPI Server

Navigate to the server directory if you're not already there:

```bash
cd Chat-Counselor-Server
```

Activate the virtual environment:

- On macOS/Linux:

  ```bash
  source venv/bin/activate
  ```

- On Windows:

  ```bash
  venv\Scripts\activate
  ```

Start the FastAPI server:

```bash
uvicorn scripts.LLM_AWS:app --host 0.0.0.0 --port 8000
```

**Notes:**

- `--host 0.0.0.0` makes the server accessible from other devices on your network.
- Ensure your firewall allows incoming connections on port `8000`.

### 3. Verify the Server is Running

Open a web browser and navigate to:

```
http://<your-ip-address>:8000/docs
```

You should see the FastAPI automatic API documentation page.

---

## Running the Client Application

### 1. Start the Expo Development Server

Navigate to the client application directory:

```bash
cd ClientApplication
```

Start the Expo server:

```bash
npx expo start
```

This will open the Expo Dev Tools in your browser.

### 2. Run the App on Your Mobile Device

- Install the **Expo Go** app from the App Store (iOS) or Google Play Store (Android).
- Ensure your mobile device is connected to the same Wi-Fi network as your development machine.
- In the Expo Dev Tools, scan the QR code using the Expo Go app to launch the application.

### 3. Troubleshooting

- If the app doesn't load, check that `EXPO_PUBLIC_API_URL` in your `.env` file matches the server's IP address and port.
- Ensure there are no network issues or firewall restrictions blocking the connection.

---

## Optional: Running the TTS Module

### 1. Prepare the Datasets

Download and place the **LJ Speech** and **LibriSpeech** datasets into the `tts` directory of the project.

### 2. Install Additional Dependencies

If the TTS module requires additional Python packages, install them while your virtual environment is activated:

```bash
pip install -r tts/requirements.txt
```

### 3. Run the TTS Test Scripts

Navigate to the `tts` directory:

```bash
cd tts
```

Run the test scripts:

```bash
python test-ljspeech.py
python test-librispeech.py
```

These scripts will evaluate different TTS models and output their performance metrics.

---

## Future Work

- **Local TTS Solution**: Implement a local TTS model to replace AWS Polly for improved performance and reduced costs.
- **Enhanced Conversational AI**: Integrate advanced dialogue management and sentiment analysis to provide more personalized responses.
- **Scalability and Deployment**: Containerize the application using Docker for easier deployment and scalability.

---

## References

1. **The LJ Speech Dataset**  
   [https://keithito.com/LJ-Speech-Dataset](https://keithito.com/LJ-Speech-Dataset)

2. **EmoCareAI/Psych8k Dataset**  
   [https://huggingface.co/datasets/EmoCareAI/Psych8k](https://huggingface.co/datasets/EmoCareAI/Psych8k)

3. **ChatCounselor: A Large Language Model for Mental Health Support**  
   [https://arxiv.org/abs/2309.15461](https://arxiv.org/abs/2309.15461)

---
