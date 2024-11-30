import os
import time
import librosa
import numpy as np
import pandas as pd
import json
import re
import boto3
from pydub import AudioSegment
from scipy.spatial.distance import euclidean
from python_speech_features import mfcc
from jiwer import wer, cer
from pystoi import stoi
from pesq import pesq
import torch
from TTS.api import TTS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import whisper  
import soundfile as sf

# ----------------------------- Configuration -----------------------------

# Path to the chapters.txt file
CHAPTERS_FILE = "LibriTTS/CHAPTERS.txt"  # Update with actual path

# Base directory for LibriTTS dataset
LIBRITTS_BASE_DIR = "LibriTTS/test-clean"  # Update with actual path

# Output directory for synthesized audios and plots
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# AWS Polly Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')  # Default region if not set

# Number of samples to evaluate (set to None to evaluate all)
NUM_SAMPLES = None  # e.g., 10 or None

# TTS Models to evaluate (add or remove models as needed)
MODEL_NAMES = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/speedy-speech",
    "tts_models/en/ljspeech/fast_pitch",
    "tts_models/en/ljspeech/tacotron2-DCA",
    "tts_models/en/ljspeech/glow-tts",
    "aws_polly"
]

# ----------------------------- Functions -----------------------------

def compute_cer(reference_text, synthesized_text):
    """Compute Character Error Rate."""
    if synthesized_text is None:
        return None
    return cer(reference_text, synthesized_text)

def compute_wer_metric(reference_text, synthesized_text):
    """Compute Word Error Rate."""
    if synthesized_text is None:
        return None
    return wer(reference_text, synthesized_text)

def compute_mcd(reference, synthesized, sr=22050):
    """Compute Mel Cepstral Distortion."""
    ref_mfcc = mfcc(reference, samplerate=sr)
    synth_mfcc = mfcc(synthesized, samplerate=sr)
    min_length = min(len(ref_mfcc), len(synth_mfcc))
    ref_mfcc, synth_mfcc = ref_mfcc[:min_length], synth_mfcc[:min_length]
    mcd = np.mean([euclidean(r, s) for r, s in zip(ref_mfcc, synth_mfcc)])
    return mcd

def compute_mmsd(reference, synthesized, sr=22050, n_fft=1024, hop_length=256):
    """Compute Mel-Magnitude Spectral Distance."""
    ref_mel = librosa.feature.melspectrogram(y=reference, sr=sr, n_fft=n_fft, hop_length=hop_length)
    synth_mel = librosa.feature.melspectrogram(y=synthesized, sr=sr, n_fft=n_fft, hop_length=hop_length)
    min_length = min(ref_mel.shape[1], synth_mel.shape[1])
    ref_mel, synth_mel = ref_mel[:, :min_length], synth_mel[:, :min_length]
    mmsd = np.mean((ref_mel - synth_mel) ** 2)
    return mmsd

def load_audio(file_path, sr=22050):
    """Load an audio file."""
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def match_length(reference, synthesized):
    """Match the length of synthesized audio to the reference."""
    if len(synthesized) < len(reference):
        synthesized = np.pad(synthesized, (0, len(reference) - len(synthesized)), mode='constant')
    elif len(synthesized) > len(reference):
        synthesized = synthesized[:len(reference)]
    return synthesized

def compute_stoi_metric(reference, synthesized, sr=22050):
    """Compute Short-Time Objective Intelligibility."""
    return stoi(reference, synthesized, sr, extended=False)

def compute_pesq_metric(reference, synthesized, sr=22050):
    """Compute Perceptual Evaluation of Speech Quality."""
    # Resample to 16000 Hz if necessary
    if sr != 16000:
        reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
        synthesized = librosa.resample(synthesized, orig_sr=sr, target_sr=16000)
    try:
        pesq_score = pesq(16000, reference, synthesized, 'wb')  # Wideband PESQ
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        pesq_score = None
    return pesq_score

def normalize_text(text):
    """Normalize text by removing unsupported characters."""
    text = re.sub(r'[͡]', '', text)
    return text

def parse_chapters(chapters_file):
    """
    Parse the chapters.txt file to extract sample mappings.

    Args:
        chapters_file (str): Path to the chapters.txt file.

    Returns:
        List of tuples: [(sample_id, text, wav_path), ...]
    """
    samples = []
    with open(chapters_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "" or line.startswith(';'):
                continue  # Skip empty lines and comments
            parts = [part.strip() for part in line.strip().split('|')]
            if len(parts) < 8:
                print(f"Skipping malformed line: {line}")
                continue

            # Extract metadata from the line
            chapter_id = parts[0]
            speaker_id = parts[1]
            subset = parts[3]

            # Skip non-test-clean subsets
            if "test-clean" not in subset:
                continue

            # Construct the sample directory based on speaker and chapter ID
            sample_dir = os.path.join(LIBRITTS_BASE_DIR, speaker_id, chapter_id)
            if not os.path.isdir(sample_dir):
                print(f"Directory not found: {sample_dir}")
                continue

            # Look for .wav files in the sample directory
            wav_files = [
                f for f in os.listdir(sample_dir) 
                if f.endswith(".wav")
            ]

            for wav_file in wav_files:
                # Ensure corresponding normalized text file exists
                text_file = wav_file.replace(".wav", ".normalized.txt")
                wav_path = os.path.join(sample_dir, wav_file)
                text_path = os.path.join(sample_dir, text_file)

                if not os.path.isfile(text_path):
                    print(f"Text file not found: {text_path}")
                    continue
                if not os.path.isfile(wav_path):
                    print(f"WAV file not found: {wav_path}")
                    continue

                # Read the normalized text
                with open(text_path, 'r', encoding='utf-8') as tf:
                    text = tf.read().strip()

                # Add to samples
                sample_id = f"{speaker_id}_{chapter_id}_{wav_file.split('_')[2]}"
                samples.append((sample_id, text, wav_path))

    return samples
def convert_to_builtin(obj):
    """Convert non-serializable objects to serializable ones."""
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)  # Fallback to string representation

def initialize_polly_client():
    """Initialize AWS Polly client using environment variables."""
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = AWS_REGION

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials are not set in environment variables.")

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    polly_client = session.client('polly')
    return polly_client

def transcribe_audio(asr_model, audio_path):
    """Transcribe audio using Whisper ASR model."""
    try:
        result = asr_model.transcribe(audio_path)
        return result['text'].strip()
    except Exception as e:
        print(f"ASR transcription failed for {audio_path}: {e}")
        return None

# ----------------------------- Main Script -----------------------------

def main():
    # Initialize AWS Polly client
    try:
        polly_client = initialize_polly_client()
    except Exception as e:
        print(f"Error initializing AWS Polly client: {e}")
        return

    # Initialize Whisper ASR model
    try:
        print("Loading Whisper ASR model...")
        asr_model = whisper.load_model("base")  # Choose model size as needed
        print("Whisper ASR model loaded successfully.")
    except Exception as e:
        print(f"Error loading Whisper ASR model: {e}")
        return

    # Parse chapters.txt and get sample mappings
    print("Parsing chapters.txt and mapping samples...")
    samples = parse_chapters(CHAPTERS_FILE)
    print(f"Total test-clean samples found: {len(samples)}")

    if not samples:
        print("No samples to process. Exiting.")
        return

    # If NUM_SAMPLES is set, limit the number of samples
    if NUM_SAMPLES is not None:
        samples = samples[:NUM_SAMPLES]
        print(f"Limited to {NUM_SAMPLES} samples for evaluation.")

    # Initialize metrics storage
    metrics = []

    # Iterate over each TTS model
    for model_name in MODEL_NAMES:
        print(f"\nEvaluating Model: {model_name}")
        if model_name != "aws_polly":
            try:
                tts = TTS(model_name=model_name, progress_bar=False)
                print(f"Loaded TTS model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                continue

        # Process only a specific number of samples if needed
        processed_count = 0
        for sample_id, text, wav_path in samples:
            processed_count += 1

            print(f"\nProcessing Sample ID: {sample_id}")
            normalized_text = normalize_text(text)

            # Sanitize model_name by replacing slashes with underscores
            sanitized_model_name = model_name.replace('/', '_')

            # Define output file paths based on sanitized_model_name and sample_id
            if model_name == "aws_polly":
                # Define paths for AWS Polly (MP3 and converted WAV)
                output_path = os.path.join(
                    OUTPUT_DIR, f"{sanitized_model_name}_{sample_id}.mp3"
                )
            else:
                # Define path for local TTS models (WAV)
                output_audio_path = os.path.join(OUTPUT_DIR, f"{sanitized_model_name}_{sample_id}.wav")

            # Synthesize speech
            start_time = time.time()
            try:
                if model_name == "aws_polly":
                    # Synthesize speech using AWS Polly with mp3 output
                    response = polly_client.synthesize_speech(
                        Text=text,
                        OutputFormat='mp3',
                        VoiceId='Joanna',
                        SampleRate='22050'
                    )

                    # Save the MP3 audio stream
                    if "AudioStream" in response:
                        with open(output_path, 'wb') as audio_file:
                            audio_stream = response['AudioStream'].read()
                            audio_file.write(audio_stream)
                        inference_time = time.time() - start_time
                    else:
                        print(f"Could not synthesize speech for sample {sample_id} with AWS Polly.")
                        continue

                    # Convert MP3 to WAV
                    print(f"Audio synthesized with AWS Polly in {inference_time:.2f} seconds.")

                    # Load MP3 file
                    audio, sr = librosa.load(output_path, sr=None)  # Preserve the original sampling rate
                    print(f"Loaded audio with shape: {audio.shape} and sampling rate: {sr}")

                    # Resample to a new frame rate (e.g., 22050 Hz)
                    target_sr = 22050
                    if sr != target_sr:
                        print(f"Resampling audio to {target_sr} Hz...")
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                        sr = target_sr
                        print(f"Resampled audio shape: {audio.shape}")

                    # Convert to mono (if stereo)
                    if len(audio.shape) > 1:
                        print("Converting audio to mono...")
                        audio = librosa.to_mono(audio)

                    # Define the WAV output path
                    converted_output_path = os.path.splitext(output_path)[0] + ".wav"

                    # Export the audio to WAV
                    print(f"Exporting WAV file to: {converted_output_path}...")
                    sf.write(converted_output_path, audio, sr)
                    print(f"Exported WAV file: {converted_output_path}")

                    # Remove the original MP3 file
                    os.remove(output_path)
                    print(f"Original MP3 file removed: {output_path}")

                    # Update the `output_path` to the converted WAV file
                    output_path = converted_output_path
                    print(f"Updated output path: {output_path}")
                else:
                    # Synthesize speech using the local TTS model
                    tts.tts_to_file(text=normalized_text, file_path=output_audio_path)
                    inference_time = time.time() - start_time
                    print(f"Audio synthesized with {model_name} in {inference_time:.2f} seconds.")
            except Exception as e:
                print(f"Error generating audio for sample {sample_id} with model {model_name}: {e}")
                continue  # Skip to the next sample

            # Load synthesized and reference audios
            try:
                synthesized_audio = load_audio(output_audio_path)
                reference_audio = load_audio(wav_path)
                print(f"Loaded synthesized and reference audios for sample {sample_id}.")
            except Exception as e:
                print(f"Error loading audio files for sample {sample_id}: {e}")
                continue

            # Match lengths of reference and synthesized audios
            synthesized_audio = match_length(reference_audio, synthesized_audio)
            print(f"Matched audio lengths for sample {sample_id}.")

            # Compute metrics
            try:
                stoi_score = compute_stoi_metric(reference_audio, synthesized_audio)
                pesq_score = compute_pesq_metric(reference_audio, synthesized_audio, sr=22050)
                mcd_score = compute_mcd(reference_audio, synthesized_audio)
                mmsd_score = compute_mmsd(reference_audio, synthesized_audio)

                # Transcribe synthesized audio for WER and CER
                synthesized_text = transcribe_audio(asr_model, output_audio_path)
                if synthesized_text:
                    wer_score = compute_wer_metric(normalized_text, synthesized_text)
                    cer_score = compute_cer(normalized_text, synthesized_text)
                else:
                    wer_score = None
                    cer_score = None

                print(f"Computed metrics for sample {sample_id}: "
                      f"STOI={stoi_score:.4f}, PESQ={pesq_score:.2f}, "
                      f"MCD={mcd_score:.2f}, MMSD={mmsd_score:.4f}, "
                      f"WER={wer_score}, CER={cer_score}")
            except Exception as e:
                print(f"Error computing metrics for sample {sample_id} in {model_name}: {e}")
                stoi_score = pesq_score = mcd_score = mmsd_score = wer_score = cer_score = None

            # Record metrics
            metrics.append({
                "model_name": model_name,
                "sample_id": sample_id,
                "inference_time_sec": inference_time,
                "real_time_factor": inference_time / len(reference_audio) if len(reference_audio) > 0 else None,
                "stoi_score": stoi_score,
                "pesq_score": pesq_score,
                "mcd_score": mcd_score,
                "mmsd_score": mmsd_score,
                "wer_score": wer_score,
                "cer_score": cer_score
            })


    # Save metrics to JSON for later analysis
    metrics_json_path = os.path.join(OUTPUT_DIR, "tts_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4, default=convert_to_builtin)

    print(f"\nSaved metrics to {metrics_json_path}")

if __name__ == "__main__":
    main()
