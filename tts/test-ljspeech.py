import time
import librosa
import numpy as np
import pandas as pd
import os
from TTS.api import TTS
from pystoi import stoi
from pesq import pesq
import torch
import json
from scipy.spatial.distance import euclidean
from python_speech_features import mfcc
from jiwer import wer, cer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import re
import boto3
from pydub import AudioSegment

# Function to compute CER
def compute_cer(reference_text, synthesized_text):
    return cer(reference_text, synthesized_text)

# Function to compute WER
def compute_wer_metric(reference_text, synthesized_text):
    return wer(reference_text, synthesized_text)

# Function to compute MCD
def compute_mcd(reference, synthesized, sr=22050):
    ref_mfcc = mfcc(reference, samplerate=sr)
    synth_mfcc = mfcc(synthesized, samplerate=sr)
    min_length = min(len(ref_mfcc), len(synth_mfcc))
    ref_mfcc, synth_mfcc = ref_mfcc[:min_length], synth_mfcc[:min_length]
    mcd = np.mean([euclidean(r, s) for r, s in zip(ref_mfcc, synth_mfcc)])
    return mcd

# Function to compute MMSD
def compute_mmsd(reference, synthesized, sr=22050, n_fft=1024, hop_length=256):
    ref_mel = librosa.feature.melspectrogram(y=reference, sr=sr, n_fft=n_fft, hop_length=hop_length)
    synth_mel = librosa.feature.melspectrogram(y=synthesized, sr=sr, n_fft=n_fft, hop_length=hop_length)
    min_length = min(ref_mel.shape[1], synth_mel.shape[1])
    ref_mel, synth_mel = ref_mel[:, :min_length], synth_mel[:, :min_length]
    mmsd = np.mean((ref_mel - synth_mel) ** 2)
    return mmsd

# Load the original audio as reference
def load_audio(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

# Function to match the lengths of reference and synthesized audio
def match_length(reference, synthesized):
    if len(synthesized) < len(reference):
        synthesized = np.pad(synthesized, (0, len(reference) - len(synthesized)), mode='constant')
    elif len(synthesized) > len(reference):
        synthesized = synthesized[:len(reference)]
    return synthesized

# Function to compute STOI
def compute_stoi_metric(reference, synthesized, sr=22050):
    return stoi(reference, synthesized, sr, extended=False)

# Function to compute PESQ
def compute_pesq_metric(reference, synthesized, sr=16000):
    # Resample if necessary
    if sr != 16000:
        reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
        synthesized = librosa.resample(synthesized, orig_sr=sr, target_sr=16000)
    return pesq(16000, reference, synthesized, 'wb')  # Wideband PESQ

# Function to normalize text
def normalize_text(text):
    # Remove unsupported characters
    text = re.sub(r'[͡]', '', text)
    return text

# Directory setup
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
data_path = "LJSpeech-1.1/metadata.csv"
reference_audio_dir = "LJSpeech-1.1/wavs"  # Directory with original wav files
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize metrics storage
metrics = []

# TTS Models to evaluate (including AWS Polly)
model_names = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/speedy-speech",
    "tts_models/en/ljspeech/fast_pitch",
    "tts_models/en/ljspeech/tacotron2-DCA",
    "tts_models/en/ljspeech/glow-tts",
    "aws_polly"
]

# Load the metadata for text inputs
metadata = pd.read_csv(data_path, sep="|", header=None, names=["id", "text", "unused"])
# load 1000 samples for fair evaluation
sample_texts = metadata.iloc[:1000]  
# Initialize Seaborn style
sns.set(style="whitegrid")

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

# Initialize a session using Amazon Polly
polly_session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Initialize the Polly client
polly_client = polly_session.client('polly')

# Iterate over models and texts
for model_name in model_names:
    print(f"Evaluating Model: {model_name}")
    if model_name != "aws_polly":
        try:
            tts = TTS(model_name=model_name, progress_bar=False).to(device)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue

    for idx, row in sample_texts.iterrows():
        text = normalize_text(row["text"])
        sample_id = row["id"]

        # Generate TTS audio
        start_time = time.time()
        if model_name == "aws_polly":
            output_path = os.path.join(
                output_dir, f"{model_name.replace('/', '_')}_sample_{sample_id}.mp3"
            )
        else:
            output_path = os.path.join(
                output_dir, f"{model_name.replace('/', '_')}_sample_{sample_id}.wav"
            )

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
                audio = AudioSegment.from_mp3(output_path)
                audio = audio.set_frame_rate(22050)
                audio = audio.set_channels(1)
                converted_output_path = os.path.splitext(output_path)[0] + ".wav"
                audio.export(converted_output_path, format="wav")
                os.remove(output_path)
                output_path = converted_output_path
            else:
                # Synthesize speech using the local TTS model
                tts.tts_to_file(text=text, file_path=output_path)
                inference_time = time.time() - start_time
        except Exception as e:
            print(f"Error generating audio for sample {sample_id} with model {model_name}: {e}")
            continue

        # Load synthesized and reference audios
        try:
            
            synthesized_audio = load_audio(output_path)
            reference_audio = load_audio(os.path.join(reference_audio_dir, f"{sample_id}.wav"))
        except Exception as e:
            print(f"Error loading audio files for sample {sample_id}: {e}")
            continue

        # Match lengths of reference and synthesized audios
        synthesized_audio = match_length(reference_audio, synthesized_audio)

        # Compute metrics
        try:
            stoi_score = compute_stoi_metric(reference_audio, synthesized_audio)
            pesq_score = compute_pesq_metric(reference_audio, synthesized_audio, sr=22050)
            mcd_score = compute_mcd(reference_audio, synthesized_audio)
            mmsd_score = compute_mmsd(reference_audio, synthesized_audio)
            wer_score = None
            cer_score = None
        except Exception as e:
            print(f"Error computing metrics for sample {sample_id} in {model_name}: {e}")
            stoi_score = pesq_score = mcd_score = mmsd_score = wer_score = cer_score = None

        # Record metrics
        metrics.append({
            "model_name": model_name,
            "sample_id": sample_id,
            "inference_time": inference_time,
            "real_time_factor": inference_time / len(reference_audio) if len(reference_audio) > 0 else None,
            "stoi_score": stoi_score,
            "pesq_score": pesq_score,
            "mcd_score": float(mcd_score) if mcd_score else None,
            "mmsd_score": float(mmsd_score) if mmsd_score else None,
            "wer_score": float(wer_score) if wer_score else None,
            "cer_score": float(cer_score) if cer_score else None
        })

# Save metrics to JSON for later analysis
with open("tts_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Load metrics for visualization
with open("tts_metrics.json", "r") as f:
    metrics = json.load(f)

df = pd.DataFrame(metrics)

# Handle missing values by dropping samples with any missing metrics
df.dropna(subset=['inference_time', 'real_time_factor', 'stoi_score', 'pesq_score', 'mcd_score', 'mmsd_score'], inplace=True)

# Compute average metrics for each model
avg_metrics = df.groupby('model_name').mean(numeric_only=True).reset_index()

# Plotting Inference Time
plt.figure(figsize=(12, 6))
sns.barplot(x='model_name', y='inference_time', data=avg_metrics, hue='model_name', dodge=False)
plt.title('Average Inference Time per TTS Model')
plt.ylabel('Inference Time (seconds)')
plt.xlabel('TTS Model')
plt.xticks(rotation=45)
plt.legend().remove()
plt.tight_layout()
plt.savefig('average_inference_time.png', dpi=300)
plt.close()

# Plotting Real-Time Factor
plt.figure(figsize=(12, 6))
sns.barplot(x='model_name', y='real_time_factor', data=avg_metrics, hue='model_name', dodge=False)
plt.title('Average Real-Time Factor per TTS Model')
plt.ylabel('Real-Time Factor')
plt.xlabel('TTS Model')
plt.xticks(rotation=45)
plt.legend().remove()
plt.tight_layout()
plt.savefig('average_real_time_factor.png', dpi=300)
plt.close()

# Plotting Quality Metrics
quality_metrics = ['stoi_score', 'pesq_score', 'mcd_score', 'mmsd_score']

for metric in quality_metrics:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_name', y=metric, data=avg_metrics, hue='model_name', dodge=False)
    plt.title(f'Average {metric.upper()} per TTS Model')
    plt.ylabel(metric.upper())
    plt.xlabel('TTS Model')
    plt.xticks(rotation=45)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(f'average_{metric}.png', dpi=300)
    plt.close()

# Correlation Heatmap
corr = avg_metrics[quality_metrics + ['inference_time', 'real_time_factor']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Matrix of TTS Metrics')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.close()

# Box Plots for Distribution Analysis
for metric in ['inference_time', 'real_time_factor'] + quality_metrics:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model_name', y=metric, data=df, hue='model_name', dodge=False)
    plt.title(f'Distribution of {metric.upper()} across TTS Models')
    plt.ylabel(metric.upper())
    plt.xlabel('TTS Model')
    plt.xticks(rotation=45)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(f'boxplot_{metric}.png', dpi=300)
    plt.close()
