import os
import shutil
import time
from collections import Counter

import sounddevice as sd
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
import numpy as np
import librosa
import joblib
import tensorflow as tf


def record_audio(duration=5, sample_rate=16000, channels=1):
    """
    Records audio from the microphone.

    Parameters:
    - duration (int): Duration of recording in seconds.
    - sample_rate (int): Sampling rate in Hz.
    - channels (int): Number of audio channels.

    Returns:
    - audio_data (np.ndarray): Recorded audio data.
    """
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()
    print("Recording complete.")
    return audio_data


def save_audio(file_path, audio_data, sample_rate):
    """
    Saves recorded audio to a .wav file.

    Parameters:
    - file_path (str): Path where the audio file will be saved.
    - audio_data (np.ndarray): Audio data to save.
    - sample_rate (int): Sampling rate in Hz.
    """
    write(file_path, sample_rate, audio_data)
    print(f"Audio saved as {file_path}")


def extract_mfcc_feature(file_path, n_mfcc=13, max_frames=300):
    """
    Extracts MFCC features from an audio file.

    Parameters:
    - file_path (str): Path to the audio file.
    - n_mfcc (int): Number of MFCCs to extract.
    - max_frames (int): Number of frames to pad/truncate the MFCC feature.

    Returns:
    - mfccs (np.ndarray): Extracted and normalized MFCC features with shape (n_mfcc, max_frames).
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = librosa.util.normalize(mfccs)
        mfccs = librosa.util.fix_length(mfccs, size=max_frames, axis=1)
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCC features from {file_path}: {e}")
        return None


def load_model_and_encoder(model_path, encoder_path, hdf5=False):
    """
    Loads the trained CNN model and LabelEncoder.

    Parameters:
    - model_path (str): Path to the saved CNN model (.h5 file or SavedModel directory).
    - encoder_path (str): Path to the saved LabelEncoder (.joblib file).
    - hdf5 (bool): True if the model is saved in HDF5 format, False if SavedModel.

    Returns:
    - model (tf.keras.Model): Loaded Keras model.
    - encoder (LabelEncoder): Loaded LabelEncoder.
    """
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    print(f"Model successfully loaded from {model_path}")

    # Load the LabelEncoder
    encoder = joblib.load(encoder_path)
    print(f"LabelEncoder successfully loaded from {encoder_path}")

    return model, encoder


def classify_speaker(model, encoder, audio_path, n_mfcc=13, max_frames=300):
    """
    Classifies the speaker of a given audio sample.

    Parameters:
    - model (tf.keras.Model): Trained CNN model.
    - encoder (LabelEncoder): Fitted LabelEncoder.
    - audio_path (str): Path to the new audio sample (.wav file).
    - n_mfcc (int): Number of MFCCs to extract.
    - max_frames (int): Number of frames to pad/truncate the MFCC feature.

    Returns:
    - predicted_label (str): Predicted speaker label.
    """
    # Extract features
    feature = extract_mfcc_feature(audio_path, n_mfcc, max_frames)

    if feature is not None:
        # Prepare the input for the model
        X_new = np.array(feature)  # Shape: (n_mfcc, max_frames)
        X_new = X_new[np.newaxis, ..., np.newaxis]  # Shape: (1, n_mfcc, max_frames, 1)

        # Make prediction
        prediction = model.predict(X_new)

        # Get the predicted class index
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Decode the class index to the original label
        predicted_label = encoder.inverse_transform([predicted_class_index])[0]

        return predicted_label
    else:
        print("Feature extraction failed for the new sample.")
        return None


def plot_mfcc(audio_path, sr=16000, n_mfcc=13, max_frames=300):
    """
    Plots the MFCC graph for the given audio data.

    Parameters:
    - y (np.ndarray): Audio time series.
    - sr (int): Sampling rate.
    - n_mfcc (int): Number of MFCCs to extract.
    - max_frames (int): Number of frames to pad/truncate the MFCC feature.
    """
    mfccs = extract_mfcc_feature(audio_path, n_mfcc=n_mfcc, max_frames=max_frames)
    if mfccs is not None:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='viridis', hop_length=512)
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()
    else:
        print("Failed to plot MFCC.")


def save_predictions(predictions, log_file='prediction_log.txt'):
    """
    Saves the predictions to a log file.

    Parameters:
    - predictions (list): List of predicted speaker labels.
    - log_file (str): Path to the log file.
    """
    try:
        with open(log_file, 'a') as f:
            for i, label in enumerate(predictions, 1):
                f.write(f"Chunk {i}: {label}\n")
            f.write("\n")
        print(f"Predictions saved to {log_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")


def main():
    """
    Main function to record audio, save it, plot MFCC, and classify the speaker.
    Adds a 2-second delay before recording and records three separate audios.
    """
    # Define paths
    model_path = 'model/final_cnn_model.h5'  # Replace with your actual path
    encoder_path = 'model/label_encoder.joblib'  # Replace with your actual path

    # Load the model and encoder
    model, encoder = load_model_and_encoder(model_path, encoder_path, hdf5=True)
    if model is None or encoder is None:
        print("Failed to load model or encoder. Exiting.")
        return

    # Add a 2-second delay before starting the first recording
    print("Get ready to speak. Recording will start in 2 seconds...")
    time.sleep(2)

    # Number of recordings
    num_recordings = 3

    # List to store predictions
    predictions = []

    for i in range(1, num_recordings + 1):
        print(f"\n--- Recording {i} ---")
        duration = 10  # Duration in seconds
        sample_rate = 16000  # Sampling rate in Hz
        channels = 1  # Mono recording

        # Record audio
        audio_data = record_audio(duration=duration, sample_rate=sample_rate, channels=channels)
        if audio_data is None:
            print(f"Recording {i} failed. Skipping to next.")
            continue

        # Save the recorded audio with a unique filename
        audio_save_path = f'recorded_sample_{i}.wav'
        save_audio(audio_save_path, audio_data, sample_rate)

        # Plot the MFCC for the recorded audio
        print(f"Plotting MFCC for Recording {i}...")
        plot_mfcc(audio_save_path, sr=sample_rate)

        # Classify the recorded audio
        print(f"Classifying Recording {i}...")
        predicted_speaker = classify_speaker(model, encoder, audio_save_path)

        if predicted_speaker:
            print(f"Recording {i}: The predicted speaker is: {predicted_speaker}")
            predictions.append(predicted_speaker)
        else:
            print(f"Recording {i}: Failed to classify the speaker.")

    # Determine the most frequent prediction
    if predictions:
        counter = Counter(predictions)
        most_common_speaker, count = counter.most_common(1)[0]
        print(f"\nFinal Prediction: The most frequent predicted speaker is {most_common_speaker} (Predicted {count} times).")
    else:
        print("\nNo valid predictions were made from the recordings.")

    # Save all predictions to a log file
    save_predictions(predictions)


if __name__ == "__main__":
    main()
