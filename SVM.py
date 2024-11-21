import pandas as pd
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

# Step 1: Record Audio from the Microphone
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    # Record for the given duration (in seconds)
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio_data, sample_rate

# Step 2: Save Recorded Audio to a File
def save_audio(file_path, audio_data, sample_rate):
    write(file_path, sample_rate, audio_data)  # Save the audio to a .wav file
    print(f"Audio saved as {file_path}")

# Step 3: Extract Features from Audio
def extract_features(file_path, n_mfcc=13):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)  # Take mean of MFCCs across time
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Step 4: Prepare Dataset
def prepare_dataset(dataset_folder, csv_path):
    data = pd.read_csv(csv_path, header=None, names=["file", "celebrity", "country"])
    features = []
    labels = []
    
    for _, row in data.iterrows():
        file_path = os.path.join(dataset_folder, row["file"])
        feature = extract_features(file_path)
        if feature is not None:
            features.append(feature)
            labels.append(row["celebrity"])
    return np.array(features), np.array(labels)

# Step 5: Train SVM Model
def train_svm(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm_model = SVC(kernel='rbf', probability=True)  # RBF kernel for better performance
    svm_model.fit(X_scaled, y)
    return svm_model, scaler

# Step 6: Predict Celebrity from Recorded Audio
def predict_celebrity(svm_model, scaler, recording_path):
    feature = extract_features(recording_path)
    if feature is not None:
        feature_scaled = scaler.transform([feature])
        prediction = svm_model.predict(feature_scaled)
        probability = svm_model.predict_proba(feature_scaled)
        return prediction[0], np.max(probability)
    else:
        return None, None

# Main Workflow
if __name__ == "__main__":
    dataset_folder = "dataset"  # Folder where your .wav files are located
    csv_path = "final_updated_meta.csv"  # Path to your CSV file
    recording_path = "recorded_voice.wav"  # Path to save the recorded audio

    # Step 1: Load dataset and train model
    print("Preparing dataset...")
    X, y = prepare_dataset(dataset_folder, csv_path)
    print("Training SVM model...")
    svm_model, scaler = train_svm(X, y)

    # Step 2: Record audio and save it
    audio_data, sample_rate = record_audio(duration=5)  # Record for 5 seconds
    save_audio(recording_path, audio_data, sample_rate)

    # Step 3: Predict celebrity based on recorded audio
    print("Predicting your recording...")
    celebrity, confidence = predict_celebrity(svm_model, scaler, recording_path)
    if celebrity:
        print(f"You matched with {celebrity} with confidence {confidence:.2f}")
    else:
        print("Could not process your recording.")
