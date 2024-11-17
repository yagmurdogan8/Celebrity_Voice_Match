import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import sounddevice as sd
import wavio

# Set parameters for recording
DURATION = 15  # seconds
SAMPLE_RATE = 22050  # Hz

def record_audio(filename="recorded_audio.wav", duration=DURATION, sample_rate=SAMPLE_RATE):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    wavio.write(filename, recording, sample_rate, sampwidth=2)
    return filename

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def prepare_dataset(dataset_path="dataset", meta_file="meta.csv"):
    meta_data = pd.read_csv(meta_file)
    # Filter only bona-fide samples
    bona_fide_data = meta_data[meta_data['label'] == 'bona-fide']
    features = []
    labels = []
    for index, row in bona_fide_data.iterrows():
        audio_path = os.path.join(dataset_path, row['file'])
        if os.path.exists(audio_path):
            mfcc_features = extract_features(audio_path)
            features.append(mfcc_features)
            labels.append(row['speaker'])
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def match_celebrity(features, labels, user_audio_features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    user_audio_scaled = scaler.transform([user_audio_features])
    
    # Using Nearest Neighbors for matching
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(features_scaled)
    distance, index = nn.kneighbors(user_audio_scaled)
    return labels[index[0][0]]

def main():
    dataset_path = "release_in_the_wild/release_in_the_wild"
    meta_file = "meta.csv"
    
    # Step 1: Prepare the dataset
    print("Preparing the dataset...")
    features, labels = prepare_dataset(dataset_path, meta_file)
    print("Dataset ready!")
    
    # Step 2: Record user audio
    recorded_file = record_audio()
    
    # Step 3: Extract features from the recorded audio
    user_audio_features = extract_features(recorded_file)
    
    # Step 4: Match with a celebrity
    print("Matching with a celebrity...")
    celebrity = match_celebrity(features, labels, user_audio_features)
    print(f"You sound like: {celebrity}")

if __name__ == "__main__":
    main()
