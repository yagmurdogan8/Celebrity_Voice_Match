import os

import pandas as pd
import numpy as np
import librosa
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from feature_extraction import extract_features


def main(audio_folder: str, feature_set = None, pca: bool = True, scaling: bool = True):
    # Load metadata
    meta_path = os.path.join(audio_folder, "meta.csv")
    meta = pd.read_csv(meta_path)

    # Extract features and prepare the dataset
    features, labels = [], []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing files"):
        file_path = os.path.join(audio_folder, row['file'])
        speaker = row['speaker']

        # Extract features
        feature = extract_features(file_path, feature_set=feature_set)
        if feature is not None:
            features.append(feature)
            labels.append(speaker)

    if not features:
        print("No features were extracted. Exiting.")
        return

    # Convert features and labels to arrays
    X = np.array(features)
    y = np.array(labels)

    # Feature Scaling
    if scaling:
        X = StandardScaler().fit_transform(X)

    # Dimensionality Reduction
    if pca:
        X = PCA(n_components=50).fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    audio_folder_path = "data"

    # Define the features you want to extract
    selected_features = [
        'delta_mfccs',
        'spectral',
        'prosodic',
        # 'lpc',
        # 'chroma',
        # 'mel_spectrogram',
        # 'zcr',
        # 'tempo',
        # 'harmonic_percussive'
    ]

    main(audio_folder_path, feature_set=selected_features, pca=True, scaling=True)
