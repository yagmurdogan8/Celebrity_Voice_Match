import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
import sounddevice as sd
import scipy.io.wavfile as wav
from tensorflow.keras.models import load_model
import joblib


max_frames = 300  # Maximum number of frames to pad/truncate to
encoder = joblib.load("label_encoder.pkl")
# Load the saved model
model = load_model("celebrity_voice_cnn_model.h5")
# To avoid Keras warning
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Helper Functions ---
def extract_features(audio_file, sr=16000, n_mfcc=13):
    try:
        y, sr = librosa.load(audio_file, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = librosa.util.normalize(mfccs)
        mfccs = librosa.util.fix_length(mfccs, size=max_frames, axis=1)  # Pad/Truncate
        return mfccs
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

user_audio_path = "/Users/giuliarivetti/Desktop/Repos/Bernie Sanders.wav"

# Record user's voice
#record_audio(user_audio_path, duration=15, sr=sr)

# Process user audio
user_feature = extract_features(user_audio_path)
if user_feature is not None:
    user_feature = user_feature[..., np.newaxis]  # Add channel dimension
    user_feature = np.expand_dims(user_feature, axis=0)  # Add batch dimension

    # Predict the closest celebrity
    prediction = model.predict(user_feature)
    predicted_label = np.argmax(prediction)
    celebrity = encoder.inverse_transform([predicted_label])[0]
    print(f"The voice is closest to: {celebrity}")
else:
    print("Could not extract features from the recorded audio.")