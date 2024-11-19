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
import joblib

# --- Parameters ---
audio_folder = "/Users/giuliarivetti/Desktop/Repos/dataset"  # Update with your dataset folder
meta_path = "/Users/giuliarivetti/Desktop/Repos/final_updated_meta.csv"  # Metadata file path
n_mfcc = 13
max_frames = 300  # Maximum number of frames to pad/truncate to
sr = 16000  # Sampling rate for audio
batch_size = 32
epochs = 20

# 1. TRASNFORM AUDIO FEATURES FOR CNN INPUT
# CNNs work best with 2D data like images, so your audio features (MFCCs) should be treated as 
# 2D "images" or spectrogram-like representations.
def extract_features(audio_file, sr=16000, n_mfcc=13):
    """ This function extracts 2D MFCC features and returns a 2D array for each audio file."""
    try:
        y, sr = librosa.load(audio_file, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = librosa.util.normalize(mfccs)
        mfccs = librosa.util.fix_length(mfccs, size=max_frames, axis=1)  # Pad/Truncate
        return mfccs
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# --- Load Metadata ---
meta = pd.read_csv(meta_path)

# --- Feature Extraction ---
features, labels = [], []
for _, row in meta.iterrows():
    file_path = os.path.join(audio_folder, row['file'])
    speaker = row['speaker']
    feature = extract_features(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(speaker)

# 2. PREPARE DATA FOR CNN INPUT
# Convert Data to 4D Tensor: CNNs expect input with dimensions (samples, height, width, channels)
X = np.array(features)  # Shape: (samples, n_mfcc, max_frames)
X = X[..., np.newaxis]  # Add channel dimension: (samples, n_mfcc, max_frames, 1)
y = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. BUILD THE CNN MODEL
# The CNN model takes the 2D MFCC matrix as input
input_shape = (n_mfcc, max_frames, 1)
print(f"Feature shape: {X.shape}")  # Expected: (samples, n_mfcc, max_frames, 1)

model = Sequential([
    tf.keras.Input(shape=(n_mfcc, max_frames, 1)),  # Define input explicitly

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Display model architecture
model.summary()

# --- Learning Rate Scheduler ---
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# 4. TRAIN THE MODEL
model.fit(X_train, y_train, 
          epochs=epochs, 
          batch_size=batch_size, 
          validation_split=0.2, 
          callbacks=[lr_scheduler])

# 5. EVALUATE THE MODEL
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the model and the encoder for the second part, so tha you don't have to re-run again everything
model.save("celebrity_voice_cnn_model.h5")
joblib.dump(encoder, "label_encoder.pkl")

# 6. REAL-TIME PREDICTION (PREDICT MOST SIMILAR VOICE CELEBRITY)
def record_audio(filename, duration=15, sr=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    wav.write(filename, sr, audio)
    print(f"Recording saved to {filename}")

user_audio_path = "/Users/giuliarivetti/Desktop/Repos/Nuovo memo 21.wav"

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
