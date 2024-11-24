import os

import librosa

# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow logs (INFO and WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
import joblib
from tqdm import tqdm

# To use feature_set import it from feature_extraction.py instead of mel_spectrogram_extraction.py
from mel_spectrogram_extraction import extract_features


# Audio files
audio_folder = "data"
meta_path = "data/meta.csv"

# Features configuration
# feature_set = ['mel_spectrogram', 'zcr', 'harmonic_percussive']
sr = 16000
n_mels = 128
max_length = 300

# Training configuration
batch_size = 32
epochs = 50

# Load audio files metadata
meta = pd.read_csv(meta_path)

# Feature Extraction
features, labels = [], []
for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing files"):
    file_path = os.path.join(audio_folder, row['file'])
    speaker = row['speaker']

    # Uncomment to use feature_set
    # feature = extract_features(file_path, sr=sr, feature_set=feature_set)
    # if feature is not None:
    #     features.append(feature)
    #     labels.append(speaker)

    mel_spectrogram = extract_features(file_path, sr=sr, n_mels=n_mels, max_length=max_length)
    if mel_spectrogram is not None:
        features.append(mel_spectrogram)
        labels.append(speaker)
    else:
        print(f"Skipping file {file_path} due to extraction failure.")

# Convert to NumPy arrays
X = np.array(features)  # Shape: (samples, combined_feature_length)
y = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape data
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], 1))
input_shape = X_train.shape[1:]  # (1, feature_length, 1)
print(f"Reshaped X_train: {X_train.shape}")
print(f"Reshaped X_test: {X_test.shape}")

model = Sequential([
    tf.keras.Input(shape=input_shape),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((1, 2)),
    BatchNormalization(),

    Conv2D(64, (1, 3), activation='relu', padding='same'),
    MaxPooling2D((1, 2)),
    BatchNormalization(),

    Conv2D(128, (1, 3), activation='relu', padding='same'),
    MaxPooling2D((1, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
model.fit(X_train, y_train, 
          epochs=epochs, 
          batch_size=batch_size, 
          validation_split=0.2, 
          callbacks=[lr_scheduler])


# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Detailed Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_,
            yticklabels=encoder.classes_, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Save Model
os.makedirs("model", exist_ok=True)
model.save("model/celebrity_voice_cnn_model.h5")
joblib.dump(encoder, "model/label_encoder.pkl")

print("Model and encoder saved successfully.")

# 6. REAL-TIME PREDICTION (PREDICT MOST SIMILAR VOICE CELEBRITY)
# def record_audio(filename, duration=15, sr=16000):
#     print(f"Recording for {duration} seconds...")
#     audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
#     sd.wait()  # Wait until the recording is finished
#     wav.write(filename, sr, audio)
#     print(f"Recording saved to {filename}")
#
# user_audio_path = "/Users/giuliarivetti/Desktop/Repos/Nuovo memo 21.wav"

# Record user's voice
#record_audio(user_audio_path, duration=15, sr=sr)

# Process user audio
# user_feature = extract_features(user_audio_path)
# if user_feature is not None:
#     user_feature = user_feature[..., np.newaxis]  # Add channel dimension
#     user_feature = np.expand_dims(user_feature, axis=0)  # Add batch dimension
#
#     # Predict the closest celebrity
#     prediction = model.predict(user_feature)
#     predicted_label = np.argmax(prediction)
#     celebrity = encoder.inverse_transform([predicted_label])[0]
#     print(f"The voice is closest to: {celebrity}")
# else:
#     print("Could not extract features from the recorded audio.")
