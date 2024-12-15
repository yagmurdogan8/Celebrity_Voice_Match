import librosa
import numpy as np


def extract_mel_spectrogram(y, sr=16000, n_mels=128, max_length=300):
    """
    Extracts the Mel-Spectrogram from an audio signal.

    Parameters:
    - y (np.ndarray): Audio time series.
    - sr (int): Sampling rate.
    - n_mels (int): Number of Mel bands to generate.
    - max_length (int): Maximum number of time frames.

    Returns:
    - np.ndarray: Normalized and padded Mel-Spectrogram.
    """
    try:
        # Compute Mel-Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalize
        mel_spectrogram = librosa.util.normalize(mel_spectrogram)

        # Pad or truncate to ensure consistent size
        if mel_spectrogram.shape[1] < max_length:
            mel_spectrogram = np.pad(mel_spectrogram,
                                     pad_width=((0, 0), (0, max_length - mel_spectrogram.shape[1])),
                                     mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :max_length]

        return mel_spectrogram
    except Exception as e:
        print(f"Error extracting Mel-Spectrogram: {e}")
        return None


def extract_features(audio_file, sr=16000, n_mels=128, max_length=300):
    """
    Wrapper function to load an audio file and extract its Mel-Spectrogram.

    Parameters:
    - audio_file (str): Path to the audio file.
    - sr (int): Sampling rate.
    - n_mels (int): Number of Mel bands to generate.
    - max_length (int): Maximum number of time frames.

    Returns:
    - np.ndarray or None: Extracted Mel-Spectrogram or None if extraction fails.
    """
    try:
        y, sr = librosa.load(audio_file, sr=sr)
        mel_spectrogram = extract_mel_spectrogram(y, sr=sr, n_mels=n_mels, max_length=max_length)
        return mel_spectrogram
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None