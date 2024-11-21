import librosa
import numpy as np


def extract_delta_mfccs(y, sr, n_mfcc=13):
    """
    Delta MFCCs: These capture the temporal dynamics of the MFCC features by computing the first-order derivatives.
    Delta-Delta MFCCs: These represent the second-order derivatives, providing information about the acceleration of the MFCC features.
    """
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        combined = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        return np.mean(combined, axis=1)
    except Exception as e:
        print(f"Error extracting Delta MFCCs: {e}")
        return None


def extract_spectral_features(y, sr):
    """
    Spectral Centroid: Indicates where the "center of mass" of the spectrum is located, perceived as the brightness of the sound.
    Spectral Bandwidth: Measures the spread of the spectrum around the centroid.
    Spectral Contrast: Captures the difference between peaks and valleys in the spectrum.
    Spectral Rolloff: The frequency below which a certain percentage of the total spectral energy lies.
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = np.hstack([
        np.mean(spectral_centroid, axis=1),
        np.mean(spectral_bandwidth, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(spectral_rolloff, axis=1)
    ])
    return features


def extract_prosodic_features(y, sr):
    """
    Pitch (Fundamental Frequency): The perceived frequency of sound, crucial for intonation.
    Energy (Loudness): Reflects the amplitude variations, which can vary between speakers.

    Speech Rhythm and Intonation: Prosodic features capture the rhythm, stress, and intonation patterns unique to each speaker.
    """
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch = pitch[pitch > 0]
    mean_pitch = np.mean(pitch) if pitch.size > 0 else 0

    energy = np.sum(y ** 2) / len(y)

    return np.array([mean_pitch, energy])


def extract_lpc_features(y, sr, order=12):
    """
    LPC: Models the vocal tract by estimating the coefficients that minimize the error between the actual and predicted speech signal.

    Vocal Tract Characteristics: LPC captures the characteristics of the speaker's vocal tract, which are highly speaker-specific.
    """
    try:
        # Compute LPC coefficients
        lpc_coeffs = librosa.lpc(y, order=order)
        return lpc_coeffs
    except Exception as e:
        print(f"LPC extraction error: {e}")
        return np.zeros(order)


def extract_chroma_features(y, sr):
    """
    Chroma Features: Represent the 12 different pitch classes (e.g., C, C#, D, ...) in music, extended to speech for capturing harmonic content.

    Harmonic Content: While more common in music analysis, chroma features can capture harmonic patterns in speech that might be distinctive for different speakers.
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)


def extract_mel_spectrogram_features(y, sr, n_mels=128):
    """
    Mel-Spectrogram: A spectrogram where the frequency axis is converted to the Mel scale, which aligns more closely with human auditory perception.
    Log-Mel Spectrogram: The logarithm of the Mel-spectrogram, which can stabilize variance and make the features more Gaussian-like.

    Perceptually Relevant Features: Mel-spectrograms emphasize frequencies that are more important to human perception, potentially capturing more relevant information for speaker identification.
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return np.mean(log_mel_spectrogram, axis=1)


def extract_zcr(y, sr):
    """
    ZCR: The rate at which the signal changes sign, i.e., from positive to negative or vice versa.

    Signal Complexity: ZCR can provide information about the noisiness or tonal properties of the speech, which can vary between speakers.
    :param y:
    :return:
    """
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.mean(zcr)


def extract_tempo_features(y, sr):
    """
    Tempo: The speed or pace of the speaker.
    Onset Rate: Frequency of new sounds or phonemes starting.

    Speech Rhythm: Temporal dynamics like speaking rate and rhythm can be distinguishing factors between speakers.
    """
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return np.array([tempo])


def extract_harmonic_percussive_features(y, sr):
    """
    Harmonic: Steady-state components representing the pitch.
    Percussive: Transient components representing the rhythm or percussive sounds.

    Separation of Elements: Isolating harmonic and percussive elements can help in capturing different aspects of the speech signal.
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(y_harmonic**2) / len(y_harmonic)
    percussive_energy = np.sum(y_percussive**2) / len(y_percussive)
    return np.array([harmonic_energy, percussive_energy])


def extract_features(audio_file,
                    sr=16000,
                    feature_set=None,
                    n_mfcc=13,
                    n_mels=128,
                    lpc_order=12):
    """
    Extracts a combination of audio features based on the specified feature_set.

    Parameters:
    - audio_file (str): Path to the audio file.
    - sr (int): Sampling rate.
    - feature_set (list or None): List of feature names to extract. If None, extract all.
    - n_mfcc (int): Number of MFCCs to extract.
    - n_mels (int): Number of Mel bands to generate.
    - lpc_order (int): Order of the LPC.

    Returns:
    - np.ndarray or None: Combined feature vector or None if extraction fails.
    """
    # Define all possible features
    available_features = {
        'delta_mfccs': lambda y, sr: extract_delta_mfccs(y, sr, n_mfcc),
        'spectral': lambda y, sr: extract_spectral_features(y, sr),
        'prosodic': lambda y, sr: extract_prosodic_features(y, sr),
        'lpc': lambda y, sr: extract_lpc_features(y, sr, order=lpc_order),
        'chroma': lambda y, sr: extract_chroma_features(y, sr),
        'mel_spectrogram': lambda y, sr: extract_mel_spectrogram_features(y, sr, n_mels=n_mels),
        'zcr': lambda y, sr: extract_zcr(y, sr),
        'tempo': lambda y, sr: extract_tempo_features(y, sr),
        'harmonic_percussive': lambda y, sr: extract_harmonic_percussive_features(y, sr)
    }

    # If feature_set is not specified, use all features
    if feature_set is None:
        feature_set = list(available_features.keys())

    # Initialize a list to hold all features
    feature_vector = []

    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=sr)

        # Iterate over the selected features and extract them
        for feature_name in feature_set:
            extractor = available_features.get(feature_name)
            if extractor:
                feature = extractor(y, sr)
                if feature is not None:
                    feature_vector.append(feature)
                else:
                    print(f"Feature '{feature_name}' extraction returned None for {audio_file}.")
            else:
                print(f"Feature '{feature_name}' is not available.")

        if not feature_vector:
            print(f"No features extracted for {audio_file}.")
            return None

        # Concatenate all features into a single vector
        combined_features = np.hstack(feature_vector)
        return combined_features

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

