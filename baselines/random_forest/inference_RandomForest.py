import time
import numpy as np
import librosa
import joblib
from scipy.io.wavfile import write
from collections import Counter
import sounddevice as sd
import joblib

# Dictionary to store traits for each celebrity
celebrity_traits = {
    "Gilbert Gottfried": {
        "pitch": "High-pitched, often strident and nasal, with frequent spikes in volume.",
        "speech_rhythm": "Rapid speech with little pause, conveying an anxious or excitable demeanor.",
        "tone": "Shrill, abrasive, and exaggerated.",
        "energy": "High-energy, fast-paced delivery, often delivering punchlines quickly.",
        "timing": "Minimal pauses, giving a sense of urgency or constant agitation."
    },
    "Barack Obama": {
        "pitch": "Deep and controlled, with occasional rises for emphasis.",
        "speech_rhythm": "Steady, methodical, and calm, with deliberate pacing.",
        "tone": "Warm, authoritative, and composed.",
        "energy": "Moderate energy, using pauses effectively to emphasize key points.",
        "timing": "Strategic pauses to highlight important messages, often with measured cadences."
    },
    "Alan Watts": {
        "pitch": "Smooth and calm, with a moderate pitch that conveys introspection.",
        "speech_rhythm": "Slower pace, often pausing for reflection.",
        "tone": "Gentle, contemplative, and soothing.",
        "energy": "Low energy, meditative, with a focus on calm and understanding.",
        "timing": "Very deliberate pauses to allow the listener to reflect on his thoughts."
    },
    "Bernie Sanders": {
        "pitch": "Mid to high pitch, often forceful in delivery.",
        "speech_rhythm": "Rapid and sometimes staccato, particularly when making a passionate point.",
        "tone": "Assertive, direct, and passionate, with an element of urgency.",
        "energy": "High energy, conveying a sense of urgency and activism.",
        "timing": "Quick pauses, usually at the end of sentences to emphasize key political points."
    },
    "Donald Trump": {
        "pitch": "Mid-range, with occasional emphasis on high-pitched exclamations.",
        "speech_rhythm": "Erratic and sometimes scattered, with bursts of intensity.",
        "tone": "Boisterous, self-assured, and occasionally combative.",
        "energy": "High energy, often using exaggerated expressions to emphasize points.",
        "timing": "Pauses to make dramatic effect, especially during claims or points of contention."
    },
    "Arnold Schwarzenegger": {
        "pitch": "Deep and throaty, with distinct accents and occasional rising pitch.",
        "speech_rhythm": "Slower, with deliberate emphasis on key words.",
        "tone": "Commanding, with a strong and distinctive accent.",
        "energy": "Moderate to high energy, focusing on authority and strength.",
        "timing": "Clear, calculated pauses to emphasize dramatic delivery."
    },
    "Bill Burr": {
        "pitch": "Mid-range with occasional sharp rises for comedic effect.",
        "speech_rhythm": "Rapid-paced, sometimes erratic with humorously timed pauses.",
        "tone": "Sarcastic, blunt, and confrontational.",
        "energy": "High energy, with bursts of intensity when delivering punchlines or jokes.",
        "timing": "Pauses for comedic effect, especially in the middle of sentences for emphasis."
    },
    "Queen Elizabeth II": {
        "pitch": "Soft and controlled, with minimal fluctuation in pitch.",
        "speech_rhythm": "Steady, formal, and deliberate.",
        "tone": "Authoritative, elegant, and regal.",
        "energy": "Low energy, maintaining a composed and dignified presence.",
        "timing": "Controlled pauses, often for emphasis or for the gravitas of the moment."
    },
    "Christopher Hitchens": {
        "pitch": "Varied pitch, often rising in moments of strong rhetoric.",
        "speech_rhythm": "Moderate speed, with occasional rapid delivery in moments of emphasis.",
        "tone": "Intellectual, sharp, and occasionally sardonic.",
        "energy": "Moderate to high energy, especially when making a rhetorical argument.",
        "timing": "Deliberate pauses to emphasize key points or to allow a counterpoint to sink in."
    },
    "Winston Churchill": {
        "pitch": "Deep, resonant, with powerful fluctuations for effect.",
        "speech_rhythm": "Steady and deliberate, often with powerful cadences.",
        "tone": "Authoritative, persuasive, and commanding.",
        "energy": "High energy, focused on delivering powerful, motivational messages.",
        "timing": "Long pauses to give weight to critical points."
    }
}


# Function to generate explanations
def generate_explanation(predicted_speaker):
    traits = celebrity_traits.get(predicted_speaker, None)
    if traits:
        explanation = (
            f"Recording classified as {predicted_speaker}. Here are the vocal traits:\n"
            f"Pitch: {traits['pitch']}\n"
            f"Speech Rhythm: {traits['speech_rhythm']}\n"
            f"Tone: {traits['tone']}\n"
            f"Energy: {traits['energy']}\n"
            f"Timing: {traits['timing']}\n"
        )
        return explanation
    return "No traits available for this speaker."


def record_audio(duration=5, sample_rate=16000, channels=1):
    """
    Records audio from the microphone.
    """
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()
    print("Recording complete.")
    return audio_data


def save_audio(file_path, audio_data, sample_rate):
    """
    Saves recorded audio to a .wav file.
    """
    write(file_path, sample_rate, audio_data)
    print(f"Audio saved as {file_path}")


def extract_features(audio_file, sr=16000, n_mfcc=13):
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=sr)
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Return the mean of the MFCCs over time
        return np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


def load_model(model_path):
    """
    Loads the trained Random Forest model.
    """
    model = joblib.load(model_path)
    print(f"Model successfully loaded from {model_path}")
    return model


def classify_speaker(model, audio_path):
    """
    Classifies the speaker of a given audio sample and provides an explanation.
    """
    feature = extract_features(audio_path, sr=16000, n_mfcc=13)

    if feature is not None:
        X_new = feature.reshape(1, -1)
        prediction = model.predict(X_new)
        predicted_label = prediction[0]

        # Generate explanation
        explanation = generate_explanation(predicted_label)

        return predicted_label, explanation
    else:
        print("Feature extraction failed for the new sample.")
        return None, None


def preprocess_audio(file_path):
    """
    Preprocess audio file with preemphasis.
    """
    audio, sr = librosa.load(file_path, sr=16000)
    audio = librosa.effects.preemphasis(audio)
    return audio, sr


def save_predictions(predictions, log_file='prediction_log.txt'):
    """
    Saves predictions to a log file.
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
    Main function with two modes:
    - Part 1: Record audio live, classify and show explanations.
    - Part 2: Process pre-existing audio files and show explanations.
    """
    # Define paths
    model_path = 'models/random_forest_A.joblib'

    # Load the model
    model = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Choose mode
    part = int(input("Run part 1 (live recording) or part 2 (pre-existing files)? "))

    num_recordings = 3
    predictions = []

    if part == 1:
        # Part 1: Live Recording
        print("Get ready to speak. Recording will start in 2 seconds...")
        time.sleep(2)
        for i in range(num_recordings):
            print(f"Recording {i+1}...")
            audio_data = record_audio(duration=10, sample_rate=16000, channels=1)
            audio_path = f"recorded_sample_{i}.wav"
            save_audio(audio_path, audio_data, 16000)

            # Preprocess and classify
            predicted_speaker, explanation = classify_speaker(model, audio_path)
            if predicted_speaker:
                print(f"Prediction: {predicted_speaker}")
                print(f"Explanation:\n{explanation}")
                predictions.append(predicted_speaker)
    else:
        # Part 2: Pre-existing audio files
        print("\nProcessing pre-existing audio files...")
        for i in range(3):  # Process three files (e.g., Trump_0.wav, Trump_1.wav, Trump_2.wav)
            audio_save_path = f'Test Audios/Trump_{i}.wav'
            print(f"\n--- Processing file: {audio_save_path} ---")

            # Preprocess the audio file
            preprocessed_audio, sr = preprocess_audio(audio_save_path)
            if preprocessed_audio is not None:
                preprocessed_audio_save_path = f'pre_existing_sample_{i + 1}_preprocessed.wav'
                save_audio(preprocessed_audio_save_path, preprocessed_audio, sample_rate=16000)

                # Classify the speaker
                predicted_speaker, explanation = classify_speaker(model, preprocessed_audio_save_path)

                if predicted_speaker:
                    print(f"File {i + 1}: The predicted speaker is {predicted_speaker}")
                    print(f"Explanation: {explanation}")
                    predictions.append(predicted_speaker)
                else:
                    print(f"File {i + 1}: Failed to classify the speaker.")
            else:
                print(f"Preprocessing failed for file {audio_save_path}. Skipping.")

    if predictions:
        # Determine majority vote
        counter = Counter(predictions)
        most_common, count = counter.most_common(1)[0]
        print(f"\nMost Common Prediction: {most_common} predicted {count} times.")

    save_predictions(predictions)


if __name__ == "__main__":
    main()