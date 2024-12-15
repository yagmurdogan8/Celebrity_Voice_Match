from pydub import AudioSegment
import os
import math
from pydub import AudioSegment

def get_audio_length_pydub(audio_path):
    """
    Get the duration of an audio file using pydub.

    Args:
    - audio_path (str): Path to the audio file.

    Returns:
    - float: Length of the audio in seconds.
    """
    try:
        audio = AudioSegment.from_file(audio_path)  # Load the audio file
        duration = len(audio) / 1000.0  # Convert milliseconds to seconds
        return duration
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None


def split_audio(input_file, output_folder, duration):
    audio = AudioSegment.from_mp3(input_file)
    total_length = len(audio)
    length = get_audio_length_pydub(input_file)
    print(length)
    num_parts = math.ceil(total_length / (duration * 1000))
    print(num_parts)
    input("Continue")
    j = 3726
    for i in range(num_parts):
        start = i * duration * 1000
        end = (i + 1) * duration * 1000
        split_audio = audio[start:end]
        output_path = os.path.join(output_folder, f"audio{j}.wav")
        j+=1
        split_audio.export(output_path, format="wav")
        print(f"Exported {output_path}")

input_file = "/Users/giuliarivetti/Desktop/Leiden Univ/Semester 3/Audio Processing/Final Project/improving dataset/Full Audios/RonaldRaegan.wav"
output_folder = "newDataset"  
duration = 4  # Duration in seconds for each split audio file
 
split_audio(input_file, output_folder, duration)

