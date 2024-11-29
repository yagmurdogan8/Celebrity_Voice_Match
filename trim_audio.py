from pydub import AudioSegment

def trim_audio(input_path, output_path, cutoff_time=300):
    """
    Trims an audio file to the specified cutoff time.

    Args:
    - input_path (str): Path to the input audio file.
    - output_path (str): Path to save the trimmed audio file.
    - cutoff_time (int): Maximum duration in seconds (default: 300 seconds).

    Returns:
    - None
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_path)
        
        # Convert cutoff time to milliseconds
        cutoff_time_ms = cutoff_time * 1000
        
        # Trim the audio to the cutoff time
        trimmed_audio = audio[:cutoff_time_ms]
        
        # Export the trimmed audio to a new file
        trimmed_audio.export(output_path, format="wav") 
        print(f"Trimmed audio saved to: {output_path}")
    except Exception as e:
        print(f"Error trimming audio: {e}")

# Example usage
input_audio = "input_audio_file.wav"
output_audio = "trimmed_audio_file.wav" 
trim_audio(input_audio, output_audio, cutoff_time=300)  # Trim to the first 5 minutes
