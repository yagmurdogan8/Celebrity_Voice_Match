import os
from pydub import AudioSegment

def convert_to_wav(input_dir, output_dir):
    """
    Convert all audio files in the input directory to WAV format and save them in the output directory.
    
    Args:
    - input_dir (str): Path to the directory containing audio files to convert.
    - output_dir (str): Path to the directory where converted WAV files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported file extensions
    supported_extensions = ('.mp3', '.ogg', '.flac', '.aac', '.m4a', '.wma', '.mp4')  # Add more if needed

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        # Process only supported audio files
        if filename.lower().endswith(supported_extensions):
            try:
                print(f"Converting {filename} to WAV...")
                # Load the audio file
                audio = AudioSegment.from_file(file_path)
                # Define the output path with the same filename but .wav extension
                output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".wav")
                # Export as WAV
                audio.export(output_path, format="wav")
                print(f"Saved WAV file to {output_path}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Example usage
input_directory = "/path/to/your/input/directory"  # Replace with your input directory
output_directory = "/path/to/your/output/directory"  # Replace with your output directory

convert_to_wav(input_directory, output_directory)
