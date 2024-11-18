import os
import pandas as pd
import shutil

# Load the original CSV file
csv_path = r'C:\Users\Hobbi\PycharmProjects\Celebrity_Voice_Match\meta.csv'
df = pd.read_csv(csv_path)

# Filter only bona-fide files
bonafide_df = df[df['label'] == 'bona-fide']

# Directory containing the .wav files
wav_directory = r'C:\Users\Hobbi\PycharmProjects\Celebrity_Voice_Match\release_in_the_wild'

# Initialize list to hold new metadata information
new_metadata = []

# Sort and rename bona-fide files in numerical order
for idx, original_filename in enumerate(bonafide_df['file']):
    old_filepath = os.path.join(wav_directory, original_filename)
    new_filename = f"{idx}.wav"
    new_filepath = os.path.join(wav_directory, new_filename)

    # Rename the file only if it exists in the directory
    if os.path.exists(old_filepath):
        shutil.move(old_filepath, new_filepath)

        # Append new metadata information
        speaker = bonafide_df[bonafide_df['file'] == original_filename]['speaker'].values[0]
        label = 'bona-fide'
        new_metadata.append([new_filename, speaker, label])

# Create a new DataFrame for the updated metadata
new_meta_df = pd.DataFrame(new_metadata, columns=['file', 'speaker', 'label'])

# Save the new metadata to a CSV
new_csv_path = r'C:\Users\Hobbi\PycharmProjects\Celebrity_Voice_Match\new_meta.csv'
new_meta_df.to_csv(new_csv_path, index=False)

print(f"New metadata CSV created at {new_csv_path}. The files have been renamed in numerical order.")
