import pandas as pd

# Load the original CSV
copied_file = "copied_file.csv"
new_entries = []

def addEntries(name, start, end, location):
    for i in range(start, end+1):
        entry = {"file": "audio{i}.wav", "speaker": name, "label": location}
        new_entries.add(entry)

addEntries("2pac", 1, 18, "United States")
addEntries("Alan Watts", 19, 570, "United States")
addEntries("2pac", 1, 18, "United States")
addEntries("2pac", 1, 18, "United States")
addEntries("2pac", 1, 18, "United States")
addEntries("2pac", 1, 18, "United States")
addEntries("2pac", 1, 18, "United States")
addEntries("2pac", 1, 18, "United States")
addEntries("2pac", 1, 18, "United States")
addEntries("2pac", 1, 18, "United States")
addEntries("2pac", 1, 18, "United States")
addEntries("2pac", 1, 18, "United States")


# Convert the list of new entries to a DataFrame and append
new_df = pd.DataFrame(new_entries)
# Save the updated DataFrame to a new CSV file
new_df.to_csv(copied_file, index=False)

print(f"Copied and updated CSV saved to: {copied_file}")
