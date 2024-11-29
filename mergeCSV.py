import pandas as pd

# Paths to the input CSV files
file1_path = "myDataset.csv"  # Replace with your first file's path
file2_path = "final_updated_meta.csv"  # Replace with your second file's path

# Read the CSV files
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Merge the two DataFrames
# Option 1: Concatenate rows (stack them on top of each other)
merged_df = pd.concat([df1, df2], ignore_index=True)

# Option 2: Merge columns if they have common keys
# merged_df = pd.merge(df1, df2, on="common_column", how="inner")  # Specify `on` column

# Save the merged DataFrame to a new CSV
output_path = "merged_file.csv" 
merged_df.to_csv(output_path, index=False)

print(f"Merged file saved to: {output_path}")
