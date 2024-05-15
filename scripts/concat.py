import pandas as pd
import os

# Directory containing CSV files
directory = './data/rockland/'

print(f"Directory path: {os.path.abspath(directory)}")

# List to store DataFrames
df_list = []

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Read each CSV file
        filepath = os.path.join(directory, filename)
        print(f"Reading file: {filepath}")  # Debugging print
        df = pd.read_csv(filepath)
        df_list.append(df)

# Check if df_list is empty
if not df_list:
    print("No CSV files found or read successfully.")
else:
    # Concatenate all DataFrames
    concatenated_df = pd.concat(df_list, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    output_filepath = './data/concatenated_file.csv'  # Adjusted output path
    concatenated_df.to_csv(output_filepath, index=False)

    print(f"Concatenated file saved to {output_filepath}")
