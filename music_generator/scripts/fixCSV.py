import pandas as pd

# Load the CSV file
csv_path = "/Users/drinkurtishi/Desktop/AI-project/music_generator/data/vgmidi_labelled.csv"  # Replace with the actual path to your CSV
df = pd.read_csv(csv_path)

# Hardcoded replacements to remove specific suffixes (_0, _1, _2, _3) from the 'midi' column
for suffix in ['_0', '_1', '_2', '_3']:
    df['midi'] = df['midi'].str.replace(f'{suffix}.mid', '.mid', regex=False)

# Save the updated CSV
output_path = "/Users/drinkurtishi/Desktop/AI-project/music_generator/data/vgmidi_labelled.csv"  # Replace with your desired output path
df.to_csv(output_path, index=False)

print(f"Updated CSV saved to {output_path}")
