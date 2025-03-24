import pandas as pd
import glob

# Path to your historical data
historical_path = 'data/historical/'

# Find all merged_gw.csv files across seasons
gw_files = glob.glob(historical_path + '*/gws/merged_gw.csv')

# Load and concatenate into one DataFrame
df_list = []
for file in gw_files:
    print(f"Loading {file}...")
    try:
        df = pd.read_csv(file)
        df_list.append(df)
    except pd.errors.ParserError as e:
        print(f"Error in file {file}: {e}")
        break  # Stop at the problematic file

# If no errors, concatenate and save
if len(df_list) == len(gw_files):
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv('data/combined_historical_gw.csv', index=False)
    print(combined_df.head())
else:
    print("Fix the error in the file above before proceeding.")