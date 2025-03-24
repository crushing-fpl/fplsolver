import pandas as pd
import glob
import os

# Path to your historical data
historical_path = 'data/historical/'

# Find all merged_gw.csv files across seasons
gw_files = glob.glob(historical_path + '*/gws/merged_gw.csv')

# Load and concatenate into one DataFrame
df_list = []
for file in gw_files:
    print(f"Loading {file}...")
    try:
        # Extract season from the path (e.g., '2024-25')
        # This gets the parent directory of 'gws', which should be the season folder
        season = os.path.basename(os.path.dirname(os.path.dirname(file)))
        # Read the CSV with latin1 encoding to handle special characters
        df = pd.read_csv(file, encoding='latin1')
        # Add the season column to this DataFrame
        df['season'] = season
        df_list.append(df)
    except Exception as e:
        print(f"Error in file {file}: {e}")
        break  # Stop at the problematic file

# If no errors, concatenate and save
if len(df_list) == len(gw_files):
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv('data/combined_historical_gw.csv', index=False)
    print("Combined data saved successfully. First few rows:")
    print(combined_df.head())
else:
    print("Fix the error in the file above before proceeding.")

# Optional: Verify the results
print("\nVerifying gameweeks per season:")
for season in combined_df['season'].unique():
    season_data = combined_df[combined_df['season'] == season]
    gw_count = season_data['GW'].nunique()
    print(f"Season {season}: {gw_count} gameweeks")

# Check latest gameweek for 2024-25
ongoing_season = '2024-25'
latest_gw = combined_df[combined_df['season'] == ongoing_season]['GW'].max()
print(f"Latest gameweek for {ongoing_season}: GW{latest_gw}")