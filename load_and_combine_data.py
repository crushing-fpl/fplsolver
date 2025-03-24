import pandas as pd
import glob

# Path to your historical data
historical_path = 'data/historical/'

# Find all merged_gw.csv files across seasons
gw_files = glob.glob(historical_path + '*/gws/merged_gw.csv')

# Load and concatenate into one DataFrame
df_list = [pd.read_csv(file) for file in gw_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined dataset
combined_df.to_csv('data/combined_historical_gw.csv', index=False)
print(combined_df.head())