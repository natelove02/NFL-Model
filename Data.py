import nfl_data_py as nfl
import pandas as pd

# Load play-by-play data for the 2023 season
pbp_data = nfl.import_pbp_data([2023])
nfl.see_weekly_cols()
# Display the first few rows of data to see the columns
#print("Play-by-Play Data Columns:")
#print(pbp_data.columns)  # Displays all available columns
#print("\nSample of Play-by-Play Data:")
#print(pbp_data.head())  # Shows a sample of the data
# Save play-by-play data to CSV for inspection
#pbp_data.to_csv("pbp_data_2023.csv", index=False)
#print("Play-by-play data saved as 'pbp_data_2023.csv'")
