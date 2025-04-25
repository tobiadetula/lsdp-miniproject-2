import utm
import os
import pandas as pd
import numpy as np

csv_file = "DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv"
main_dir = "../"
csv_file_path = os.path.join(main_dir, csv_file)

# Check if the file exists
def read_file(file_path):
    # Read the CSV file with an alternative encoding
    try:
        df = pd.read_csv(csv_file_path, encoding='latin1')  # Use 'latin1' to handle non-UTF-8 characters
    except UnicodeDecodeError as e:
        print(f"Error reading the file: {e}")
        exit(1)
    return df
    

df = read_file(csv_file_path)
# Extract the required columns
gps_data = df[['OSD.latitude', 'OSD.longitude']]

# Save the extracted data to a new CSV file
output_file = os.path.join(os.path.dirname(__file__), "gps_coordinates.csv")
gps_data.to_csv(output_file, index=False)