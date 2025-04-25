import utm
import os
import pandas as pd
import numpy as np

csv_file = "DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv"
main_dir = "../"
csv_file_path = os.path.join(main_dir, csv_file)

# Append UTM coordinates to a list
utm_data = []

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

first_only = True
for index, row in gps_data.iterrows():
    lat = row['OSD.latitude']
    lon = row['OSD.longitude']
    
    # Convert latitude and longitude to UTM coordinates
    utm_coords = utm.from_latlon(lat, lon)
    
    if first_only:
        # Print the UTM coordinates for the first row only
        print(f"Latitude: {lat}, Longitude: {lon} -> UTM Coordinates: {utm_coords}")
        first_only = False
    # Add UTM coordinates to the list
    utm_data.append({
        'Latitude': lat,
        'Longitude': lon,
        'UTM_Easting': utm_coords[0],
        'UTM_Northing': utm_coords[1],
        'UTM_Zone_Number': utm_coords[2],
        'UTM_Zone_Letter': utm_coords[3]
    })

# After the loop, save the UTM data to a CSV file
utm_output_file = os.path.join(os.path.dirname(__file__), "utm_coordinates.csv")
utm_df = pd.DataFrame(utm_data)
utm_df.to_csv(utm_output_file, index=False)