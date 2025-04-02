#!/usr/bin/python3
#import cgi
import json
import numpy as np
import requests
import pandas as pd
from scipy.stats import linregress
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#inputs
country_name = "Fiji"
location = "Suva"
station_id	= "IDO70054"
#end inputs

#Functions
def download_txt_file(url, filename):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Save the content to a file
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"File downloaded successfully as '{filename}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")

def read_sea_level_data(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Skip header lines until the data starts
        start_reading = False
        for line in lines:
            line = line.strip()
            
            # Detect the start of the data section (after "Mth Year ...")
            if line.startswith("Mth Year  Gaps"):
                start_reading = True
                continue
            
            # Stop reading at the footer (e.g., "Mean sea level = ...")
            if line.startswith("Mean sea level") or line.startswith("Maximum recorded"):
                break
            
            # Parse data rows (e.g., "11 1997  2172 ...")
            if start_reading and line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 7:  # Ensure all columns are present
                    record = {
                        "Month": int(parts[0]),
                        "Year": int(parts[1]),
                        "Gaps": int(parts[2]),
                        "Good": int(parts[3]),
                        "Minimum": float(parts[4]),
                        "Maximum": float(parts[5]),
                        "Mean": float(parts[6]),
                        "St Devn": float(parts[7]) if len(parts) > 7 else None,
                    }
                    data.append(record)
    
    return pd.DataFrame(data)


# Main
url = f"http://reg.bom.gov.au/ntc/{station_id}/{station_id}SLD.txt"
filename = f"{station_id}SLD.txt"

# Download and read data
download_txt_file(url, filename)
df = read_sea_level_data(filename)

# Ensure 'Date' column is created correctly
df["Date"] = pd.to_datetime(df.assign(Day=1)[["Year", "Month", "Day"]])

# Compute trend line for the "Mean" values
x = np.arange(len(df))
slope_mean, intercept_mean, _, _, _ = linregress(x, df["Mean"])

# Convert slope from mm/month to mm/year
slope_mean_per_year = slope_mean * 12 * 1000

df["Mean_Trend"] = intercept_mean + slope_mean * x

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Mean"], label="Mean", color="blue")
plt.plot(df["Date"], df["Maximum"], label="Maximum", color="red", alpha=0.5)
plt.plot(df["Date"], df["Minimum"], label="Minimum", color="green", alpha=0.5)
plt.plot(df["Date"], df["Mean_Trend"], "--", color="blue", label="Mean Trend")

# Display the gradient (slope) in mm/year on the plot
plt.text(df["Date"].iloc[5], max(df["Mean"]), 
         f"Mean Trend Slope: {slope_mean_per_year:.2f} mm/year", 
         fontsize=12, color="blue", bbox=dict(facecolor="white", alpha=0.6))

# Formatting
plt.xlabel("Year")
plt.ylabel("Sea Level (m)")
plt.legend()
plt.title("Monthly Sea Level (Mean, Max, Min) with Mean Trend")
plt.grid()

plt.show()
