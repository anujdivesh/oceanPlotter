#!/usr/bin/python3
#import cgi
import json
import numpy as np
import requests
import pandas as pd

#inputs
country_name = "Fiji"
location = "Suva"
station_id	= "IDO70063"
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
#main
url = "http://reg.bom.gov.au/ntc/%s/%sSLD.txt"% (station_id,station_id)
print(url)
filename = "%sSLD.txt" % (station_id)

#DOWNLOAD THE FILE
download_txt_file(url, filename)

#READ THE DATA
data = read_sea_level_data(filename)

print(data)