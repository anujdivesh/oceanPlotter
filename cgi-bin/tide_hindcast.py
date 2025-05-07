#!/usr/bin/python3
import cgi
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PIL import Image
from io import BytesIO
import geopandas as gpd
import sys
from datetime import datetime
from io import BytesIO
import sys, os
import cgi, cgitb
import io
import requests
import pandas as pd
from scipy.stats import linregress
#inputs
form = cgi.FieldStorage() 
country_name = form.getvalue('country')
location = form.getvalue('location')
station_id = form.getvalue('station_id')

#country_name = "Fiji"
#location = "Suva"
#station_id	= "IDO70063"
#end inputs
copyright_text = "© Pacific Community (SPC) 2025"
footer_text = "Climate and Ocean Support Program in the Pacific (COSPPac)"
logo_url = "./Logo_cropped.png" 


#Functions
def download_txt_file(url, filename):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Save the content to a file
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(response.text)
        #print(f"File downloaded successfully as '{filename}'")
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
#print(url)
filename = "%sSLD.txt" % (station_id)

#DOWNLOAD THE FILE
download_txt_file(url, filename)

#READ THE DATA
df = read_sea_level_data("/usr/lib/cgi-bin/"+filename)
#print('Anuj')
# Ensure 'Date' column is created correctly
df["Date"] = pd.to_datetime(df.assign(Day=1)[["Year", "Month", "Day"]])

# Compute trend line for the "Mean" values
x = np.arange(len(df))
slope_mean, intercept_mean, _, _, _ = linregress(x, df["Mean"])

# Convert slope from mm/month to mm/year
slope_mean_per_year = slope_mean * 12 * 1000

df["Mean_Trend"] = intercept_mean + slope_mean * x

fig, ax = plt.subplots(figsize=(12, 6.5))

# Plot the data using ax
ax.plot(df["Date"], df["Mean"], label="Mean", color="blue")
ax.plot(df["Date"], df["Maximum"], label="Maximum", color="red", alpha=0.5)
ax.plot(df["Date"], df["Minimum"], label="Minimum", color="green", alpha=0.5)
ax.plot(df["Date"], df["Mean_Trend"], "--", color="blue", label="Mean Trend")

# Display the gradient (slope) in mm/year on the plot
ax.text(df["Date"].iloc[5], max(df["Mean"]), 
        f"Mean Trend Slope: {slope_mean_per_year:.2f} mm/year", 
        fontsize=12, color="blue", bbox=dict(facecolor="white", alpha=0.6))

# Set axis limits to remove gaps
ax.set_xlim([df["Date"].min(), df["Date"].max()])
ax.set_ylim([min(df["Minimum"].min(), df["Mean_Trend"].min()) * 0.98, 
             max(df["Maximum"].max(), df["Mean_Trend"].max()) * 1.02])

# Formatting using ax
ax.set_xlabel("Year")
ax.set_ylabel("Sea Level (m)")
ax.legend()
ax.set_title("Relative Sea Level (Mean, Max, Min) with Mean Trend \n %s - %s" % (country_name, location))
ax.grid(True)

logo = Image.open(logo_url)
logo = logo.resize((440, 150))
ax2_pos = ax.get_position()
#ax_logo = fig.add_axes([0.07, ax2_pos.y1 - 0.02, 0.13, 0.15])  # Adjust the y-position slightly above ax2
#ax_logo.imshow(logo)
#ax_logo.axis('off')  

ax.text(-0.08, ax2_pos.y0-0.17,copyright_text, transform=ax.transAxes,fontsize=7, verticalalignment='top')
ax.text(-0.08, ax2_pos.y0-0.195,footer_text, transform=ax.transAxes,fontsize=7, verticalalignment='top')

output_filename = station_id+".png"
plt.savefig(output_filename, dpi=300) 

length = os.stat(output_filename).st_size

sys.stdout.write("Content-Type: image/png\n")
sys.stdout.write("Content-Length: " + str(length) + "\n")
sys.stdout.write("\n")
sys.stdout.flush()
sys.stdout.buffer.write(open(output_filename, "rb").read())