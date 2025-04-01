import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from mpl_toolkits.basemap import Basemap

# WMS URL for Sea Surface Temperature (SST)
wms_url = "https://dev-oceanportal.spc.int/thredds/wms/POP/model/regional/bom/forecast/monthly/accesss/sst/latest.nc"

# Set up the parameters for the WMS request with higher resolution
params = {
    "service": "WMS",
    "request": "GetMap",
    "layers": "sst",
    "styles": "scalar-contour/x-Sst",
    "format": "image/png",
    "transparent": "true",
    "version": "1.1.1",  # WMS version 1.1.1
    "colorscalerange": "-1, 5",
    "numcolorbands": "250",
    "time": "2025-07-16T12:00:00Z",  # Specific time for SST data
    "logscale": "false",
    "width": "1024",  # Increased image width for higher resolution
    "height": "1024",  # Increased image height for higher resolution
    "srs": "EPSG:4326",  # Using EPSG:4326 projection (latitude/longitude)
    "bbox": "120,-45,280,45",  # Bounding box for the Pacific Ocean region (0-360 longitudes)
}

# Make the WMS request using requests
response = requests.get(wms_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Convert the response content to an image
    img = Image.open(BytesIO(response.content))

    # Create the figure for the map
    fig, ax = plt.subplots(figsize=(14, 6))

    # Set up Basemap with cylindrical projection for the Pacific Ocean region (0 to 360 longitudes)
    m = Basemap(projection='cyl', llcrnrlat=-45, urcrnrlat=45, 
                llcrnrlon=120, urcrnrlon=280, resolution='l')

    # Draw coastlines and countries
    m.drawcoastlines()
    m.drawcountries()

    # Add grid lines for latitude and longitude
    m.drawparallels(np.arange(-45., 46., 15.), labels=[1,0,0,0])  # Latitude lines

    # Draw meridians (longitudes) without initial labels here
    m.drawmeridians(np.arange(120., 281., 60.), labels=[0,0,0,0])  # Longitude lines without labels

    # Manually set longitude labels for 120, 180, 240 (without repeating 120E)
    meridian_labels = ['120°E', '180°E', '240°E']  # Set the labels for the meridians
    # Manually place labels for meridians
    m.drawmeridians(np.arange(120, 241, 60), labels=[0, 0, 0, 1], fmt='%d°E')  # Labels for 120°, 180°, 240°

    # Set title
    plt.title('Sea Surface Temperature (SST) - Pacific Ocean (2025-07-16)')

    # Convert the image coordinates to Basemap coordinates
    img_array = np.array(img)

    # Plot the SST image on the Basemap
    m.imshow(img_array, origin='upper')

    # Make a GetLegendGraphic request for the new legend (WaveWatch 3 model - Direction)
    legend_url = "https://dev-oceanportal.spc.int/thredds/wms/POP/model/regional/bom/forecast/hourly/wavewatch3/latest.nc"
    legend_params = {
        "REQUEST": "GetLegendGraphic",
        "PALETTE": "default",
        "LAYERS": "mn_wav_dir",
        "STYLES": "raster/x-Sst",
        "COLORSCALERANGE": "0,4",  # Legend scale for wave direction
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
    }

    # Fetch the legend image
    legend_response = requests.get(legend_url, params=legend_params)

    legend_img = Image.open(BytesIO(legend_response.content))

    # Create a subplot for the legend, placing it next to the map (right side)
    # Adjust the position of the legend to sit directly next to the map
    ax_legend = fig.add_axes([0.85, 0.1, 0.05, 0.8])  # Position legend on the right side
    ax_legend.imshow(legend_img)  # Show the legend image
    ax_legend.axis('off')  # Hide the axes of the legend
    footer_text = '© All rights reserved SPC'

    # Position the footer text at the bottom-right of the figure
    plt.figtext(0.95, 0.02, footer_text, ha='right', va='bottom', fontsize=10, color='black', fontweight='light')

    # Adjusting layout for extra space
    plt.subplots_adjust(top=0.92, bottom=0.05)  # 0.92 for top, 0.05 for bottom space

    # Display the combined map and legend
    plt.tight_layout()  # Ensures no overlap between map and legend
    plt.show()

else:
    print("Failed to retrieve the map. Status code:", response.status_code)
