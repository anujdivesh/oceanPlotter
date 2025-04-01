import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from mpl_toolkits.basemap import Basemap

# WMS URL
wms_url = "https://dev-oceanportal.spc.int/thredds/wms/POP/model/regional/bom/forecast/monthly/accesss/sst/latest.nc"

# Full global bounding box in EPSG:4326 (degrees)
bbox = "-180,-90,180,90"  # Full global bbox in EPSG:4326

# WMS request parameters with full global bounding box
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
    "time": "2025-07-16T12:00:00Z",  # Specific time
    "logscale": "false",
    "width": "512",  # Image width
    "height": "512",  # Image height
    "srs": "EPSG:4326",  # Using EPSG:4326 projection (WGS84 - latitude/longitude)
    "bbox": bbox  # Full global bounding box
}

# Make the WMS request using requests
response = requests.get(wms_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Convert the response content to an image
    img = Image.open(BytesIO(response.content))

    # Create the figure and the Basemap instance
    plt.figure(figsize=(12, 6))

    # Set up the Basemap with EPSG:4326 (WGS84)
    m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, 
                llcrnrlon=-180, urcrnrlon=180, resolution='l')

    # Draw coastlines and countries for reference
    m.drawcoastlines()
    m.drawcountries()

    # Convert the image coordinates to Basemap coordinates
    img_array = np.array(img)

    # Plot the image on the Basemap
    m.imshow(img_array, origin='upper')

    # Show the map
    plt.title('Sea Surface Temperature (SST) - 2025-07-16')
    plt.show()
else:
    print("Failed to retrieve the map. Status code:", response.status_code)
