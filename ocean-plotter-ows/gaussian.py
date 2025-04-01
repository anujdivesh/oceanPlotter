from owslib.wms import WebMapService
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# WMS URL for Sea Surface Temperature (SST)
wms_url = "https://dev-oceanportal.spc.int/thredds/wms/POP/model/regional/bom/forecast/monthly/accesss/sst/latest.nc"

# Connect to the WMS server
wms = WebMapService(wms_url, version='1.1.1')

# Set up the parameters for the WMS request with interpolation
params = {
    'layers': 'sst',  # Layer name
    'styles': 'scalar-contour/div-RdBu-inv',  # Style for the layer
    'format': 'image/png',  # Image format
    'transparent': 'true',  # Enable transparency
    'colorscalerange': '-1,5',  # Color scale range
    'numcolorbands': '250',  # Number of color bands
    'time': '2025-03-16T12:00:00Z',  # Time for the data
    'srs': 'EPSG:4326',  # Coordinate reference system
    'bbox': (120, -45, 280, 45),  # Bounding box for the Pacific Ocean region
    'width': 1024,  # Image width
    'height': 1024,  # Image height
    'interpolation': 'bicubic',  # Apply bicubic interpolation
}

# Make the WMS request
img = wms.getmap(**params)

# Convert the response to an image
img_data = BytesIO(img.read())
img = Image.open(img_data)

# Display the image using Matplotlib
plt.figure(figsize=(14, 6))
plt.imshow(img, extent=(120, 280, -45, 45), origin='upper')
plt.title('Sea Surface Temperature (SST) - Pacific Ocean (2025-07-16)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='SST (°C)')
plt.show()