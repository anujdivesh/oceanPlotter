import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PIL import Image
from io import BytesIO
import geopandas as gpd

def getBBox(country_id):
    # Fetch bounding box data from API
    api_url = "https://dev-oceanportal.spc.int/middleware/api/country/"+str(country_id)+"/"
    response = requests.get(api_url)
    west_bound, east_bound, south_bound, north_bound = "", "", "",""

    if response.status_code == 200:
        data = response.json()
        west_bound = data['west_bound_longitude']
        east_bound = data['east_bound_longitude']
        south_bound = data['south_bound_latitude']
        north_bound = data['north_bound_latitude']
    else:
        print(f"Failed to retrieve bounding box data. Status code: {response.status_code}")
    
    return west_bound, east_bound, south_bound, north_bound

def getLegend(ax_legend):
     # Custom legend creation
    ini_url = "https://dev-oceanportal.spc.int/thredds/wms/POP/model/regional/bom/forecast/hourly/wavewatch3/latest.nc"
    palette = "x-Sst"
    layer = "mn_wav_dir"
    min_color = 0
    max_color = 4
    steps = 6  # Number of steps
    position = steps - 1
    units = "&"
    width = 25
    height = 300

    # URL of the legend image
    legend_url = "%s?REQUEST=GetLegendGraphic&PALETTE=%s&LAYERS=%s&COLORSCALERANGE=%s,%s&COLORBARONLY=true&WIDTH=%s&HEIGHT=%s" % (ini_url, palette, layer, min_color, max_color,width, height)

    # Fetch the image from the URL
    response = requests.get(legend_url)
    if response.status_code == 200:
        # Load the image using PIL
        image = Image.open(BytesIO(response.content))

        # Add the legend on the right side of the map
        
        ax_legend.imshow(np.array(image))  # Show the legend image

        # Add labels for the steps
        steps = np.linspace(min_color, max_color, steps)  # Steps from min to max value
        for i, step in enumerate(steps):
            # Calculate the position of the label
            ax_legend.text(1.05, (i / position), f'{step:.1f}', transform=ax_legend.transAxes, fontsize=8, color='black', va='center', ha='left')
        if units != "&":
            # Add units on the left, aligned to the bottom in vertical orientation
            ax_legend.text(1.9, 0.5, units, transform=plt.gca().transAxes, fontsize=10, color='black', va='center', ha='left', rotation=90)

        ax_legend.axis('off')  # Hide the axes of the legend
    else:
        print(f"Failed to fetch the legend image. Status code: {response.status_code}")

def getEEZ(ax):
    geojson_url = "https://opmgeoserver.gem.spc.int/geoserver/spc/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=spc:pacific_eez3&srsName=EPSG:4326&outputFormat=application/json"
    geojson_response = requests.get(geojson_url)

    if geojson_response.status_code == 200:
        # Load GeoJSON data using geopandas
        geojson_data = geojson_response.json()
        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

        # Ensure the GeoDataFrame is in the same CRS as the map (EPSG:4326)
        gdf = gdf.set_crs('EPSG:4326', allow_override=True)

        # Convert GeoJSON coordinates to Basemap projection coordinates
        for geom in gdf.geometry:
            if geom.is_valid:  # Check if the geometry is valid
                # For each geometry, extract the boundary (line) and convert to Basemap projection
                for geom_line in geom.geoms:  # In case the geometry is MultiPolygon or similar
                    # Convert coordinates to Basemap projection
                    x, y = m(geom_line.xy[0], geom_line.xy[1])  # Convert coordinates to Basemap projection
                    
                    # Shift the longitudes to avoid the 180° cutoff
                    x = np.array(x)  # Ensure x is a numpy array for element-wise operations
                    x = np.where(x < 0, x + 360, x)  # For longitudes < 0 (e.g., -170°), shift to +180°
                    
                    # Plot the boundary line
                    ax.plot(x, y, marker=None, color='pink', linewidth=2)  # Plot the boundary line

    else:
        print("Failed to retrieve the GeoJSON data.")

def getMap(west_bound, east_bound, south_bound, north_bound):
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
        "bbox": "%s,%s,%s,%s" % (west_bound, south_bound, east_bound, north_bound)
        #"bbox": "120,-45,300,45",  # Bounding box for the Pacific Ocean region (0-360 longitudes)
    }

    # Make the WMS request using requests
    response = requests.get(wms_url, params=params)
    img = Image.open(BytesIO(response.content))
    return img

def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple()

#BOUNDINGBOX INFO
west_bound, east_bound, south_bound, north_bound = getBBox(1)
bbox_offset = 1
print(west_bound, east_bound, south_bound, north_bound)

figsize = cm2inch((28,16))
fig, ax = plt.subplots(figsize=figsize, dpi=300)


logo_url = "/Users/anujdivesh/Desktop/django/plotter/ows_plots/Logo_cropped.png"  # Path to the logo image
logo = Image.open(logo_url)

# Resize the logo to a larger size (e.g., 400x400)
logo = logo.resize((440, 150))

# Add logo to the plot
ax_logo = fig.add_axes([0.02, 0.84, 0.13, 0.15])  # Adjusting position and size for the larger logo
ax_logo.imshow(logo)
ax_logo.axis('off')  

# Optionally, you can hide the main axis if necessary
ax.axis('off')

#Create Main Figure
ax2 = fig.add_axes([0.09, 0.2, 0.8, 0.65])
ax2.set_title("Pacific Ocean \n Daily Average Sea Surface Temperature Anomaly: 16 January 2026", pad=10)

# Create Basemap
m = Basemap(projection='cyl', llcrnrlat=south_bound, urcrnrlat=north_bound, 
            llcrnrlon=west_bound, urcrnrlon=east_bound, resolution='l', ax=ax2)

# Draw coastlines and countries
m.drawcoastlines()
m.fillcontinents(color='grey', lake_color='white')
m.drawcountries()

#Axes labels
m.drawparallels(np.arange(south_bound, north_bound+1., 15.), labels=[1,0,0,0])  # Latitude lines
lon_ticks = np.arange(west_bound, east_bound+bbox_offset, 30)  # Create longitude ticks every 20 degrees
m.drawmeridians(lon_ticks, labels=[0, 0, 0, 1])  # Labels on all sides

#GETMAPP
img_array = np.array(getMap(west_bound, east_bound, south_bound, north_bound))

#ax2 = fig.add_axes([0.09, 0.2, 0.8, 0.65])

ax_legend = fig.add_axes([0.9, 0.25, 0.05, 0.55])  # Position legend on the right side

getLegend(ax_legend)
ax_legend.axis('off')
# Plot the SST image on the Basemap
m.imshow(img_array, origin='upper')


ax2.text(-0.08, 0.03,"Copyright Pacific Community (SPC) 2025", transform=ax.transAxes,fontsize=8, verticalalignment='top')
ax2.text(-0.08, 0.005,"Climate and Ocean Support Program in the Pacific (COSPPac)", transform=ax.transAxes,fontsize=8, verticalalignment='top')
ax2.text(0.80, 0.03,"Reynolds SST", transform=ax.transAxes,fontsize=8, verticalalignment='top')

getEEZ(ax2)
plt.savefig('sst.png', dpi=300) 
