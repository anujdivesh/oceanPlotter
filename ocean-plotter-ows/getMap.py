import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PIL import Image
from io import BytesIO
import geopandas as gpd
import sys
from datetime import datetime

##INPUT PARAMS###
region = 1
time = "2025-01-01T00:00:00Z"
layer_map = 17
legend_steps = 6
units = "null"
resolution = "l"
coral = False

def fetch_wms_layer_data(api_url):
    # Make a GET request to fetch the data from the API
    response = requests.get(api_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract the required values directly
        wms_url = data.get('url')
        layer_name = data.get('layer_name')
        transparent = data.get('transparent')
        styles = data.get('style')
        min_color = data.get('colormin')
        max_color = data.get('colormax')
        numcolorbands = data.get('numcolorbands')
        time_interval_start = data.get('timeIntervalStart')
        time_interval_end = data.get('timeIntervalEnd')
        period = data.get('period')
        interval_step = data.get('interval_step')
        logscale = data.get('logscale')
        get_map_names = data.get('get_map_names')
        
        # Return all extracted values
        return {
            'wms_url': wms_url,
            'layer_name': layer_name,
            'transparent': transparent,
            'styles': styles,
            'min_color': min_color,
            'max_color': max_color,
            'numcolorbands': numcolorbands,
            'time_interval_start': time_interval_start,
            'time_interval_end': time_interval_end,
            'period': period,
            'interval_step': interval_step,
            'logscale': logscale,
            'get_map_names':get_map_names
        }
    else:
        # If request fails, return a message
        return {'error': f"Failed to fetch data. Status code: {response.status_code}"}

region_url_prefix = "https://dev-oceanportal.spc.int/middleware/api/country/"
eez_url = "https://opmgeoserver.gem.spc.int/geoserver/spc/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=spc:pacific_eez3&srsName=EPSG:4326&outputFormat=application/json"
copyright_text = "© Pacific Community (SPC) 2025"
footer_text = "Climate and Ocean Support Program in the Pacific (COSPPac)"
layer_web_map_url = "https://dev-oceanportal.spc.int/middleware/api/layer_web_map/%s/" % (layer_map)
layer_data = fetch_wms_layer_data(layer_web_map_url)
layers_arr = layer_data['layer_name'].split(",")
total_layers = len(layers_arr)
wms_url = layer_data['wms_url']
layers = layer_data['layer_name']
transparent = layer_data['transparent']
styles = layer_data['styles']
style_split = styles.split('/')
palette = style_split[1]
min_color = layer_data['min_color']
max_color = layer_data['max_color']
numcolorbands = layer_data['numcolorbands']
logscale = layer_data['logscale']
date = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
formatted_date = date.strftime("%-d %B %Y")
formatted_date2 = date.strftime("%Y%m%d")
get_map_names = layer_data['get_map_names']

title_suffix = "Daily Average Sea Surface Temperature Anomaly: %s" % (formatted_date)
dataset_text = "Reynolds SST"
"""
print(get_map_names)
if get_map_names != None or get_map_names != "":
    get_map_names = layer_data['get_map_names'].split('/')
    formatted_date = date.strftime(get_map_names[1])
    title_suffix = "%s: %s" % (get_map_names[0],formatted_date)
    dataset_text = get_map_names[2]
"""
logo_url = "./Logo_cropped.png" 

output_filename = "%s_%s_%s.png" % (formatted_date2,layers,layer_map)

if total_layers > 1:
    layers = layer_data['layer_name'].split(',')[0]
    styles = layer_data['styles'].split(',')[0]
    style_split = styles.split('/')
    palette = style_split[1]
##END PARAMS###

####FUNCTIONS

def getBBox(country_id):
    # Fetch bounding box data from API
    api_url = "%s%s/" % (region_url_prefix,str(country_id))
    response = requests.get(api_url)
    west_bound, east_bound, south_bound, north_bound,country_name = "", "", "","",""

    if response.status_code == 200:
        data = response.json()
        west_bound = data['west_bound_longitude']
        east_bound = data['east_bound_longitude']
        south_bound = data['south_bound_latitude']
        north_bound = data['north_bound_latitude']
        country_name = data['long_name']
    else:
        print(f"Failed to retrieve bounding box data. Status code: {response.status_code}")
    
    return west_bound, east_bound, south_bound, north_bound, country_name
"""
def getLegend(ax_legend,wms_url,palette,layers,min_color,max_color,steps,units):
     # Custom legend creation
    position = steps - 1
    width = 18
    height = 300

    # URL of the legend image
    legend_url = "%s?REQUEST=GetLegendGraphic&PALETTE=%s&LAYERS=%s&COLORSCALERANGE=%s,%s&COLORBARONLY=true&WIDTH=%s&HEIGHT=%s" % (wms_url, palette, layers, min_color, max_color,width, height)

    # Fetch the image from the URL
    response = requests.get(legend_url)
    if response.status_code == 200:
        # Load the image using PIL
        image = Image.open(BytesIO(response.content))

        # Add the legend on the right side of the map
        
        ax_legend.imshow(np.array(image))  # Show the legend image

        # Add labels for the steps
        steps = np.linspace(float(min_color), float(max_color), steps)  # Steps from min to max value
        for i, step in enumerate(steps):
            # Calculate the position of the label
            ax_legend.text(1.05, (i / position), f'{step:.1f}', transform=ax_legend.transAxes, fontsize=6, color='black', va='center', ha='left')
        if units != "null":
            # Add units on the left, aligned to the bottom in vertical orientation
            ax_legend.text(1.9, 0.5, units, transform=plt.gca().transAxes, fontsize=8, color='black', va='center', ha='left', rotation=90)

        ax_legend.axis('off')  # Hide the axes of the legend
    else:
        print(f"Failed to fetch the legend image. Status code: {response.status_code}")
"""
def getLegend(ax_legend, wms_url, palette, layers, min_color, max_color, steps, units, coral):
    # Custom hardcoded legend values
    def custom_legend():
        colors = ['#ADD8E6', '#FFFF00', '#FFA500', '#FF0000', '#800000']  # Green, Yellow, Orange, Red
        labels = ['No Stress', 'Watch', 'Warning', 'Alert Level 1', 'Alert Level 2']
        values = ['0.0', '1.0', '2.5', '5.0', '10.0']  # Example values for each label

        position = steps - 1
        width = 18
        height = 300

        # Create a figure for the custom legend
        for i in range(len(colors)):
            # Create an extremely slim color bar for each color (width set to 0.02)
            ax_legend.fill_between([0, 0.1], i, i + 1, color=colors[i])  # Color bar width is now 0.02

            # Add the value and label on the side of the color bar
            ax_legend.text(0.13, (i + 0.5), f'{labels[i]}', fontsize=6, color='black', va='center', ha='left')

        ax_legend.axis('off')  # Hide the axes of the legend

    # Code for fetching the legend image from WMS server
    def fetch_legend_image():
        # URL of the legend image
        legend_url = "%s?REQUEST=GetLegendGraphic&PALETTE=%s&LAYERS=%s&COLORSCALERANGE=%s,%s&COLORBARONLY=true&WIDTH=%s&HEIGHT=%s" % (
            wms_url, palette, layers, min_color, max_color, 18, 300
        )

        # Fetch the image from the URL
        response = requests.get(legend_url)
        if response.status_code == 200:
            # Load the image using PIL
            image = Image.open(BytesIO(response.content))

            # Add the legend on the right side of the map
            ax_legend.imshow(np.array(image))  # Show the legend image

            # Add labels for the steps
            steps_values = np.linspace(float(min_color), float(max_color), steps)  # Steps from min to max value
            position = steps - 1
            for i, step in enumerate(steps_values):
                ax_legend.text(1.05, (i / position), f'{step:.1f}', transform=ax_legend.transAxes, fontsize=6, color='black', va='center', ha='left')

            if units != "null":
                # Add units on the left, aligned to the bottom in vertical orientation
                ax_legend.text(1.9, 0.5, units, transform=plt.gca().transAxes, fontsize=8, color='black', va='center', ha='left', rotation=90)

            ax_legend.axis('off')  # Hide the axes of the legend
        else:
            print(f"Failed to fetch the legend image. Status code: {response.status_code}")

    # Choose the method based on the outlook flag
    if coral:
        custom_legend()  # Display the custom legend
    else:
        fetch_legend_image()  # Fetch and display the legend from the WMS server

def getEEZ(ax):
    geojson_url = eez_url
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

def getMap(west_bound, east_bound, south_bound, north_bound,wms_url,layers,transparent,styles,min_color,max_color,numcolorbands,time,logscale):
    # WMS URL for Sea Surface Temperature (SST)

    # Set up the parameters for the WMS request with higher resolution
    params = {
        "service": "WMS",
        "request": "GetMap",
        "layers": layers,
        "styles": styles,
        "format": "image/png",
        "transparent": transparent,
        "version": "1.1.1",  # WMS version 1.1.1
        "colorscalerange": "%s,%s" % (min_color, max_color),
        "numcolorbands": numcolorbands,
        "time": time,  # Specific time for SST data
        "logscale": logscale,
        "width": "1024",  # Increased image width for higher resolution
        "height": "1024",  # Increased image height for higher resolution
        "srs": "EPSG:4326",  # Using EPSG:4326 projection (latitude/longitude)
        "bbox": "%s,%s,%s,%s" % (west_bound, south_bound, east_bound, north_bound)
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

####MAIN###
west_bound, east_bound, south_bound, north_bound, country_name = getBBox(region)
bbox_offset = 1
if region == 1:
    bbox_offset = 20
    
figsize = cm2inch((18,13))
fig, ax = plt.subplots(figsize=figsize, dpi=300)

# Optionally, you can hide the main axis if necessary
ax.axis('off')

#Create Main Figure
ax2 = fig.add_axes([0.09, 0.2, 0.8, 0.65])
title = "%s \n %s" % (country_name,title_suffix)
ax2.set_title(title, pad=10, fontsize=8)

# Create Basemap
m = Basemap(projection='cyl', llcrnrlat=south_bound, urcrnrlat=north_bound, 
            llcrnrlon=west_bound, urcrnrlon=east_bound, resolution=resolution, ax=ax2)

# Draw coastlines and countries
m.drawcoastlines()
m.fillcontinents(color='#A9A9A9', lake_color='white')
m.drawcountries()

num_longitudes = east_bound - west_bound + 1

# Determine spacing based on the number of longitudes
if num_longitudes < 10:
    bbox_offset = 1  # Use 1-degree spacing if there are fewer than 10 longitudes
elif num_longitudes < 20:
    bbox_offset = 2  # Use 2-degree spacing if there are fewer than 20 longitudes
else:
    bbox_offset = 20  # Default spacing for 20 or more longitudes

#Axes labels
m.drawparallels(np.arange(south_bound, north_bound+1., bbox_offset), labels=[1,0,0,0],fmt='%1.0f',fontsize=6,color='grey')  # Latitude lines
lon_ticks = np.arange(west_bound, east_bound+1, bbox_offset)  # Create longitude ticks every 20 degrees
m.drawmeridians(lon_ticks, labels=[0, 0, 0, 1],fmt='%1.0f',fontsize=6,color='grey')  # Labels on all sides

#GETMAPP
img_array = np.array(getMap(west_bound, east_bound, south_bound, north_bound,wms_url,\
layers,transparent,styles,min_color,max_color,numcolorbands,time,logscale))

m.imshow(img_array, origin='upper')

if total_layers > 1:
    img_array = np.array(getMap(west_bound, east_bound, south_bound, north_bound,wms_url,\
    layer_data['layer_name'].split(',')[1],transparent,layer_data['styles'].split(',')[1],min_color,max_color,numcolorbands,time,logscale))

    m.imshow(img_array, origin='upper')


getEEZ(ax2)

ax2_pos = ax2.get_position()

# Create ax_legend next to ax2
ax_legend_width = 0.02  # Width of the legend
ax_legend_gap = 0.01  # Gap between ax2 and ax_legend
ax_legend = fig.add_axes([ax2_pos.x1 + ax_legend_gap, ax2_pos.y0, ax_legend_width, ax2_pos.height])  # Align with ax2
getLegend(ax_legend,wms_url,palette,layers,min_color,max_color,legend_steps,units,coral)

# Adjusting the logo position based on ax2's position
logo = Image.open(logo_url)

ax2.text(-0.08, ax2_pos.y0-0.17,copyright_text, transform=ax.transAxes,fontsize=6, verticalalignment='top')
ax2.text(-0.08, ax2_pos.y0-0.195,footer_text, transform=ax.transAxes,fontsize=6, verticalalignment='top')
ax2.text(0.80, ax2_pos.y0-0.17,dataset_text, transform=ax.transAxes,fontsize=6, verticalalignment='top')

# Resize the logo to a larger size (e.g., 400x400)
logo = logo.resize((440, 150))

# Add logo to the plot relative to ax2's position
# Define position based on ax2_pos
ax_logo = fig.add_axes([0.07, ax2_pos.y1 - 0.001, 0.13, 0.15])  # Adjust the y-position slightly above ax2
ax_logo.imshow(logo)
ax_logo.axis('off')  

plt.savefig(output_filename, dpi=300,bbox_inches='tight', pad_inches=0.1) 