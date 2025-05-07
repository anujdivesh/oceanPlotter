#!/usr/bin/python3
#import cgi
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
#import cgi, cgitb
import io
import ssl
import warnings
import xarray as xr
import certifi
import urllib.request
from matplotlib.colors import BoundaryNorm
import pandas as pd
from owslib.wms import WebMapService

####FUNCTIONS
def getCountryData(region_url_prefix,country_id):
    # Fetch bounding box data from API
    api_url = "%s%s/" % (region_url_prefix,str(country_id))
    response = requests.get(api_url)
    west_bound, east_bound, south_bound, north_bound,country_name = "", "", "","",""
    name = ""
    if response.status_code == 200:
        data = response.json()
        name = data['short_name']
    else:
        print(f"Failed to retrieve bounding box data. Status code: {response.status_code}")
    if name == "PAC":
        name = "pacific_eez"
    eez_url = "https://opmgeoserver.gem.spc.int/geoserver/spc/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=spc:{layername}&srsName=EPSG:4326&outputFormat=application/json"
    formatted_url = eez_url.format(layername=name)
    return formatted_url, name

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

def getEEZ(ax,geojson_url):
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
                    ax.plot(x, y, marker=None, color='white', linewidth=1,linestyle='--')  # Plot the boundary line

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
"""
def getfromDAP(url, target_time, variable_name):
    try:
        # Open dataset with SSL verification
        with xr.open_dataset(url, engine='netcdf4') as ds:
            
            # Get available times (handle bytes if needed)
            if isinstance(ds.time.values[0], bytes):
                time_str = [t.decode('utf-8') for t in ds.time.values]
                time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
            else:
                time_dt = ds.time.values
                time_str = [str(t) for t in time_dt]
            
            # Find closest time
            target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
            time_index = abs(time_dt - target_dt).argmin()
            
            # Extract variable data
            if variable_name not in ds.variables:
                available_vars = list(ds.variables.keys())
                raise ValueError(f"Variable '{variable_name}' not found. Available variables: {available_vars}")
            
            data = ds[variable_name].isel(time=time_index)
            
            # Determine coordinate names
            coord_names = {
                'lon': ['lon', 'longitude', 'x', 'X'],
                'lat': ['lat', 'latitude', 'y', 'Y']
            }
            
            # Find longitude coordinate
            lon_name = None
            for possible_name in coord_names['lon']:
                if possible_name in ds.coords:
                    lon_name = possible_name
                    break
            if lon_name is None:
                raise ValueError("Could not identify longitude coordinate variable")
            
            # Find latitude coordinate
            lat_name = None
            for possible_name in coord_names['lat']:
                if possible_name in ds.coords:
                    lat_name = possible_name
                    break
            if lat_name is None:
                raise ValueError("Could not identify latitude coordinate variable")
            
            # Get coordinates
            lon = ds[lon_name].values
            lat = ds[lat_name].values
            
            # Prepare data values
            data_values = np.ma.masked_invalid(data.values.squeeze())
            
            return lon, lat, data_values
            
    except Exception as e:
        raise RuntimeError(f"Error accessing OpenDAP data: {str(e)}")


def getfromDAP(url, target_time, variable_name):
    try:
        # Open dataset with SSL verification
        with xr.open_dataset(url, engine='netcdf4') as ds:
            
            # Get available times (handle bytes if needed)
            if isinstance(ds.time.values[0], bytes):
                time_str = [t.decode('utf-8') for t in ds.time.values]
                time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
            else:
                # Convert numpy datetime64 to datetime objects if needed
                time_dt = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]
            
            # Convert target time to datetime object
            target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
            
            # Find closest time by comparing timestamps
            time_index = np.argmin([abs((t - target_dt).total_seconds()) for t in time_dt])
            
            # Extract variable data
            if variable_name not in ds.variables:
                available_vars = list(ds.variables.keys())
                raise ValueError(f"Variable '{variable_name}' not found. Available variables: {available_vars}")
            
            data = ds[variable_name].isel(time=time_index)
            
            # Determine coordinate names
            coord_names = {
                'lon': ['lon', 'longitude', 'x', 'X'],
                'lat': ['lat', 'latitude', 'y', 'Y']
            }
            
            # Find longitude coordinate
            lon_name = None
            for possible_name in coord_names['lon']:
                if possible_name in ds.coords:
                    lon_name = possible_name
                    break
            if lon_name is None:
                raise ValueError("Could not identify longitude coordinate variable")
            
            # Find latitude coordinate
            lat_name = None
            for possible_name in coord_names['lat']:
                if possible_name in ds.coords:
                    lat_name = possible_name
                    break
            if lat_name is None:
                raise ValueError("Could not identify latitude coordinate variable")
            
            # Get coordinates
            lon = ds[lon_name].values
            lat = ds[lat_name].values
            
            # Prepare data values
            data_values = np.ma.masked_invalid(data.values.squeeze())
            
            return lon, lat, data_values
            
    except Exception as e:
        raise RuntimeError(f"Error accessing OpenDAP data: {str(e)}")
"""
def getfromDAP(url, target_time, variable_name, adjust_lon=False):
    try:
        # Open dataset with SSL verification
        with xr.open_dataset(url, engine='netcdf4') as ds:
            
            # Get available times (handle bytes if needed)
            if isinstance(ds.time.values[0], bytes):
                time_str = [t.decode('utf-8') for t in ds.time.values]
                time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
            else:
                # Convert numpy datetime64 to datetime objects if needed
                time_dt = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]
            
            # Convert target time to datetime object
            target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
            
            # Find closest time by comparing timestamps
            time_index = np.argmin([abs((t - target_dt).total_seconds()) for t in time_dt])
            
            # Extract variable data
            if variable_name not in ds.variables:
                available_vars = list(ds.variables.keys())
                raise ValueError(f"Variable '{variable_name}' not found. Available variables: {available_vars}")
            
            data = ds[variable_name].isel(time=time_index)
            
            # Determine coordinate names
            coord_names = {
                'lon': ['lon', 'longitude', 'x', 'X'],
                'lat': ['lat', 'latitude', 'y', 'Y']
            }
            
            # Find longitude coordinate
            lon_name = None
            for possible_name in coord_names['lon']:
                if possible_name in ds.coords:
                    lon_name = possible_name
                    break
            if lon_name is None:
                raise ValueError("Could not identify longitude coordinate variable")
            
            # Find latitude coordinate
            lat_name = None
            for possible_name in coord_names['lat']:
                if possible_name in ds.coords:
                    lat_name = possible_name
                    break
            if lat_name is None:
                raise ValueError("Could not identify latitude coordinate variable")
            
            # Get coordinates
            lon = ds[lon_name].values
            lat = ds[lat_name].values
            
            # Adjust longitude if requested (for 180° crossing)
            if adjust_lon:
                if np.any(lon < 0):  # Only adjust if there are negative longitudes
                    lon = np.where(lon < 0, lon + 360, lon)
            
            # Prepare data values
            data_values = np.ma.masked_invalid(data.values.squeeze())
            
            return lon, lat, data_values
            
    except Exception as e:
        raise RuntimeError(f"Error accessing OpenDAP data: {str(e)}")

def add_z_if_needed(s):
    if len(s) == 0:
        return 'Z'  # or just return s if you want empty string to remain empty
    if s[-1] != 'Z':
        return s + 'Z'
    return s

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
        specific_timestemps = data.get('specific_timestemps')
        interval_step = data.get('interval_step')
        get_map_url = data.get('get_map_url')
        has_specific_timestep = data.get('has_specific_timestep')
        
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
            'get_map_names':get_map_names,
            'specific_timestemps':specific_timestemps,
            'interval_step':interval_step,
            'get_map_url':get_map_url,
            'has_specific_timestep':has_specific_timestep
        }
    else:
        # If request fails, return a message
        return {'error': f"Failed to fetch data. Status code: {response.status_code}"}


##########INPUT PARAMS#############
region = 1
layer_map = 5
time="2025-07-16T15:59:03Z"
resolution = "h"
coral = False
##########Input parameters##########

region_url_prefix = "https://ocean-middleware.spc.int/middleware/api/country/"
eez_country_url,name_of_layer = getCountryData("https://ocean-middleware.spc.int/middleware/api/country/",region)


##LOADING PARAMS#####
copyright_text = "© Pacific Community (SPC) 2025"
footer_text = "Climate and Ocean Support Program in the Pacific (COSPPac)"
layer_web_map_url = "https://ocean-middleware.spc.int/middleware/api/layer_web_map/%s/" % (layer_map)
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
date2 = date.strftime("%Y-%m-%dT%H%M%SZ")
formatted_date = date.strftime("%-d %B %Y")
formatted_date2 = date.strftime("%Y%m%d")
get_map_names = layer_data['get_map_names']
dap_url = layer_data['wms_url']
plotter_config = layer_data['get_map_url'].split("/")

##Plotter configurator
##AUTO TIME
if layer_data['has_specific_timestep']:
    spec = layer_data['specific_timestemps']
    specsplit = spec.split(',')
    time = specsplit[0]
else:
    time = layer_data['time_interval_end']

##REMOVEEE THIS
time = add_z_if_needed(time)
cmap_name = "jet"
plot_type = "contourf"
min_color_plot = 0
max_color_plot = 33
steps = 1
units = "m"
if plotter_config[0] == "custom_plot":
    cmap_name = plotter_config[2]
    plot_type = plotter_config[1]
    min_color_plot = float(plotter_config[3])
    max_color_plot = float(plotter_config[4])
    steps = float(plotter_config[5])
    units = plotter_config[6]
##END plotter configrator


if "wms" in dap_url:
    dap_url = dap_url.replace("wms", "dodsC")
dap_variable = layer_data['layer_name']
dap_time = time

new_name = []
week = False
if "{week}" in get_map_names:
    spec = layer_data['specific_timestemps']
    specsplit = spec.split(',')
    specsplit = [s.replace(" ", "") for s in specsplit]
    interval = layer_data['interval_step']
    intsplot = interval.split(',')

    cleaned_text = time.replace("Z", "")
    index = specsplit.index(cleaned_text)
    new_nametmp = get_map_names.replace("{week}", "%s Week"%(intsplot[index]))
    new_name = new_nametmp.split('/')
    week = True


title_suffix = "Daily Average Sea Surface Temperature Anomaly: %s" % (formatted_date)
dataset_text = "Reynolds SST"

if get_map_names != None or get_map_names != "":
    get_map_names = layer_data['get_map_names'].split('/')
    formatted_date = date.strftime(get_map_names[1])
    if week:
        title_suffix = "%s: %s" % (new_name[0],formatted_date)
    else:
        title_suffix = "%s: %s" % (get_map_names[0],formatted_date)
    dataset_text = get_map_names[2]


##LOADING PARAMS#####
west_bound, east_bound, south_bound, north_bound, country_name = getBBox(region)

bbox_offset = 1
if region == 1:
    bbox_offset = 20
    
figsize = cm2inch((18,13))
fig, ax = plt.subplots(figsize=figsize, dpi=300)
ax.axis('off')

ax2 = fig.add_axes([0.09, 0.2, 0.8, 0.65])
title = "%s \n %s" % (country_name,title_suffix)
ax2.set_title(title, pad=10, fontsize=8)

m = Basemap(projection='cyl', llcrnrlat=south_bound, urcrnrlat=north_bound, 
            llcrnrlon=west_bound, urcrnrlon=east_bound, resolution=resolution, ax=ax2)

m.drawcoastlines(linewidth=0.3)
m.fillcontinents(color='#A9A9A9', lake_color='white')
m.drawcountries()

num_longitudes = east_bound - west_bound + 1

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

# Add colorbar to ax2
ax2_pos = ax2.get_position()
ax_legend_width = 0.03  # Width of the legend
ax_legend_gap = 0.1    # Gap between ax2 and ax_legend
ax_legend = fig.add_axes([ax2_pos.x1 +0.02, ax2_pos.y0, ax_legend_width, ax2_pos.height])

lon, lat, sst_anomaly = getfromDAP(dap_url, dap_time, dap_variable,adjust_lon=True)
# Plot colorbar in the legend axes
levels = np.arange(min_color_plot, max_color_plot, steps)  # Ensures 2.0 is included

# Plot filled contours with fixed levels
cs = ax2.contourf(
    lon, lat, sst_anomaly,
    levels=levels,  # Use predefined levels
    cmap=cmap_name,
    extend='both'    # Adds arrows if data exceeds [-2, 2]
)

# Add colorbar with matching ticks
cbar = plt.colorbar(cs, cax=ax_legend)
cbar.set_ticks(levels)  # Same ticks as contour levels
cbar.ax.tick_params(labelsize=7)
cbar.set_label(
    '(°C)',
    fontsize=6,
    rotation=0,
    va='center',  # Vertical alignment (centered)
    ha='left',    # Horizontal alignment (left-justified)
    labelpad=1    # Space between label and colorbar
)

if region == 1:
    ##ADD LOGO
    logo_url = "./Logo_cropped.png" 
    logo_img = Image.open(logo_url)
    logo_ax = fig.add_axes([0.08, 0.85, 0.12, 0.12])  # [left, bottom, width, height]
    logo_ax.imshow(logo_img)
    logo_ax.axis('off')
    # Add footer elements
    ax2.text(-0.08, ax2_pos.y0-0.20, copyright_text,  # Increased from -0.17 to -0.20
            transform=ax.transAxes, fontsize=6, verticalalignment='top')
    ax2.text(-0.08, ax2_pos.y0-0.225, footer_text,  # Increased from -0.195 to -0.225
            transform=ax.transAxes, fontsize=6, verticalalignment='top')
    ax2.text(0.90, ax2_pos.y0-0.20, dataset_text,  # Increased from -0.17 to -0.20
            transform=ax.transAxes, fontsize=6, verticalalignment='top', ha='right')

else:
    ##ADD LOGO
    logo_url = "./Logo_cropped.png" 
    logo_img = Image.open(logo_url)
    logo_ax = fig.add_axes([0.12, 0.85, 0.12, 0.12])  # [left, bottom, width, height]
    logo_ax.imshow(logo_img)
    logo_ax.axis('off')

    # Add footer elements
    ax2.text(0.05, ax2_pos.y0-0.20, copyright_text,  # Increased from -0.17 to -0.20
            transform=ax.transAxes, fontsize=6, verticalalignment='top')
    ax2.text(0.05, ax2_pos.y0-0.225, footer_text,  # Increased from -0.195 to -0.225
            transform=ax.transAxes, fontsize=6, verticalalignment='top')
    ax2.text(0.90, ax2_pos.y0-0.20, dataset_text,  # Increased from -0.17 to -0.20
            transform=ax.transAxes, fontsize=6, verticalalignment='top', ha='right')

    plt.subplots_adjust(bottom=0.15)  
    plt.subplots_adjust(bottom=0.15)  

#plot eez
getEEZ(ax2,eez_country_url)

plt.savefig('anuj2.png', bbox_inches='tight', pad_inches=0.1)