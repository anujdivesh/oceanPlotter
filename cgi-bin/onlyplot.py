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

def add_z_if_needed(s):
    if len(s) == 0:
        return 'Z'  # or just return s if you want empty string to remain empty
    if s[-1] != 'Z':
        return s + 'Z'
    return s

##########INPUT PARAMS#############
region = 3
layer_map = 2
time="2025-07-16T15:59:03Z"
resolution = "h"
coral = False
##########Input parameters##########

region_url_prefix = "https://ocean-middleware.spc.int/middleware/api/country/"


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

"""
##LOADING PARAMS#####
#lon, lat, chrolophyll = getfromDAP(dap_url, dap_time, dap_variable)
#print(chrolophyll)

# Create the figure and axis
# Create the figure and axis
# Create the figure and axis
# Define your bounds (example values - adjust as needed)
south_bound = -40  # -40°S
north_bound = 40    # 40°N
west_bound = 120    # 120°E
east_bound = 260    # 260°E (equivalent to -100°W)

# Get data with longitude adjustment for 180° crossing
target_time = dap_time
variable_name = dap_variable
lon, lat, chl = getfromDAP(dap_url, dap_time, dap_variable, adjust_lon=True)

# Create plot
plt.figure(figsize=(15, 8))
ax = plt.gca()

# Set up basemap with cylindrical projection
m = Basemap(projection='cyl',
            llcrnrlat=south_bound, urcrnrlat=north_bound,
            llcrnrlon=west_bound, urcrnrlon=east_bound,
            resolution='i', ax=ax)  # 'i' for intermediate resolution

# Draw map features
m.drawcoastlines()
m.fillcontinents(color='lightgray', lake_color='lightblue')

# Draw parallels and meridians
parallels = np.arange(-90, 91, 10)
m.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

# Handle meridian labels properly across 180°
if east_bound > 180:
    meridians = np.unique(np.concatenate([
        np.arange(west_bound, 180, 20),
        np.arange(-180, east_bound-360, 20)
    ]))
else:
    meridians = np.arange(west_bound, east_bound, 20)
    
m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10)

# Create grid and plot
x, y = m(*np.meshgrid(lon, lat))
cs = m.pcolormesh(x, y, chl, cmap='viridis', shading='auto')

# Add colorbar
cbar = m.colorbar(cs, location='bottom', pad=0.1, fraction=0.05)
cbar.set_label('Chlorophyll Concentration (mg/m³)', fontsize=12)

# Add title and info
plt.title(f'Ocean Chlorophyll - {datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ").strftime("%d %B %Y")}',
            pad=20, fontsize=14)
plt.figtext(0.5, 0.01, "Data Source: Your Data Provider", ha='center', fontsize=10)

# Save or display
plt.tight_layout()
plt.show()
"""

# Get wave height data
lon, lat, wave_height = getfromDAP(dap_url, dap_time, "sig_wav_ht", adjust_lon=True)

# Get wave direction data (convert to radians for quiver plot)
_, _, wave_dir = getfromDAP(dap_url, dap_time, "mn_wav_dir", adjust_lon=True)
wave_dir_rad = np.radians(wave_dir)  # Convert degrees to radians

# Calculate U and V components for arrows
# We'll scale the arrows by wave height for visualization
u = wave_height * np.sin(wave_dir_rad)  # Eastward component
v = wave_height * np.cos(wave_dir_rad)  # Northward component

# Define plot bounds
south_bound = -40
north_bound = 40
west_bound = 120
east_bound = 260

# Create plot
plt.figure(figsize=(15, 8))
ax = plt.gca()

# Set up basemap
m = Basemap(projection='cyl',
            llcrnrlat=south_bound, urcrnrlat=north_bound,
            llcrnrlon=west_bound, urcrnrlon=east_bound,
            resolution='i', ax=ax)

# Draw map features
m.drawcoastlines()
m.fillcontinents(color='lightgray', lake_color='lightblue')

# Draw parallels and meridians
parallels = np.arange(-90, 91, 10)
m.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

meridians = np.unique(np.concatenate([
    np.arange(west_bound, 180, 20),
    np.arange(-180, east_bound-360, 20)
])) if east_bound > 180 else np.arange(west_bound, east_bound, 20)
m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10)

# Create grid for plotting
x, y = m(*np.meshgrid(lon, lat))

# Plot wave height as colormap
cs = m.pcolormesh(x, y, wave_height, cmap='jet', shading='auto')

# Add colorbar for wave height
cbar = m.colorbar(cs, location='bottom', pad=0.1, fraction=0.05)
cbar.set_label('Significant Wave Height (m)', fontsize=12)

# Plot wave direction as arrows
# We'll plot every 5th point to avoid overcrowding
# Prepare quiver plot for wave direction
step = 30  # Show fewer arrows
scale = 30  # Larger value = smaller arrows
arrow_scale = 0.5  # Scales arrows with wave height

# Convert direction to radians
theta = np.radians(wave_dir)
u = arrow_scale * np.sin(theta)  # East component
v = arrow_scale * np.cos(theta)  # North component

# Plot quiver with black arrows
q = m.quiver(x[::step, ::step], y[::step, ::step], 
             u[::step, ::step], v[::step, ::step],
             scale=scale, width=0.003, 
             headwidth=2.5, headlength=3, headaxislength=2.5,
             color='black', pivot='middle', minshaft=2,
             edgecolor='black', linewidth=0.3)

# Add subtle quiver key
qk = plt.quiverkey(q, 0.82, 0.12, 1, 
                  'Wave Direction', labelpos='E',
                  coordinates='axes', fontproperties={'size': 9},
                  labelsep=0.05, labelcolor='black')



# Add title and info
plt.title(f'Wave Height and Direction - {datetime.strptime(dap_time, "%Y-%m-%dT%H:%M:%SZ").strftime("%d %B %Y")}',
          pad=20, fontsize=14)
plt.figtext(0.5, 0.01, "Data Source: Your Data Provider", ha='center', fontsize=10)

# Save or display
plt.tight_layout()
plt.show()