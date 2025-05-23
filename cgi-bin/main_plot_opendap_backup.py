#!/usr/bin/python3
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
import io
import xarray as xr
from matplotlib.colors import BoundaryNorm
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import cgi, cgitb
form = cgi.FieldStorage() 
region_api = form.getvalue('region')
layer_map_api = form.getvalue('layer_map')
time_api = form.getvalue('time')


#####FUNCTIONS#####
def fetch_wms_layer_data(layer_id):
    try:
        url_tmp = "https://ocean-middleware.spc.int/middleware/api/layer_web_map/{layerid}/"
        url = url_tmp.format(layerid=layer_id)
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data_dict = response.json()
        
        # Convert the dictionary to an object with attribute access
        class DataObject:
            def __init__(self, data):
                self.__dict__.update(data)
        
        data = DataObject(data_dict)
        return data
    except requests.exceptions.RequestException as e:
        return None
"""
def getfromDAP(url, target_time, variable_name, adjust_lon=False):
    try:
        # Open dataset with SSL verification
        with xr.open_dataset(url, engine='netcdf4', mask_and_scale=True,decode_cf=True) as ds:
            
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
            
            if len(data.dims) == 3:
                data = data.isel({data.dims[0]: 0})
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
"""
def getfromDAP(url, target_time, variable_name, adjust_lon=False):
    try:
        # Open dataset with SSL verification
        with xr.open_dataset(url, engine='netcdf4', mask_and_scale=True, decode_cf=True) as ds:
            
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
            
            # If variable has 3 dimensions (e.g., depth), select first depth level
            if len(data.dims) == 3:
                data = data.isel({data.dims[0]: 0})  # Select first index of first dimension
            
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
            
            # Extract and mask data values
            data_extract = np.ma.masked_invalid(data.values.squeeze())
            
            return lon, lat, data_extract
            
    except Exception as e:
        raise RuntimeError(f"Error accessing OpenDAP data: {str(e)}")

def getCountryData(country_id):
    # Fetch bounding box data from API
    region_url_prefix = "https://ocean-middleware.spc.int/middleware/api/country/"
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
    return formatted_url

def getBBox(country_id):
    region_url_prefix = "https://ocean-middleware.spc.int/middleware/api/country/"
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
                    ax.plot(x, y, marker=None, color='black', linewidth=0.5,linestyle='--')  # Plot the boundary line

    else:
        print("Failed to retrieve the GeoJSON data.")

def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple()

def add_z_if_needed(s):
    if len(s) == 0:
        return 'Z'  # or just return s if you want empty string to remain empty
    if s[-1] != 'Z':
        return s + 'Z'
    return s

def get_config_variables():
    config = {
        
        # Additional configuration variables
        "copyright_text": "© Pacific Community (SPC) 2025",
        "footer_text": "Climate and Ocean Support Program in the Pacific (COSPPac)",
        "app_name": "Ocean Data Viewer",
        "version": "1.2.0",
        "default_theme": "light",
        "max_upload_size": 10,  # in MB
        "supported_formats": ["geojson", "shapefile", "netcdf", "csv"]
    }
    
    # Convert the dictionary to an object with dot notation access
    class ConfigObject:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigObject(value))
                else:
                    setattr(self, key, value)
    
    return ConfigObject(config)

def demo_time(layer_map_data):
    try:
        if layer_map_data.has_specific_timestep:
            spec = layer_map_data.specific_timestemps
            specsplit = spec.split(',')
            time = specsplit[0]
        else:
            time = layer_map_data.timeIntervalEnd
    except Exception as e:
        time = layer_map_data.timeIntervalEnd
    time2 = add_z_if_needed(time)
    return time2

def get_dap_config(layer_map_data):
    dap_url = layer_map_data.url.replace("wms", "dodsC")
    dap_variable = layer_map_data.layer_name
    dapvaribsplit = dap_variable.split(',')
    #if len(dapvaribsplit) >= 1:
    #    dap_variable = dapvaribsplit[0]
    return dap_url, dap_variable

def get_title(layer_map_data,time):
    new_name = []
    week = False
    date = datetime.strptime(add_z_if_needed(time), "%Y-%m-%dT%H:%M:%SZ")
    date2 = date.strftime("%Y-%m-%dT%H%M%SZ")
    formatted_date = date.strftime("%-d %B %Y")
    if "{week}" in layer_map_data.get_map_names:
        spec = layer_map_data.specific_timestemps
        specsplit = spec.split(',')
        specsplit = [s.replace(" ", "") for s in specsplit]
        interval = layer_map_data.interval_step
        intsplot = interval.split(',')

        cleaned_text = time.replace("Z", "")
        index = specsplit.index(cleaned_text)
        new_nametmp = layer_map_data.get_map_names.replace("{week}", "%s Week"%(intsplot[index]))
        new_name = new_nametmp.split('/')
        week = True

    title_suffix = "Daily Average Sea Surface Temperature Anomaly: %s" % (formatted_date)
    dataset_text = "Reynolds SST"

    if layer_map_data.get_map_names != None or layer_map_data.get_map_names != "":
        layer_map_data.get_map_names = layer_map_data.get_map_names.split('/')
        formatted_date = date.strftime(layer_map_data.get_map_names[1])
        if week:
            title_suffix = "%s: %s" % (new_name[0],formatted_date)
        else:
            title_suffix = "%s: %s" % (layer_map_data.get_map_names[0],formatted_date)
        dataset_text = layer_map_data.get_map_names[2]

    return title_suffix, dataset_text

def get_plot_config(layer_map_data):
    plotter_config = layer_map_data.get_map_url.split("/")
    plot_type = plotter_config[0]
    cmap_name = plotter_config[1]
    min_color_plot = float(plotter_config[2])
    max_color_plot = float(plotter_config[3])
    steps = float(plotter_config[4])
    units = plotter_config[5]
    levels = []
    if plotter_config[6] != "null":
        levels = np.array(eval(plotter_config[6]), dtype=float)
    discrete = plotter_config[7]
    
    return cmap_name, plot_type, min_color_plot,max_color_plot,steps,units,levels,discrete

def plot_filled_contours(ax, ax_legend, lon, lat, data, 
                        min_color_plot, max_color_plot, steps,
                        cmap_name='RdBu_r', units='(°C)'):
    # Create fixed levels for contours
    levels = np.arange(min_color_plot, max_color_plot, steps)
    
    # Plot filled contours with fixed levels
    cs = ax.contourf(
        lon, lat, data,
        levels=levels,
        cmap=cmap_name,
        extend='both'  # Adds arrows if data exceeds min/max
    )
    
    # Add colorbar with matching ticks
    cbar = plt.colorbar(cs, cax=ax_legend)
    cbar.set_ticks(levels)  # Same ticks as contour levels
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(
        units,
        fontsize=6,
        rotation=0,
        va='center',
        ha='left',
        labelpad=1
    )
    
    return cs, cbar

def plot_filled_pcolor(ax, ax_legend, lon, lat, data, 
                min_color_plot, max_color_plot, steps,
                cmap_name='RdBu_r', units='(°C)'):
    # Create fixed levels for color normalization
    levels = np.arange(min_color_plot, max_color_plot, steps)
    
    # Create a BoundaryNorm to discretize the colorbar
    norm = BoundaryNorm(levels, ncolors=256, clip=True)
    
    # Plot pcolor with fixed levels
    pc = ax.pcolormesh(
        lon, lat, data,
        norm=norm,
        cmap=cmap_name,
        shading='auto'  # Can be 'nearest', 'flat', 'auto', etc.
    )
    
    # Add colorbar with matching ticks
    cbar = plt.colorbar(pc, cax=ax_legend)
    cbar.set_ticks(levels)  # Same ticks as contour levels
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(
        units,
        fontsize=6,
        rotation=0,
        va='center',
        ha='left',
        labelpad=1
    )
    
    return pc, cbar

def plot_wave_field(ax, ax_legend, m, lon, lat, wave_height, wave_dir,
                   min_color_plot, max_color_plot, steps,region,step,
                   cmap_name='jet', units='m',
                   scale=30, arrow_scale=0.5):
    
    # Convert wave direction to components
    wave_dir_rad = np.radians(wave_dir)
    u = wave_height * np.sin(wave_dir_rad)  # Eastward component
    v = wave_height * np.cos(wave_dir_rad)  # Northward component
    
    # Create grid coordinates
    x, y = m(*np.meshgrid(lon, lat))
    
    # Create levels and normalization
    levels = np.arange(min_color_plot, max_color_plot, steps)
    norm = BoundaryNorm(levels, ncolors=256)
    
    # Plot wave height field
    cs = ax.pcolormesh(
        x, y, wave_height,
        cmap=cmap_name,
        norm=norm,
        shading='auto'
    )
    
    # Create colorbar in specified legend axes
    cbar = plt.colorbar(cs, cax=ax_legend)
    cbar.set_label(
        units,
        fontsize=8,
        rotation=0,
        va='center',
        ha='left',
        labelpad=1
    )
    
    # Create directional arrows
    theta = np.radians(wave_dir)
    u_arrows = arrow_scale * np.sin(theta)
    v_arrows = arrow_scale * np.cos(theta)
    
    q = ax.quiver(x[::step, ::step], y[::step, ::step], 
                u_arrows[::step, ::step], v_arrows[::step, ::step],
                scale=scale, width=0.003, 
                headwidth=2.5, headlength=3, headaxislength=2.5,
                color='black', pivot='middle', minshaft=2,
                edgecolor='black', linewidth=0.3)
    
    # Add quiver key (without text)
    qk = plt.quiverkey(q, 0.82, 0.12, 1, 
                     '', labelpos='E',
                     coordinates='axes', fontproperties={'size': 9},
                     labelsep=0.05, labelcolor='black')
    cbar.ax.tick_params(labelsize=6)
    
    return cs, q, cbar

def plot_discrete_map(ax, ax_legend, lons, lats, bleaching_data, 
                        cmap_colors=None, colorbar_labels=None):
  
    # Validate inputs
    if bleaching_data.ndim != 2:
        raise ValueError("bleaching_data must be a 2D array")
    
    if len(cmap_colors) != len(colorbar_labels):
        raise ValueError("Number of colors must match number of labels")
    
    try:
        # Calculate bounds automatically based on number of categories
        n_categories = len(cmap_colors)
        bounds = np.arange(n_categories + 1)  # [0, 1, 2, ..., n_categories]
        
        # Calculate tick positions (middle of each color band)
        ticks = bounds[:-1] + 0.5
        
        # Create colormap and normalization
        cmap = mcolors.ListedColormap(cmap_colors)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Create plot
        cs = ax.pcolormesh(lons, lats, bleaching_data, 
                          cmap=cmap, norm=norm, 
                          shading='auto')
        
        # Create colorbar
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(colorbar_labels)
        cbar.ax.tick_params(labelsize=6)
        
        return cs, cbar
        
    except Exception as e:
        raise RuntimeError(f"Error plotting coral bleaching data: {str(e)}")

def add_logo_and_footer(fig, ax, ax2, ax2_pos, region, 
                       copyright_text, footer_text, dataset_text,
                       logo_path="./Logo_cropped.png"):
    # Add logo
    """
    try:
        logo_img = Image.open(logo_path)
        if region == 1:
            logo_ax = fig.add_axes([0.08, 0.85, 0.12, 0.12])  # [left, bottom, width, height]
        else:
            logo_ax = fig.add_axes([0.12, 0.85, 0.12, 0.12])
        
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')
    except FileNotFoundError:
        print(f"Logo file not found at {logo_path}")
    """

    # Add footer elements
    if region == 1:
        ax2.text(-0.08, ax2_pos.y0-0.20, copyright_text,
                transform=ax.transAxes, fontsize=6, verticalalignment='top')
        """
        ax2.text(-0.08, ax2_pos.y0-0.225, footer_text,
                transform=ax.transAxes, fontsize=6, verticalalignment='top')
        """
    else:
        ax2.text(0.05, ax2_pos.y0-0.20, copyright_text,
                transform=ax.transAxes, fontsize=6, verticalalignment='top')
        """
        ax2.text(0.05, ax2_pos.y0-0.225, footer_text,
                transform=ax.transAxes, fontsize=6, verticalalignment='top')
        """
    
    # Common dataset text placement
    ax2.text(0.90, ax2_pos.y0-0.20, dataset_text,
            transform=ax.transAxes, fontsize=6, verticalalignment='top', ha='right')

    # Adjust subplots
    plt.subplots_adjust(bottom=0.15)

def plot_levels_pcolor(ax, ax_legend, lons, lats, chl_data,cmap_name='jet', units='mg/m³',levels=[]):
    # Clip data to level boundaries
    chl_clipped = np.clip(chl_data, levels[0], levels[-1])
    
    # Create colormap with one less color than levels
    cmap = plt.get_cmap(cmap_name, len(levels)-1)
    
    # Create normalization with extend to handle out-of-range values
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    
    # Plot with discrete levels
    cs = ax.pcolormesh(lons, lats, chl_clipped,
                      cmap=cmap,
                      norm=norm,
                      shading='auto')
    
    # Create colorbar with exact level ticks
    cbar = plt.colorbar(cs, cax=ax_legend, extend='both')
    cbar.set_ticks(levels)  # Set ticks exactly at level boundaries
    cbar.set_ticklabels([f"{x:.2f}" for x in levels])
    cbar.set_label(units, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    return cs, cbar

def plot_levels_contour(ax, ax_legend, lons, lats, chl_data, cmap_name='jet',
                          units='mg/m³', levels=None, add_contours=True,
                          contour_kwargs=None):
    
    # Default contour style
    if contour_kwargs is None:
        contour_kwargs = {
            'colors': 'k',  # Black contour lines
            'linewidths': 0.5,
            'linestyles': 'solid',
            'alpha': 0.5  # Semi-transparent
        }
    
    # Clip data to level boundaries
    chl_clipped = np.clip(chl_data, levels[0], levels[-1])
    
    # Create colormap and normalization
    cmap = plt.get_cmap(cmap_name, len(levels)-1)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    
    # Plot filled colors
    mesh = ax.pcolormesh(lons, lats, chl_clipped,
                        cmap=cmap,
                        norm=norm,
                        shading='auto')
    
    # Add contour lines if requested
    if add_contours:
        # Create 2D grid if needed (for contour)
        if lons.ndim == 1 or lats.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            lon_grid, lat_grid = lons, lats
            
        contours = ax.contour(lon_grid, lat_grid, chl_clipped,
                            levels=levels,
                            **contour_kwargs)
        
        # Optionally add contour labels
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    # Create colorbar
    cbar = plt.colorbar(mesh, cax=ax_legend, extend='both')
    cbar.set_ticks(levels)
    cbar.set_ticklabels([f"{x:.2f}" for x in levels])
    cbar.set_label(units, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    return mesh, cbar

def plot_map_grid(m, south_bound, north_bound, west_bound, east_bound):
    """
    Draw latitude/longitude grid lines with:
    - 6 latitude divisions
    - 5 longitude divisions
    - Labels only on left (lat) and bottom (lon)
    - Clean single degree symbols
    """
    # Calculate divisions
    parallels = np.linspace(south_bound, north_bound, 7)  # 6 divisions
    meridians = np.linspace(west_bound, east_bound, 6)    # 5 divisions
    
    # Draw latitude lines (left only) with simple numeric format
    m.drawparallels(parallels,
                   labels=[1,0,0,0],
                   fmt='%.0f',  # Simple numeric format
                   fontsize=6,
                   color='grey',
                   linewidth=0.5,
                   dashes=[1,1])
    
    # Draw longitude lines (bottom only) with simple numeric format
    m.drawmeridians(meridians,
                   labels=[0,0,0,1],
                   fmt='%.0f',  # Simple numeric format
                   fontsize=6,
                   color='grey',
                   linewidth=0.5,
                   dashes=[1,1])
    
    # Add degree symbols manually by modifying the text objects
    ax = plt.gca()
    
    # For latitude labels (left side)
    for text in ax.yaxis.get_ticklabels():
        text.set_text(text.get_text() + '°')
    
    # For longitude labels (bottom side)
    for text in ax.xaxis.get_ticklabels():
        text.set_text(text.get_text() + '°')
    
    # Add finer grid lines without labels
    m.drawparallels(np.linspace(south_bound, north_bound, 13),
                   labels=[0,0,0,0],
                   color='lightgrey',
                   linewidth=0.2,
                   dashes=[1,1])
    
    m.drawmeridians(np.linspace(west_bound, east_bound, 11),
                   labels=[0,0,0,0],
                   color='lightgrey',
                   linewidth=0.2,
                   dashes=[1,1])

def plot_filled_contours_no_zero(ax, ax_legend, lon, lat, data, 
                        min_color_plot, max_color_plot, steps,
                        cmap_name='RdBu_r', units='(°C)'):

    # Create fixed levels for contours, excluding zero
    levels = np.arange(min_color_plot, max_color_plot, steps)
    levels = levels[levels != 0]  # Remove zero level
    
    # Plot filled contours with fixed levels
    cs = ax.contourf(
        lon, lat, data,
        levels=levels,
        cmap=cmap_name,
        extend='both'  # Adds arrows if data exceeds min/max
    )
    
    # Add colorbar with matching ticks
    cbar = plt.colorbar(cs, cax=ax_legend)
    cbar.set_ticks(levels)  # Same ticks as contour levels
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(
        units,
        fontsize=6,
        rotation=0,
        va='center',
        ha='left',
        labelpad=1
    )
    
    return cs, cbar

def plot_discrete_map_ranges(ax, ax_legend, lons, lats, bleaching_data,
                           cmap_colors=None, colorbar_labels=None, ranges=None):
    """
    Final working version that properly handles:
    - Discontinuous ranges (like 2-3, 4-7)
    - Exactly matches 7 colors to 7 range segments
    - Maintains your original range definitions
    """
    # Validate inputs
    if bleaching_data.ndim != 2:
        raise ValueError("bleaching_data must be a 2D array")
    
    if len(cmap_colors) != len(colorbar_labels) or len(cmap_colors) != len(ranges):
        raise ValueError("Number of colors, labels, and ranges must match")

    try:
        # Create segments from ranges
        segments = []
        for r in ranges:
            if '-' in r:
                start, end = map(float, r.split('-'))
                segments.append((start, end))
            else:
                val = float(r)
                segments.append((val, val))  # Treat single values as range with equal start/end
        
        # Create colormap with exactly N colors for N segments
        cmap = mcolors.ListedColormap(cmap_colors)
        
        # Create normalization that maps each segment to one color
        # We'll use the midpoint of each segment to determine color mapping
        norm = mcolors.Normalize(vmin=min(s[0] for s in segments), 
                               vmax=max(s[1] for s in segments))
        
        # Create plot - we'll manually map values to colors
        # First, create an array where each value is mapped to its segment index
        segment_idx = np.zeros_like(bleaching_data, dtype=int)
        for i, (start, end) in enumerate(segments):
            mask = (bleaching_data >= start) & (bleaching_data <= end)
            segment_idx[mask] = i
        
        # Now plot using the segment indices
        cs = ax.pcolormesh(lons, lats, segment_idx,
                          cmap=cmap, 
                          vmin=0, vmax=len(segments)-1,
                          shading='auto')
        
        # Calculate midpoints for ticks
        ticks = [(seg[0] + seg[1])/2 for seg in segments]
        
        # Create colorbar
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_ticks(np.arange(len(segments)))  # One tick per segment
        cbar.set_ticklabels(colorbar_labels)
        cbar.ax.tick_params(labelsize=6)
        
        # Adjust label rotation if needed
        for label in cbar.ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        
        return cs, cbar
        
    except Exception as e:
        raise RuntimeError(f"Error plotting discrete map: {str(e)}")

def plot_climatology(dap_url, time, ax, ax_legend, lon, lat, data, 
                        min_color_plot, max_color_plot, steps,
                        cmap_name='RdBu_r', units='(°C)'):
    # Create fixed levels for contours
    levels = np.arange(min_color_plot, max_color_plot, steps)
    
    # Plot filled contours with fixed levels
    cs = ax.contourf(
        lon, lat, data,
        levels=levels,
        cmap=cmap_name,
        extend='both'  # Adds arrows if data exceeds min/max
    )
    contour_29 = ax.contour(
        lon, lat, data,
        levels=[29],
        colors='purple',
        linewidths=2,
        linestyles='solid',
        zorder=5,
        label=f'(SST))'
    )
    clim_lon, clim_lat, sst_clim = getfromDAP(dap_url, time, "sst_clim",adjust_lon=True)
    
    contour_clim = ax.contour(
            clim_lon, clim_lat, sst_clim,
            levels=[29],
            colors='green',
            linewidths=2,
            linestyles='solid',
            zorder=6,
            label='Climatology'
        )
    # Optional: Add labels to the contour line
    #ax.clabel(contour_29, inline=True, fontsize=8, fmt='%1.0f')
    legend_elements = [
        Line2D([0], [0], color='purple', lw=1, label=f'SST Forecast'),
        Line2D([0], [0], color='green', lw=1, label='Climatology')
    ]

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6)

    # Add colorbar with matching ticks
    cbar = plt.colorbar(cs, cax=ax_legend)
    cbar.set_ticks(levels)  # Same ticks as contour levels
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(
        units,
        fontsize=6,
        rotation=0,
        va='center',
        ha='left',
        labelpad=1
    )
    
    return cs, cbar

#####FUNCTIONS#####
##INIT##
config = get_config_variables()

#####PARAMETER#####
region = region_api
layer_id = layer_map_api
time= add_z_if_needed(time_api)
resolution = "h"
#####PARAMETER#####

#LOAD LAYER WEB MAP
layer_map_data = fetch_wms_layer_data(layer_id)

#OUTPUT FILE
date = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
date2 = date.strftime("%Y_%m_%dT%H%M%SZ")
layers_r = layer_map_data.layer_name
first_layer = layers_r.split(',')[0].strip()
output_filename = "./maps/%s_%s_%s_%s.png" % (first_layer,layer_id,region,date2)

if os.path.exists(output_filename):
    length = os.stat(output_filename).st_size
    sys.stdout.write("Content-Type: image/png\n")
    sys.stdout.write("Content-Length: " + str(length) + "\n")
    sys.stdout.write("\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(open(output_filename, "rb").read())
else:
    #REMOVE DEMO
    #time = demo_time(layer_map_data)
    #REMOVE DEMO

    #####MAIN#####
    dap_url, dap_variable = get_dap_config(layer_map_data)
    title, dataset_text = get_title(layer_map_data,time)
    cmap_name, plot_type, min_color_plot, max_color_plot, steps, units, levels, discrete = get_plot_config(layer_map_data)
    west_bound, east_bound, south_bound, north_bound, country_name = getBBox(region)
    eez_url = getCountryData(region)

    #MAPPING
    bbox_offset = 1
    if region == 1:
        bbox_offset = 20

    figsize = cm2inch((18,13))
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.axis('off')

    ax2 = fig.add_axes([0.09, 0.2, 0.8, 0.65])
    title = "%s \n %s" % (country_name,title)
    ax2.set_title(title, pad=10, fontsize=8)

    m = Basemap(projection='cyl', llcrnrlat=south_bound, urcrnrlat=north_bound, 
                llcrnrlon=west_bound, urcrnrlon=east_bound, resolution=resolution, ax=ax2)

    plot_map_grid(m, south_bound, north_bound, west_bound, east_bound)

    # Add colorbar to ax2
    ax2_pos = ax2.get_position()
    ax_legend_width = 0.03  # Width of the legend
    ax_legend_gap = 0.1    # Gap between ax2 and ax_legend
    ax_legend = fig.add_axes([ax2_pos.x1 +0.02, ax2_pos.y0, ax_legend_width, ax2_pos.height])

    ##MAIN PLOTTER
    if plot_type == "contourf":
        lon, lat, data_extract = getfromDAP(dap_url, time, dap_variable,adjust_lon=True)
        cs, cbar = plot_filled_contours(ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
            min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units
        )
    elif plot_type == "contourf_nozero":
        lon, lat, data_extract = getfromDAP(dap_url, time, dap_variable,adjust_lon=True)
        cs, cbar = plot_filled_contours_no_zero(ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
            min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units
        )
    elif plot_type == "pcolormesh":
        lon, lat, data_extract = getfromDAP(dap_url, time, dap_variable,adjust_lon=True)
        cs, cbar = plot_filled_pcolor(ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
            min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units
        )
    elif plot_type == "wave_with_dir":
        wave_height_varib, wave_dir_varib = dap_variable.split(',')
        lon, lat, wave_height = getfromDAP(dap_url, time, wave_height_varib, adjust_lon=True)
        _, _, wave_dir = getfromDAP(dap_url, time, wave_dir_varib, adjust_lon=True)
        step = 10
        if int(region) == 1:
            step = 30
        cs, q, cbar = plot_wave_field(ax2, ax_legend, m, lon, lat, wave_height, wave_dir,\
                                min_color_plot, max_color_plot, steps,region, step, cmap_name=cmap_name, units=units)
    elif plot_type == "discrete":
        lons, lats, bleaching_data = getfromDAP(dap_url, time, dap_variable, adjust_lon=True)
        splitBy_ = discrete.split("_")
        if len(splitBy_) > 1:
            colors = splitBy_[0]
            split_1 = splitBy_[1]
            range_nums, range_name = split_1.split('%')
            color_arr = np.array(eval(colors), dtype=str)
            range_nums_arr = np.array(eval(range_nums), dtype=str)
            range_name_arr = np.array(eval(range_name), dtype=str)

            cs, cbar = plot_discrete_map_ranges(ax=ax2, ax_legend=ax_legend, lons=lons, lats=lats, bleaching_data=bleaching_data,\
                cmap_colors=color_arr, colorbar_labels=range_name_arr, ranges=range_nums_arr)
        else:
            tmp_color, tmp_label = discrete.split('-')
            color_arr = np.array(eval(tmp_color), dtype=str)
            label_arr = np.array(eval(tmp_label), dtype=str)

            cs, cbar = plot_discrete_map(ax=ax2, ax_legend=ax_legend, lons=lons, lats=lats, bleaching_data=bleaching_data,\
                cmap_colors=color_arr, colorbar_labels=label_arr)

    elif plot_type == "levels_pcolor":
        lons, lats, chl_data = getfromDAP(dap_url, time, dap_variable, adjust_lon=True)
        plot_levels_pcolor(ax2, ax_legend, lons, lats, chl_data,cmap_name, units=units,levels=levels)

    elif plot_type == "levels_contourf":
        lons, lats, chl_data = getfromDAP(dap_url, time, dap_variable, adjust_lon=True)
        plot_levels_contour(ax2, ax_legend, lons, lats, chl_data,cmap_name, units=units,levels=levels,)

    elif plot_type == "climate":
        lon, lat, data_extract = getfromDAP(dap_url, time, dap_variable,adjust_lon=True)
        cs, cbar = plot_climatology(dap_url,time,ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
            min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units
        )

    #ADD LOGO AND FOOTER
    add_logo_and_footer(fig=fig, ax=ax, ax2=ax2, ax2_pos=ax2_pos, region=1, copyright_text=config.copyright_text,\
        footer_text=config.footer_text,dataset_text=dataset_text)

    #PLOT EEZ
    getEEZ(ax2,eez_url)

    #FILL COASTLINES
    m.drawcoastlines(linewidth=0.3)
    m.fillcontinents(color='#A9A9A9', lake_color='white')
    m.drawcountries()

    plt.savefig(output_filename,  dpi=300, bbox_inches='tight', pad_inches=0.1)

    length = os.stat(output_filename).st_size
    sys.stdout.write("Content-Type: image/png\n")
    sys.stdout.write("Content-Length: " + str(length) + "\n")
    sys.stdout.write("\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(open(output_filename, "rb").read())