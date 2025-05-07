import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PIL import Image
from io import BytesIO
import geopandas as gpd
import sys
from datetime import datetime,timedelta
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
import matplotlib.colors as mcolors
import netCDF4 as nc
from matplotlib.lines import Line2D

#ADJUST THE FOLLOWING
region = 2


#####FUNCTIONS#####
def plot_coastline_from_shapefile(ax, shapefile_path):
    """
    Plot coastline polygons from local shapefile with proper dateline handling
    
    Parameters:
    - ax: matplotlib axis object
    - shapefile_path: path to the shapefile (without .shp extension)
    """
    try:
        # Read shapefile using geopandas
        gdf = gpd.read_file(shapefile_path)
        
        # Ensure correct CRS (EPSG:4326)
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326', allow_override=True)
        else:
            gdf = gdf.to_crs('EPSG:4326')
        
        # Simplify complex geometries (adjust tolerance as needed)
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01)
        
        # Plot with dateline handling
        for geom in gdf.geometry:
            if not geom.is_valid:
                geom = geom.buffer(0)  # Fix invalid geometries
                
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                # Create two versions - original and shifted by 360°
                original = gpd.GeoSeries([geom], crs='EPSG:4326')
                shifted = original.translate(xoff=360)
                
                # Combine both versions
                combined = pd.concat([original, shifted])
                
                # Plot each geometry
                for poly in combined:
                    if poly.geom_type == 'Polygon':
                        x, y = m(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1])
                        ax.fill(x, y, color='#A9A9A9', ec='black', lw=0.5, zorder=2)
                    elif poly.geom_type == 'MultiPolygon':
                        for part in poly.geoms:
                            x, y = m(part.exterior.coords.xy[0], part.exterior.coords.xy[1])
                            ax.fill(x, y, color='#A9A9A9', ec='black', lw=0.5, zorder=2)
        
        
    except Exception as e:
        print(f"Error plotting coastline from shapefile: {str(e)}")

def plot_eez_from_shapefile(ax, shapefile_path, line_color='#FF69B4', line_width=0.8, linestyle='--'):
    """
    Plot EEZ boundaries from local shapefile with proper dateline handling
    
    Parameters:
    - ax: matplotlib axis object
    - shapefile_path: path to the shapefile (including .shp extension)
    - line_color: color of EEZ boundary lines (default: pink #FF69B4)
    - line_width: width of boundary lines (default: 0.8)
    - linestyle: line style (default: dashed '--')
    """
    try:
        # Read shapefile using geopandas
        print(f"Attempting to read shapefile from: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        # Check if the file was read successfully
        if len(gdf) == 0:
            print("Warning: Empty GeoDataFrame loaded from shapefile")
            return
            
        # Ensure correct CRS (EPSG:4326)
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326', allow_override=True)
        else:
            gdf = gdf.to_crs('EPSG:4326')
        
        # Simplify complex geometries
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01)
        
        # Plot with dateline handling
        for geom in gdf.geometry:
            if not geom.is_valid:
                geom = geom.buffer(0)  # Fix invalid geometries
                
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                # Create two versions - original and shifted by 360°
                original = gpd.GeoSeries([geom], crs='EPSG:4326')
                shifted = original.translate(xoff=360)
                
                # Combine both versions
                combined = pd.concat([original, shifted])
                
                # Plot each geometry's boundary
                for poly in combined:
                    if poly.geom_type == 'Polygon':
                        # Plot exterior
                        x, y = m(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1])
                        ax.plot(x, y, color=line_color, linewidth=line_width, 
                               linestyle=linestyle, zorder=3)
                        
                        # Plot any interior rings (holes)
                        for interior in poly.interiors:
                            xi, yi = m(interior.coords.xy[0], interior.coords.xy[1])
                            ax.plot(xi, yi, color=line_color, linewidth=line_width,
                                   linestyle=linestyle, zorder=3)
                                
                    elif poly.geom_type == 'MultiPolygon':
                        for part in poly.geoms:
                            # Plot exterior
                            x, y = m(part.exterior.coords.xy[0], part.exterior.coords.xy[1])
                            ax.plot(x, y, color=line_color, linewidth=line_width,
                                   linestyle=linestyle, zorder=3)
                            
                            # Plot any interior rings
                            for interior in part.interiors:
                                xi, yi = m(interior.coords.xy[0], interior.coords.xy[1])
                                ax.plot(xi, yi, color=line_color, linewidth=line_width,
                                       linestyle=linestyle, zorder=3)
        
        print(f"Successfully plotted {len(gdf)} EEZ boundaries")
        
    except Exception as e:
        print(f"Error plotting EEZ boundaries: {str(e)}")
        import traceback
        traceback.print_exc()

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
        print(f"Error fetching data from API: {e}")
        return None

def plot_city_names(m, csv_path='names/pacific_city_names.csv', country_code='FJI', 
                   text_color='red', font_size=10, offset=0.5):
    """
    Enhanced city name plotting with debugging
    """
    try:
        # 1. Verify CSV file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        
        # 2. Read and filter data
        df = pd.read_csv(csv_path)
        #print(f"CSV contains {len(df)} entries")
        
        cities = df[df['country_code'].str.upper() == country_code.upper()]
        if cities.empty:
            print(f"No cities found for country: {country_code}")
            print("Available country codes:", df['country_code'].unique())
            return
            
        #print(f"Found {len(cities)} cities for {country_code}")
        
        # 3. Convert coordinates
        x, y = m(cities['lon'].values, cities['lat'].values)
        #print(f"First city coordinates - Lon: {cities['lon'].iloc[0]}, Lat: {cities['lat'].iloc[0]}")
        #print(f"Converted to X: {x[0]}, Y: {y[0]}")
        
        # 4. Plot with visible settings
        ax = plt.gca()
        for idx, row in cities.iterrows():
            ax.text(x[idx]+offset, y[idx]+offset, row['name'],
                   color=text_color,        # Using bright red for visibility
                   fontsize=font_size,      # Larger font
                   fontweight='bold',
                   ha='left',
                   va='bottom',
                   clip_on=True,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))  # White background
        
        # 5. Verify plotting
        #print(f"Plotted text at {len(cities)} locations")
        
    except Exception as e:
        print(f"Error in plot_city_names: {str(e)}")
        import traceback
        traceback.print_exc()

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

def get_from_file(file_path, target_time, variable_name, adjust_lon=False):
    try:
        # Open dataset from local file
        with xr.open_dataset(file_path, engine='netcdf4', mask_and_scale=True, decode_cf=True) as ds:
            
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
        raise RuntimeError(f"Error accessing file data: {str(e)}")

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
    elif name == 'VUT':
        name = 'VAN'
    elif name == 'SLB':
        name = 'SOL'
    elif name == 'WSM':
        name = 'SAM'
    elif name == 'NRU':
        name = 'NAU'
    elif name == 'PLW':
        name = 'PAU'
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
        short_name = data['short_name']
    else:
        print(f"Failed to retrieve bounding box data. Status code: {response.status_code}")
    
    return west_bound, east_bound, south_bound, north_bound, country_name, short_name

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
                    ax.plot(x, y, marker=None, color='black', linewidth=1,linestyle='--')  # Plot the boundary line

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

    title_suffix = "Daily Average Sea Surface Temperature Anomaly:"
    dataset_text = "Reynolds SST"
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
    
#####FUNCTIONS#####
##INIT##
config = get_config_variables()

#####PARAMETER#####

layer_id = 5
time= add_z_if_needed("2024-10-01T00:00:00Z")
resolution = "l"
#####PARAMETER#####

layer_map_data = fetch_wms_layer_data(layer_id)

#REMOVE DEMO
#time = demo_time(layer_map_data)
#REMOVE DEMO
#####MAIN#####
dap_url, dap_variable = get_dap_config(layer_map_data)
title, dataset_text = get_title(layer_map_data,time)
cmap_name, plot_type, min_color_plot, max_color_plot, steps, units, levels, discrete = get_plot_config(layer_map_data)
west_bound, east_bound, south_bound, north_bound, country_name, short_name = getBBox(region)

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
if plot_type == "contourf_nozero":
    lon, lat, data_extract = getfromDAP(dap_url, time, dap_variable,adjust_lon=True)
    cs, cbar = plot_filled_contours_no_zero(ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
        min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units
    )

#ADD LOGO AND FOOTER
add_logo_and_footer(fig=fig, ax=ax, ax2=ax2, ax2_pos=ax2_pos, region=1, copyright_text=config.copyright_text,\
     footer_text=config.footer_text,dataset_text=dataset_text)

#PLOT EEZ
#getEEZ(ax2,eez_url)
print('shapefile/country_eez/'+short_name+'.shp')

cities = pd.read_json('names/pac_names.json')

# Filter for selected country's cities
filtered_cities = cities[cities['country_code'] == short_name]

# Transform coordinates
longitudes = filtered_cities['lon'].values
latitudes = filtered_cities['lat'].values
x_coords, y_coords = m(longitudes, latitudes)

# Plot city names
for x, y, name in zip(x_coords, y_coords, filtered_cities['name']):
    print(x)
    ax2.text(x + 0.3, y + 0.1, name,
             fontsize=5, color='black',
             ha='left', va='center')


plot_coastline_from_shapefile(ax2, 'shapefile/coastline/Pacific_Coastlines_openstreet_polygon.shp')
eez_shapefile_path = f'shapefile/country_eez/{short_name}.shp'
print(f"Looking for EEZ shapefile at: {eez_shapefile_path}")

if os.path.exists(eez_shapefile_path):
    print("EEZ shapefile found, plotting...")
    plot_eez_from_shapefile(ax2, eez_shapefile_path,
                          line_color='#FF69B4',  # Bright pink
                          line_width=1.2,       # Slightly thicker
                          linestyle='-')        # Solid line

plt.savefig('anuj4.png', bbox_inches='tight', pad_inches=0.1)