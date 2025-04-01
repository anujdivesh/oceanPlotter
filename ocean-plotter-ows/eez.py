import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import geopandas as gpd

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

# Fetch the GeoJSON data for the EEZ boundaries
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
                ax.plot(x, y, marker=None, color='blue', linewidth=2)  # Plot the boundary line

else:
    print("Failed to retrieve the GeoJSON data.")

# Footer text
footer_text = '© All rights reserved SPC'

# Position the footer text at the bottom-right of the figure
plt.figtext(0.95, 0.02, footer_text, ha='right', va='bottom', fontsize=10, color='black', fontweight='light')

# Adjusting layout for extra space
plt.subplots_adjust(top=0.92, bottom=0.05)  # 0.92 for top, 0.05 for bottom space

# Display the combined map and custom legend
plt.tight_layout()  # Ensures no overlap between map and legend
plt.show()
