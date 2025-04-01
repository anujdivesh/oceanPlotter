from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import requests

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

west_bound, east_bound, south_bound, north_bound = getBBox(2)
bbox_offset = 1

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize the Basemap with cylindrical projection
m = Basemap(projection='cyl', llcrnrlat=south_bound, urcrnrlat=north_bound, 
            llcrnrlon=west_bound, urcrnrlon=east_bound, resolution='h', ax=ax)


# Draw coastlines and countries
m.drawcoastlines()
m.drawcountries()

# Add latitude lines (parallels) with labels on top and bottom
#m.drawparallels(range(-45, 46, 15), labels=[1, 0, 0, 1])  # Labels on top and bottom

# Manually define the longitude lines from 120 to 280, with even ticks in between
#lon_ticks = np.arange(120, 281, 40)  # Create longitude ticks every 20 degrees

# Draw the meridians (longitude lines) manually
#m.drawmeridians(lon_ticks, labels=[0, 0, 0, 1])  # Labels on all sides

# Set the xticks manually (this will add labels to the plot)
#ax.set_xticks(lon_ticks)

# Show the plot
plt.show()
