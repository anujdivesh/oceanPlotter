import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

##INPUT
layer_map = 5
units = "null"
outlook = False
##INPUT

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
            'logscale': logscale
        }
    else:
        # If request fails, return a message
        return {'error': f"Failed to fetch data. Status code: {response.status_code}"}

layer_web_map_url = "https://dev-oceanportal.spc.int/middleware/api/layer_web_map/%s/" % (layer_map)
layer_data = fetch_wms_layer_data(layer_web_map_url)
# Input
ini_url = layer_data['wms_url']
styles = layer_data['styles'].split(',')[0]
style_split = styles.split('/')
palette = style_split[1]
layer = layer_data['layer_name'].split(',')[0]
min_color = layer_data['min_color']
max_color = layer_data['max_color']
steps = 6  # Number of steps
position = steps - 1


# URL of the legend image
url = "%s?REQUEST=GetLegendGraphic&PALETTE=%s&LAYERS=%s&COLORSCALERANGE=%s,%s&COLORBARONLY=true" % (ini_url, palette, layer, min_color, max_color)


# Check if outlook is True
if outlook:
    # Custom hardcoded color legend values
    colors = ['#ADD8E6', '#FFFF00', '#FFA500', '#FF0000', '#800000']  # Green, Yellow, Orange, Red
    labels = ['No Stress', 'Watch', 'Warning', 'Alert Level 1', 'Alert Level 2']
    values = ['0.0', '1.0', '2.5', '5.0', '10.0']  # Example values for each label

    # Create a figure for the custom legend
    plt.figure(figsize=(1.5, 6))  # Make the figure thinner
    for i in range(len(colors)):
        # Create a narrow color bar for each color (make the bars thinner by reducing the width)
        plt.fill_between([0, 0.3], i, i + 1, color=colors[i])  # Color bar width is now 0.5

        # Add the value and label on the side of the color bar
        plt.text(0.35, (i + 0.5), f'{labels[i]}', fontsize=10, color='black', va='center', ha='left')

    # Add a title for the custom legend
    plt.title('Legend', fontsize=14, fontweight='bold', color='black')

    # Remove axes for a cleaner look
    plt.axis('off')

    # Adjust the plot limits to ensure the labels fit inside the plot
    plt.xlim(0, 1)  # We extended the width for text, but kept bars narrower
    plt.savefig('legend.png', dpi=300,bbox_inches='tight', pad_inches=0.1) 
    # Show the plot
    #plt.show()
else:
    # Fetch the image from the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Load the image using PIL
        image = Image.open(BytesIO(response.content))

        # Display the image using matplotlib
        plt.figure(figsize=(1.5, 6))  # Adjust the figure size as needed
        plt.imshow(np.array(image))

        # Add labels for the steps
        steps = np.linspace(min_color, max_color, steps)  # Steps from min to max value
        for i, step in enumerate(steps):
            # Calculate the position of the label
            plt.text(1.05, (i / position), f'{step:.1f}', transform=plt.gca().transAxes, fontsize=12, color='black', va='center', ha='left')

        # Add a title for the legend
        plt.title('Legend', fontsize=14, fontweight='bold', color='black')

        if units != "null":
            # Add units on the left, aligned to the bottom in vertical orientation
            plt.text(-0.05, 0.02, 'Units (°)', transform=plt.gca().transAxes, fontsize=12, color='black', va='bottom', ha='center', rotation=90)

        # Remove axes for a cleaner look
        plt.axis('off')

        # Show the plot
        #plt.show()
        plt.savefig('legend.png', dpi=300,bbox_inches='tight', pad_inches=0.1) 
    else:
        print(f"Failed to fetch the image. Status code: {response.status_code}")
