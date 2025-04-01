import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Input
ini_url = "https://dev-oceanportal.spc.int/thredds/wms/POP/model/regional/nasa/nrt/daily/chlorophyll/latest.ncml"
palette = "x-Sst"
layer = "CRW_BAA"
min_color = 0.001
max_color = 10.0
steps = 6  # Number of steps
position = steps - 1
units = "&"
outlook = True

# URL of the legend image
url = "%s?REQUEST=GetLegendGraphic&PALETTE=%s&LAYERS=%s&COLORSCALERANGE=%s,%s&COLORBARONLY=true" % (ini_url, palette, layer, min_color, max_color)
print(url)

# Check if outlook is True
if outlook:
    # Custom hardcoded color legend values
    colors = ['#00FF00', '#FFFF00', '#FF9900', '#FF6600', '#FF0000']  # Green, Yellow, Orange, Red
    labels = ['No Stress', 'Watch', 'Warning', 'Alert Level 1', 'Alert Level 2']
    values = ['0.0', '1.0', '2.5', '5.0', '10.0']  # Example values for each label

    # Create a figure for the custom legend
    plt.figure(figsize=(2, 6))  # Make the figure thinner
    for i in range(len(colors)):
        # Create a narrow color bar for each color (make the bars thinner by reducing the width)
        plt.fill_between([0, 0.2], i, i + 1, color=colors[i])  # Color bar width is now 0.5

        # Add the value and label on the side of the color bar
        plt.text(0.25, (i + 0.5), f'{labels[i]}', fontsize=6, color='black', va='center', ha='left')

    # Add a title for the custom legend
    plt.title('Legend', fontsize=14, fontweight='bold', color='black')

    # Remove axes for a cleaner look
    plt.axis('off')

    # Adjust the plot limits to ensure the labels fit inside the plot
    plt.xlim(0, 1)  # We extended the width for text, but kept bars narrower

    # Show the plot
    plt.show()
else:
    # Fetch the image from the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Load the image using PIL
        image = Image.open(BytesIO(response.content))

        # Display the image using matplotlib
        plt.figure(figsize=(2, 6))  # Adjust the figure size as needed
        plt.imshow(np.array(image))

        # Add labels for the steps
        steps = np.linspace(min_color, max_color, steps)  # Steps from min to max value
        for i, step in enumerate(steps):
            # Calculate the position of the label
            plt.text(1.05, (i / position), f'{step:.1f}', transform=plt.gca().transAxes, fontsize=12, color='black', va='center', ha='left')

        # Add a title for the legend
        plt.title('Legend', fontsize=14, fontweight='bold', color='black')

        if units != "&":
            # Add units on the left, aligned to the bottom in vertical orientation
            plt.text(-0.05, 0.02, 'Units (°)', transform=plt.gca().transAxes, fontsize=12, color='black', va='bottom', ha='center', rotation=90)

        # Remove axes for a cleaner look
        plt.axis('off')

        # Show the plot
        plt.show()
    else:
        print(f"Failed to fetch the image. Status code: {response.status_code}")
