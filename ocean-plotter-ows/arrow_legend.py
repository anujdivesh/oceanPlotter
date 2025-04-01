import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Input
ini_url = "https://dev-oceanportal.spc.int/thredds/wms/POP/model/regional/bom/forecast/hourly/wavewatch3/latest.nc"
palette = "x-Sst"
layer = "mn_wav_dir"
min_color = 0
max_color = 4
steps = 6  # Number of steps
position = steps - 1
units = "units"  # Units to display next to numbers

# URL of the legend image
url = "%s?REQUEST=GetLegendGraphic&PALETTE=%s&LAYERS=%s&COLORSCALERANGE=%s,%s&COLORBARONLY=true" % (ini_url, palette, layer, min_color, max_color)

# Fetch the image from the URL
response = requests.get(url)
if response.status_code == 200:
    # Load the image using PIL
    image = Image.open(BytesIO(response.content)).convert("RGBA")

    # Create a new image to draw the arrows
    width, height = image.size
    new_image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(new_image)

    # Define the shape of the arrow at the top and bottom
    arrow_height = int(height * 0.1)  # Height of the arrow

    # Top arrow: Inverted "V" (upward-pointing arrow)
    draw.polygon([
        (0, 0),              # Left point
        (width, 0),          # Right point
        (width * 0.5, arrow_height)  # Tip of the arrow
    ], fill="black")

    # Bottom arrow: Regular "V" (downward-pointing arrow)
    draw.polygon([
        (0, height),         # Left point
        (width, height),    # Right point
        (width * 0.5, height - arrow_height)  # Tip of the arrow
    ], fill="black")

    # Composite the original image with the arrows
    # Paste the original image onto the new image, preserving the middle part
    new_image.paste(image, (0, 0), image)

    # Display the modified image
    plt.figure(figsize=(3, 6))  # Adjust the figure size to make it larger towards the right
    plt.imshow(new_image)

    # Add labels for the steps
    steps_values = np.linspace(min_color, max_color, steps)  # Steps from min to max value
    for i, step in enumerate(steps_values):
        # Calculate the position of the label
        label = f'{step:.1f}'
        plt.text(1.05, (i / position), label, transform=plt.gca().transAxes, fontsize=12, color='black', va='center', ha='left')

    # Add units at the far right of the numbers (move further right) and rotate vertically
    plt.text(1.3, 0.5, units, transform=plt.gca().transAxes, fontsize=12, color='black', va='center', ha='left', rotation=90)

    # Add a title for the legend
    plt.title('Legend', fontsize=14, fontweight='bold', color='black')

    # Remove axes for a cleaner look
    plt.axis('off')

    # Show the plot
    plt.show()
else:
    print(f"Failed to fetch the image. Status code: {response.status_code}")