
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap

##### CONFIGURATION PARAMETERS #####
# Choose one of these modes:


# Standard colorbar parameters (used when mode='standard')
mode = 'decile'  # Options: 'coral_bleaching', 'marine_heat_wave', 'decile', 'standard'
min_color = -2
max_color = 2
step = 0.5
color = 'jet'
unit = 'm'
##### END CONFIGURATION #####

# Create figure with white background
fig, ax = plt.subplots(figsize=(6, 1), facecolor='white')
fig.subplots_adjust(bottom=0.5)

if mode == 'coral_bleaching':
    # Coral bleaching configuration
    colors = ['#ADD8E6', '#FFFF00', '#FFA500', '#FF0000', '#800000']
    labels = ['No Stress', 'Watch', 'Warning', 'Alert Level 1', 'Alert Level 2']
    bounds = [0, 1, 2, 3, 4, 5]
    title = 'Coral Bleaching Alert Level'
    
elif mode == 'marine_heat_wave':
    # Marine Heat Wave configuration
    colors = ['#B0E0E6', '#FFFF00', '#FFA500', '#FF0000', '#8B0000']
    labels = ['0', '1', '2', '3', '4']
    bounds = [0, 1, 2, 3, 4, 5]
    title = 'Marine Heat Wave Level'
    
elif mode == 'decile':
    # Decile configuration
    colors = ['#00305A', '#4A89AF', '#A9C8DA', '#FFFFFF', '#F4B7A1', '#A8413F', '#5B001F']
    labels = ['Lowest\non Record', 'Very much\nbelow average', 'Below\nAverage', 
              'Average', 'Above\nAverage', 'Very much\nabove average', 'Highest\non record']
    bounds = [0, 1, 2, 3, 4, 5, 6, 7]
    title = 'Decile Categories'
    # Adjust figure size to accommodate multi-line labels
    fig.set_size_inches(9, 1.8)
    fig.subplots_adjust(bottom=0.45)
    
else:
    # Standard continuous colorbar
    cmap = getattr(plt.cm, color)
    norm = mpl.colors.Normalize(vmin=min_color, vmax=max_color)
    ticks = np.arange(min_color, max_color + step/2, step)
    
    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal',
        extend='both',
        ticks=ticks,
        spacing='uniform'
    )
    # Corrected this line - using set_ticklabels instead of set_xticklabels
    cb.set_ticklabels([f'{tick:g}' for tick in ticks])
    cb.set_label(unit, labelpad=10)
    plt.savefig('alegend.png', bbox_inches='tight', pad_inches=0.1, dpi=200, facecolor='white')
    plt.close()
    print(f"Standard colorbar using '{color}' colormap saved as 'alegend.png'")
    exit()

# Create discrete colormap for categorical data
cmap = ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Calculate tick positions (center of each color band)
tick_positions = [i + 0.5 for i in range(len(labels))]

# Create colorbar with discrete colors
cb = mpl.colorbar.ColorbarBase(
    ax,
    cmap=cmap,
    norm=norm,
    orientation='horizontal',
    boundaries=bounds,
    ticks=tick_positions,
    spacing='uniform',
    extend='neither'
)

# Configure ticks and labels
cb.set_ticks(tick_positions)
cb.set_ticklabels(labels)
if mode == 'standard':
    cb.set_label(title, labelpad=10)

# Adjust label formatting
if mode == 'decile':
    cb.ax.tick_params(labelsize=8, rotation=0)  # No rotation, just multi-line
    # Center-align the multi-line labels
    for label in cb.ax.get_xticklabels():
        label.set_horizontalalignment('center')
        label.set_verticalalignment('top')
        label.set_linespacing(0.8)
else:
    cb.ax.tick_params(labelsize=9)

# Save with white background
plt.savefig('alegend.png', 
           bbox_inches='tight', 
           pad_inches=0.1, 
           dpi=200, 
           facecolor='white')
plt.close()