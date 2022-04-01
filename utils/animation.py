import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mtpimg

from pathlib import Path
import os

# Ax imports 
from ax.service.ax_client import AxClient
from ax.plot.contour import _get_contour_predictions

import plot_utils # plotting utils


# Plot settings
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.axisbelow'] = True
    

## Preparing data for plotting

# Loading the AxClient
client_filename ="hex_thick_50_10_so_0_05_rd.json"
client_directory = r"D:\Temp\kuszczak_i\Studentship\bayes-opt-for-abaqus\database"
client_filepath = os.path.abspath(os.path.join(client_directory, client_filename))

# Load the Ax Client from a JSON file
ax_client = AxClient.load_from_json_file(filepath=client_filepath)
best_parameters,values = ax_client.get_best_parameters()
model = ax_client.generation_strategy.model


param_x = 'eta'
param_y = 'xi'
metric_name = 'stiffness_ratio'
density = 50


# Creating a contour plot
data, f_plt, sd_plt, grid_x, grid_y, scales = _get_contour_predictions(
    model=model,
    x_param_name=param_x,
    y_param_name=param_y,
    metric=metric_name,
    generator_runs_dict = None,
    density=density)

X, Y = np.meshgrid(grid_x,grid_y) 
Z_f = np.asarray(f_plt).reshape(density,density)
Z_sd = np.asarray(sd_plt).reshape(density,density)

labels=[]
evaluations = []

for key, value in data[1].items():
    labels.append(key)
    evaluations.append(list(value[1].values()))

evaluations = np.asarray(evaluations)

###############################################################################
## Plotting data
# We have 2 animated plots, side by side
fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2) 

# On the right (ax2), we want to animate a precomputed list of images
# List of images to go through
image_folder = r"D:\Temp\kuszczak_i\Studentship\photos_for_animation"
image_paths= Path(image_folder).glob('*.tif')

frames= []

for i in image_paths:
    img = mtpimg.imread(i)
    frames.append(img)

# On the left (ax1), we need plot consecutive evaluations on the contour plot
x = evaluations[:,0]
y = evaluations[:,1]


# initialize two objects (one in each axes)
graph, = ax1.plot([], [], 'o', markersize = 6, mec='black')

# force equal axes
ax1.set_aspect('equal', 'box')

cont1 = ax1.contourf(X, Y, Z_sd, 20, cmap='plasma')
#ax1.set_title('Parametrization')
ax1.set(xlabel=param_x, ylabel=param_y)

fig.colorbar(cont1, ax=ax1)

unit_cell = ax2.plot([],[])
ax2.set_axis_off()
#ax2.set_title('Unit Cell')

def animate_points(i):
    graph.set_data(x[:i+1], y[:i+1])
    graph.set_color('white')
    frame = ax2.imshow(frames[i],animated=True)
    return graph
  
 
ani = animation.FuncAnimation(fig,animate_points,frames=50,interval=50)



