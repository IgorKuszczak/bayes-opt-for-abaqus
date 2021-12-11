from project_utils import Simulation
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit as logistic
from scipy.optimize import curve_fit

# This is an example file for how to do sequential runs in nTop while using the methods defined in the Simulation object

# ntop command line .exe file (CHANGE THIS!!!)
exe_path = r'C:\Program Files\nTopology\nTopology\ntopcl.exe'

# Main directory used as reference
main_dir = os.path.dirname(os.getcwd())

# Manually specify the paths for the ntop notebook model you want to use
model_name = 'rel_dens_calc'
model_dir = os.path.abspath(os.path.join(main_dir, 'models', model_name))

directory_list = [model_dir]
for directory in directory_list:
    Path(directory).mkdir(parents=True, exist_ok=True)

notebook_name = 'rel_dens_calc'
notebook_dir = os.path.abspath(os.path.join(model_dir, f'{notebook_name}.ntop'))

result_metrics = ['relative_density']

sim = Simulation(model_dir, notebook_dir, exe_path, result_metrics)

# Now we check if there is an existing template
sim.check_template()

# Generate a list of ratios to check
t_l = np.arange(0.0, 0.50, 0.025)

rel_dens = []
for counter, i in enumerate(t_l, start=1):
    print(f'Starting iteration {counter}: of {np.size(t_l)}')
    result = list(sim.get_results({'t_l': i}).values())
    rel_dens += result
    print(f'Finished iteration {counter}: of {np.size(t_l)}')


# Curve fit
def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


p0 = [max(rel_dens), np.median(t_l), 1, min(rel_dens)]  # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, t_l, rel_dens, p0, method='dogbox')

t_l_range = np.arange(0.0, 1.0, 0.01)
rel_dens_pred = sigmoid(t_l_range, *popt)

# Plotting
# Data
plt.scatter(t_l, rel_dens, label='data')

# Logistic fit
plt.plot(t_l_range, rel_dens_pred, label='logistic fit')


plt.title('Relative density vs. (t/l) ratio for octet truss')
plt.xlabel('Thickness to length ratio')
plt.ylabel('Relative density')
plt.legend(loc='best')
plt.show()

print(f'The best fit values for (L, x0, k, b)  are: ({popt[0]}, {popt[1]}, {popt[2]}, {popt[3]}) where we fit the following form:')
print('y = L / (1 + np.exp(-k * (x - x0))) + b')