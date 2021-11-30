from project_utils import Simulation
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

# NUmber of points to evaluate
n=2

# Generate a list of ratios to check
t_l = np.arange(0.0,1.0,0.05)
rel_dens = []
for counter, i in enumerate(t_l, start=1):
    print(f'Starting iteration {counter}')
    result = list(sim.get_results({'t_l': i}).values())
    rel_dens+= result
    print(f'Finished iteration: {counter}')

# Remove ratios where relative density is 1.0 to improve fit
rel_dens = np.array(rel_dens)
condition = np.where(rel_dens <= 1.0)

rel_dens_trim = rel_dens[condition]
t_l_trim = t_l[condition]

plt.title('Relative density vs. t/l ratio for octet truss')
plt.scatter(t_l, rel_dens, label='data')

poly_fit = np.poly1d(np.polyfit(t_l_trim, rel_dens_trim, 2))
plt.plot(t_l_trim, poly_fit(t_l_trim), label='polyfit')

plt.xlabel('Thickness to length ratio')
plt.ylabel('Relative density')
plt.legend(loc='best')
plt.show()
