from project_utils import Simulation
import os
from pathlib import Path
import numpy as np

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

print(model_dir)

notebook_name = 'rel_dens_calc'
notebook_dir = os.path.abspath(os.path.join(model_dir, f'{notebook_name}.ntop'))

result_metrics = ['relative_density']

sim = Simulation(model_dir, notebook_dir, exe_path, result_metrics)

# Now we check if there is an existing template
sim.check_template()

result = sim.get_results({'t/l': 0.1})
print(result)
