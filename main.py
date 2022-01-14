import os
import time
import yaml
from pathlib import Path
import pprint

import concurrent.futures
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

from utils.project_utils import Simulation, clean_replay, generate_report, get_datestring
import utils.plot_utils

import numpy as np


# ntop command line .exe file (CHANGE THIS!!!)
exe_path = r'C:\Program Files\nTopology\nTopology\ntopcl.exe'

# reading configuration
current_dir = os.getcwd()
config_file = 'cantilever_moo/cantilever_moo_config.yml'  # this is used to switch between models
config_dir = os.path.abspath(os.path.join(current_dir, 'models', config_file))

# extracting parameters
with open(config_dir) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_name = config['model_name']  # model name
opt_config = config['optimisation']  # optimisation setup

model_dir = os.path.abspath(os.path.join(current_dir, 'models', model_name))
db_dir = os.path.abspath(os.path.join(current_dir, 'database'))

# checking if directories exist and creating new directories if not
directory_list = [model_dir, db_dir]
for directory in directory_list:
    Path(directory).mkdir(parents=True, exist_ok=True)

# nTopology notebook
notebook_name = config['notebook_name']
notebook_dir = os.path.abspath(os.path.join(model_dir, f'{notebook_name}.ntop'))

# multiobjective flag
multiobjective = opt_config['multiobjective']


def main():
    if multiobjective:
        objective_config = opt_config['multi']
        # result_metrics is a list of objective and constraint names
        result_metrics = opt_config['constraint_metrics'] + [i.get('name') for i in
                                                             objective_config['objective_metrics']]
    else:
        objective_config = opt_config['single']
        result_metrics = opt_config['constraint_metrics'] + [objective_config['objective_metric']]

    # instantiate the Simulation object
    # we provide directories for temporary input and output JSON files, template for the input JSON,
    # name of the notebook used in the simulation and the result metrics of interest

    sim = Simulation(model_dir, notebook_dir, exe_path, result_metrics)
    sim.check_template()

    ## Bayesian Optimization in Service API

    # Generation strategy
    NUM_SOBOL_STEPS = 5
    gs = GenerationStrategy(steps=
                            [GenerationStep(model=Models.SOBOL, num_trials=NUM_SOBOL_STEPS),
                             GenerationStep(model=Models[opt_config['model']], num_trials=-1)])

    # Initialize the ax client
    ax_client = AxClient(generation_strategy=gs, random_seed=12345, verbose_logging=True)

    # Define parameters

    if opt_config['uniform_params']:  # for uniform parameter range
        params = [{'name': f'x{i + 1}',
                   'type': 'range',
                   'bounds': [opt_config['lo_bound'], opt_config['up_bound']],
                   'value_type': 'float'
                   } for i in range(opt_config['num_of_params'])]

    else:
        params = opt_config['parameters']


    # Creating an experiment
    if multiobjective:
        ax_client.create_experiment(
            name=opt_config['experiment_name'],
            parameters=params,
            objectives={i['name']: ObjectiveProperties(minimize=i['minimize'], threshold=i['threshold']) for i in
                        objective_config['objective_metrics']},
            outcome_constraints=opt_config['outcome_constraints'])
    else:
        ax_client.create_experiment(
            name=opt_config['experiment_name'],
            parameters=params,
            objective_name=objective_config['objective_metric'],
            minimize=objective_config['minimize'],  # Optional, defaults to False.
            outcome_constraints=opt_config['outcome_constraints'])

    NUM_OF_ITERS = opt_config['num_of_iters']

    # Manual override used for dev
    NUM_OF_ITERS = 10
    BATCH_SIZE = 3

    # Initializing variables used in the iteration loop
    
    abandoned_trials_count = 0
    NUM_OF_BATCHES = NUM_OF_ITERS//BATCH_SIZE if NUM_OF_ITERS%BATCH_SIZE==0 else NUM_OF_ITERS//BATCH_SIZE
    
    for i in range(NUM_OF_BATCHES):
        print()
        results = {}
        
        trials_to_evaluate = {}
        # Sequentially generate the batch
        for j in range(min(NUM_OF_ITERS-i*BATCH_SIZE, BATCH_SIZE)):
            parameterization, trial_index = ax_client.get_next_trial()
            trials_to_evaluate[trial_index] = parameterization
        
        # Evaluate the results in parallel and append results to a dictionary
        for trial_index, parametrization in trials_to_evaluate.items():
            with concurrent.futures.ProcessPoolExecutor() as executor:
                try:
                    name_index = f'{i}_{j}'
                    exec = executor.submit(sim.get_results, parametrization,name_index)
                    results.update({trial_index: exec.result()})
                except KeyboardInterrupt:
                    print('Program interrupted by user')
                    break
                except Exception as e:
                    ax_client.abandon_trial(trial_index=trial_index)
                    abandoned_trials_count += 1
                    print(f'[WARNING] Abandoning trial {trial_index} due to processing errors.')
                    print(e)
                    if abandoned_trials_count > 0.1 * NUM_OF_ITERS:
                        print('[WARNING] More than 10 % of iterations were abandoned. Consider improving the parametrization.')
                    # break
                continue

        for trial_index in results:
            ax_client.complete_trial(trial_index, results.get(trial_index))
            

    try:
        # Save `AxClient` to a JSON snapshot.
        _, dt_string = get_datestring()

        db_save_name = f'simulation_run_{dt_string}.json'
        ax_client.save_to_json_file(filepath=os.path.join(db_dir, db_save_name))

    except Exception:
        print('[WARNING] The JSON snapshot of the Ax Client has not been saved.')

    return ax_client


if __name__ == "__main__":

    load_existing_client = False
    client_filename = 'simulation_run_13012022184206.json'
    client_filepath = os.path.join(db_dir, client_filename)
    
    start = time.perf_counter()
    if load_existing_client:
        # (Optional) Reinstantiate an `AxClient` from a JSON snapshot.
        ax_client = AxClient.load_from_json_file(filepath=client_filepath)
    else:
        # Run the simulation
        ax_client = main()
    
    finish = time.perf_counter()
    
    print(f'Simulation took {finish-start} seconds to complete')

    # Plotting

    save_pdf = True
    save_png = True  # This must be true for generating reports
    plot_dir = os.path.join(os.getcwd(), 'reports', 'plots')

    P = utils.plot_utils.Plot(ax_client, plot_dir, save_pdf, save_png)
    P.clean_plot_dir()
    if multiobjective:
        try:
            P.plot_moo_trials()
            P.plot_posterior_pareto_frontier()
        except Exception as e:
            print('[WARNING] An exception occured while plotting!')
            print(e)
    else:
        try:
            P.plot_single_objective_trials()
            P.plot_single_objective_convergence()
            P.plot_single_objective_distances()
        except Exception as e:
            print('[WARNING] An exception occured while plotting!')
            print(e)
    print(ax_client.generation_strategy.trials_as_df)
    print(exp_to_df(ax_client.experiment))
    #         # Generating a report
    #     try:
    #         generate_report(opt_config, means, best_parameters)
    #     except Exception:
    #         print(' [WARNING] An exception occured while generating the report')
