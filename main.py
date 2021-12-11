import os
import yaml
from pathlib import Path
import pprint

pprint.sorted = lambda x, key=None: x
from tqdm import tqdm

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

from utils.project_utils import Simulation, clean_replay, generate_report, get_datestring
import utils.plot_utils

import numpy as np

# ntop command line .exe file (CHANGE THIS!!!)
exe_path = r'C:\Program Files\nTopology\nTopology\ntopcl.exe'

# reading configuration
current_dir = os.getcwd()
config_file = 'mbb_latticed/mbb_latticed_config.yml'  # this is used to switch between models
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
        result_metrics =opt_config['constraint_metrics'] + [i.get('name') for i in objective_config['objective_metrics']]
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
    gs = GenerationStrategy(steps=
                            [GenerationStep(model=Models.SOBOL, num_trials=opt_config['num_sobol_steps']),
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

    param_names = [param['name'] for param in params]

    # Define the evaluation function
    def eval_func(parametrization):

        x = np.array([parametrization.get(name) for name in param_names])

        results = sim.get_results(parametrization)

        # return_dict = {k: (getattr(sim, k), 0.0) for k in result_metrics}

        return results

    # Creating an experiment
    if multiobjective:
        ax_client.create_experiment(
            name=opt_config['experiment_name'],
            parameters=params,
            objectives={i['name']: ObjectiveProperties(minimize=i['minimize'], threshold=i['threshold']) for i in objective_config['objective_metrics']},
            outcome_constraints=opt_config['outcome_constraints'])
    else:
        ax_client.create_experiment(
            name=opt_config['experiment_name'],
            parameters=params,
            objective_name=objective_config['objective_metric'],
            minimize=objective_config['minimize'],  # Optional, defaults to False.
            outcome_constraints=opt_config['outcome_constraints'])

    num_of_iterations = opt_config['num_of_iters']
    abandoned_trials_count = 0

    for _ in range(num_of_iterations):

        parameters, trial_index = ax_client.get_next_trial()

        try:
            data = eval_func(parameters)
        except KeyboardInterrupt:
            break
            print('Program interrupted by user')
            clean_replay()

        except Exception:
            ax_client.abandon_trial(trial_index=trial_index)
            abandoned_trials_count += 1
            print('[WARNING] Abandoning trial due to processing errors.')
            if abandoned_trials_count > 0.1 * num_of_iterations:
                print('[WARNING] More than 10 % of iterations were abandoned. Consider improving the parametrization.')
                # break
            continue

        ax_client.complete_trial(trial_index=trial_index, raw_data=data)

    try:
        # Save `AxClient` to a JSON snapshot.
        _, dt_string = get_datestring()

        db_save_name = f'final_110_run_{dt_string}.json'
        ax_client.save_to_json_file(filepath=os.path.join(db_dir, db_save_name))

    except Exception:
        print('[WARNING] The JSON snapshot of the Ax Client has not been saved.')

    return ax_client


if __name__ == "__main__":

    load_existing_client = False
    client_filename = 'final_110_run_02122021143251.json'
    client_filepath = os.path.join(db_dir, client_filename)

    if load_existing_client:
        # (Optional) Reinstantiate an `AxClient` from a JSON snapshot.
        ax_client = AxClient.load_from_json_file(filepath=client_filepath)
    else:
        # Run the simulation
        ax_client = main()

    if multiobjective:
        best_parameters = ax_client.get_pareto_optimal_parameters()

    else:
        best_parameters, values = ax_client.get_best_parameters()

        print(f'Best parameters are: {best_parameters}')

        experiment = ax_client.experiment
        model = ax_client.generation_strategy.model
        trials_df = ax_client.generation_strategy.trials_as_df

        trial_values = experiment.trials.values()

        best_objectives = np.array([trial.objective_mean
                                    if trial.status.is_completed
                                    else 0.0
                                    for trial in trial_values])

        # Consecutive evaluations
        arms_by_trial = np.array([list(trial.arm.parameters.values())
                                  if trial.status.is_completed
                                  else [np.nan] * opt_config['num_of_params']
                                  for trial in trial_values])

        # print(arms_by_trial)
        distances = np.linalg.norm(np.diff(arms_by_trial, axis=0), ord=2, axis=1)

        # Mask for finding feasible solutions
        mask = np.ones(len(trial_values), dtype=bool)

        ## Evaluating solution feasibility
        constraint_metrics = opt_config['constraint_metrics']

        for metric in constraint_metrics:
            vals = [trial.get_metric_mean(metric)
                    if trial.status.is_completed
                    else 0
                    for trial in trial_values]
            metric_name = str(metric)
            exec(f'{metric_name}=np.array({vals})')

        for constraint in opt_config['outcome_constraints']:
            mask = np.logical_and(mask, eval(constraint)).ravel()

        idx = np.where(mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        feasible_objectives = best_objectives[idx]
        feasible_objectives[idx == 0] = feasible_objectives[(idx != 0).argmax(axis=0)]
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11

        # Minimum/maximum accumulate for single objective 1
        minimize = True
        if minimize:
            y = np.minimum.accumulate(feasible_objectives, axis=0)
            best_trial = np.argmin(feasible_objectives)
        else:
            y = np.maximum.accumulate(feasible_objectives, axis=0)
            best_trial = np.argmax(feasible_objectives)

        y[idx == 0] = np.nan
        # Best solution as proposed by the client
        means, covariances = values

        print('The best objectives are:')
        pprint.pprint(means)
        print('The best parameters are:')
        pprint.pprint(best_parameters)
        print(f'The best trial occured at iteration {best_trial + 1}')

        # Plotting
        save_pdf = True
        save_png = True  # This must be true for generating reports
        plot_dir = os.path.join(os.getcwd(), 'reports', 'plots')
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        P = utils.plot_utils.Plot(opt_config['num_sobol_steps'], plot_dir, save_pdf, save_png)
        P.clean_plot_dir()
        try:
            P.plot_convergence_plt(y)
            P.plot_evaluations_plt(best_objectives, mask)
            P.plot_distances_plt(distances)
            # This probably should be changed to be not model specific:
            # P.plot_contour_plt(model, 'eta', 'xi', 'stiff_ratio', best_parameters, 100)

        except Exception:
            print('[WARNING] An exception occured while plotting!')
            pass

            # Generating a report
        try:
            generate_report(opt_config, means, best_parameters)
        except Exception:
            print(' [WARNING] An exception occured while generating the report')
        #
        # clean_replay()  # clean replay files
