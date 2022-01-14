from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
from .project_utils import clean_directory


class Plot:

    def __init__(self, ax_client, plot_dir, save_pdf=False, save_png=False):

        self.ax_client = ax_client
        self.experiment = ax_client.experiment
        self.objective = ax_client.experiment.optimization_config.objective
        self.trials_df = ax_client.get_trials_data_frame().sort_values(by=['trial_index'], ascending=True)
        self.trial_values = self.experiment.trials.values()
        self.sobol_num = ax_client.generation_strategy._steps[0].num_trials

        self.plot_dir = plot_dir
        Path(plot_dir).mkdir(parents=True, exist_ok=True)  # check if plot directory exists

        self.save_pdf = save_pdf
        self.save_png = save_png

        # Plot style setup
        mpl.rcParams.update(mpl.rcParamsDefault)  # reset style
        if not mpl.is_interactive(): plt.ion()  # enable interactive mode
        # plt.style.use('dark_background')
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['axes.axisbelow'] = True
        plt.rcParams["figure.figsize"] = (10, 6)

    # Single Objective Plotting

    def plot_single_objective_convergence(self):
        # Calculations
        best_objectives = np.array([[trial.objective_mean for trial in self.experiment.trials.values()]])

        if self.objective.minimize:
            y = np.minimum.accumulate(best_objectives, axis=1)
        else:
            y = np.maximum.accumulate(best_objectives, axis=1)

        x = np.arange(1, np.size(y) + 1)

        # Plotting
        fig = plt.figure()

        plt.plot(x, y.T, linewidth=3)
        plt.axvline(self.sobol_num, linewidth=2, linestyle='--', color='k')

        plt.title('Convergence plot')
        plt.xlabel('Trial #')
        plt.ylabel('Best objective')
        plt.ylim(*plt.ylim())
        plt.xlim([1, np.size(y) + 1])

        self.save_plot('convergence_plot', fig)
        plt.show()

    def plot_single_objective_trials(self):
        # Consecutive evaluations

        objective_name = self.objective.metric.name

        values = np.asarray([trial.get_metric_mean(objective_name)
                             if trial.status.is_completed
                             else np.nan
                             for trial in self.trial_values])

        x = np.arange(1, np.size(values) + 1)
        # Plotting
        fig = plt.figure()

        plt.plot(x, values.T, marker='.', markersize=20, linewidth=3)
        plt.axvline(self.sobol_num, linewidth=2, linestyle='--', color='k')
        plt.title('Consecutive evaluations plot')
        plt.xlabel('Trial #')
        plt.ylabel('Objective value')
        plt.ylim(*plt.ylim())
        plt.xlim([1, np.size(values) + 1])

        self.save_plot('evaluations_plot', fig)
        plt.show()

    def plot_single_objective_distances(self):
        arms_by_trial = np.array([list(trial.arm.parameters.values())
                                  for trial in self.trial_values])

        # Distances between evaluations
        distances = np.linalg.norm(np.diff(arms_by_trial, axis=0), ord=2, axis=1)

        fig = plt.figure()

        plt.plot(np.arange(0, len(distances)), distances, linewidth=3, marker='.', markersize=20)
        plt.axvline(self.sobol_num, linewidth=2, linestyle='--', color='k')

        plt.title('Distances plot')
        plt.xlabel('Trial #')
        plt.ylabel('Distance |x[n]-x[n-1]|')

        self.save_plot('distances_plot', fig)
        plt.show()

    # Multiple Objective Plotting
    def plot_moo_trials(self):
        objective_names = [i.metric.name for i in self.objective.objectives]

        fig, axes = plt.subplots()
        df = self.trials_df
        objective_values = {i: df.get(i).values for i in objective_names}
        x, y = objective_values.values()

        axes.scatter(x, y, s=70, c=df.index, cmap='viridis')  # All trials
        fig.colorbar(axes.collections[0], ax=axes, label='trial #')

        # for idx, label in enumerate(df.index.values):
        #     axes.annotate(label, (x[idx], y[idx]))

        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])
        axes.set_title('Consecutive MOO Trials')
        fig.tight_layout()
        plt.show()

    def plot_posterior_pareto_frontier(self):
        objective_names = [i.metric.name for i in self.objective.objectives]
        frontier = compute_posterior_pareto_frontier(
            experiment=self.experiment,
            data=self.experiment.fetch_data(),
            primary_objective=self.objective.objectives[0].metric,
            secondary_objective=self.objective.objectives[1].metric,
            absolute_metrics=objective_names,  # we choose all metrics
            num_points=30,  # number of points in the pareto frontier
        )

        fig, axes = plt.subplots()
        axes.scatter(*[frontier.means[i] for i in objective_names], s=70, c='k')  # Pareto front

        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])
        axes.set_title('Posterior Pareto Frontier')
        fig.tight_layout()
        plt.show()

    # Plot utilities
    def clean_plot_dir(self):
        clean_directory(self.plot_dir)

    def save_plot(self, name, fig):
        save_name = os.path.join(self.plot_dir, name)
        if self.save_pdf:
            fig.savefig(f'{save_name}.pdf', transparent=False, bbox_inches='tight')

        if self.save_png:
            fig.savefig(f'{save_name}.png', transparent=False, bbox_inches='tight')
