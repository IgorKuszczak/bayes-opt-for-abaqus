from pathlib import Path
from ax.plot.contour import _get_contour_predictions
from ax.plot.slice import _get_slice_predictions

import matplotlib.pyplot as plt
plt.interactive(False)
import numpy as np
import os

from .project_utils import clean_directory





class Plot:
    
    def __init__(self,sobol_num, plot_dir, save_pdf = False, save_png = False):
        
        self.plot_dir = plot_dir
        Path(plot_dir).mkdir(parents=True, exist_ok=True) 
        self.sobol_num = sobol_num
        self.figsize = (10,6)
        
        self.save_pdf = save_pdf
        self.save_png = save_png     
        
        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['axes.axisbelow'] = True
    
        
    def clean_plot_dir(self):
        clean_directory(self.plot_dir)
        
    def save_plot(self, plt, name):
        
        save_name = os.path.join(self.plot_dir,name)
        if self.save_pdf:
            plt.savefig(f'{save_name}.pdf', transparent=False, bbox_inches='tight')
            
        if self.save_png:    
            plt.savefig(f'{save_name}.png', transparent=False, bbox_inches='tight')
        
    
    def plot_convergence_plt(self, y):
        
        x = np.arange(0,y.shape[0])
        
        # Convergence plot
        fig, ax = plt.subplots(figsize=self.figsize)       
        ax.plot(x,y.T,c='blue',linewidth=2,zorder = -1)
        # ax.scatter(x,y.T,c='b',marker = 'o',s = 50, label='Feasible solution')
        
        ymin,ymax = plt.ylim()
        plt.vlines(self.sobol_num, ymin, ymax, colors='green',linewidth=2, linestyles='dashed')
        plt.ylim(ymin,ymax)
        # plt.grid(which='both')
        
        plt.xlabel('Trial')
        plt.ylabel('Best objective')
        plt.show()
        
        
        self.save_plot(plt,'convergence_plot')
        
        
        
    def plot_evaluations_plt(self, best_objectives, mask):     
        
        x = np.arange(0,best_objectives.shape[0])
        
        # Consecutive evaluations
        fig, ax = plt.subplots(figsize=self.figsize)       
        ax.plot(x,best_objectives.T,c='black',linewidth=2,zorder=0)
        ax.scatter(x[mask],best_objectives.T[mask],c='b',edgecolor = 'black', marker = '*', s = 300,label='Feasible solution',zorder=2) # Feasible points
        
        if not np.all(mask==True):
            ax.scatter(x[~mask],best_objectives.T[~mask],c='r', marker = 'o',edgecolor='black', s = 40,label = 'Infeasible solution',zorder=1) # Infeasible points
            plt.legend()
            
        ymin,ymax = plt.ylim()
        plt.vlines(self.sobol_num, ymin, ymax, colors='green',linewidth=2, linestyles='dashed')
        plt.ylim(ymin,ymax)  
        
        plt.xlabel('Trial')
        plt.ylabel('Objective')
        plt.show()
        
        self.save_plot(plt,'evaluations_plot')
        
    def plot_distances_plt(self, distances):
        
        # Distances between evaluations
        plt.figure(figsize=self.figsize)
        plt.plot(np.arange(0, len(distances)), distances,'k-', linewidth=2)
        ymin,ymax = plt.ylim()
        plt.vlines(self.sobol_num, ymin, ymax, colors='green',linewidth=2, linestyles='dashed')
        plt.ylim(ymin,ymax)
        plt.xlabel('Trial')
        plt.ylabel('Distance |x[n]-x[n-1]|')
        plt.show()
        # plt.grid()
        
        
        self.save_plot(plt,'distances_plot')
        
    def plot_contour_plt(self, model, param_x, param_y, metric_name, best_parameters, density = 50):
        
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
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=self.figsize)
        cont1 = axes[0].contourf(X, Y, Z_f, 20, cmap='viridis')
        fig.colorbar(cont1, ax=axes[0])
        axes[0].set_title('Mean')
        axes[0].plot(evaluations[:,0],
                        evaluations[:,1],
                        'o',
                        markersize = 12,
                        mfc='white',
                        mec='black')
        
        axes[0].plot(best_parameters[param_x],
                     best_parameters[param_y],
                     'o',
                     markersize = 13,
                     mfc='red',
                     mec='black')
    
        
        
        cont2 = axes[1].contourf(X, Y, Z_sd, 20, cmap='plasma')
        fig.colorbar(cont2, ax=axes[1])
        axes[1].set_title('Standard Deviation')
        axes[1].plot(evaluations[:,0],
                        evaluations[:,1],
                        'o',
                        markersize =12,
                        mfc='white',
                        mec='black')
        
        axes[1].plot(best_parameters[param_x],
                     best_parameters[param_y],
                     'o',
                     markersize = 13,
                     mfc='red',
                     mec='black')
        
        for axs in axes.flat:
            axs.set(xlabel=param_x, ylabel=param_y)
        
        fig.tight_layout()
        
        self.save_plot(plt,'contours_plot')
        
        
    # def plot_slice(self, model, param_name, metric_name,density=50):     
                   
    #     # pd, cntp, f_plt, rd, grid, _, _, _, fv, sd_plt, ls  =
        
    #     return _get_contour_predictions(
    #     model=model,
    #     param_name=param_name,
    #     metric_name=metric_name,
    #     generator_runs_dict=None,
    #     relative=False,
    #     density=density,
    # )
        
    #     # plot_data_dict[metric_name] = pd
    #     # raw_data_dict[metric_name] = rd
    #     # cond_name_to_parameters_dict[metric_name] = cntp
    
    #     # sd_plt_dict[metric_name] = np.sqrt(cov[metric_name][metric_name])
    #     # is_log_dict[metric_name] = ls
    
    #     # fig, axes = plt.subplots(figsize=self.figsize)
    #     # plt.plot(grid_x, f_plt)
    #     # plt.fill_between(param_x,f_plt-sd_plt,f_plt+sd_plt,
    #     #                  facecolor="orange", # The fill color
    #     #                  color='blue',       # The outline color
    #     #                  alpha=0.2)          # Transparency of the fill
        
        
        

    

    

    
