from pathlib import Path
from datetime import datetime

from fpdf import FPDF
import os
import json
import csv
import uuid
from sys import exit
import glob
import subprocess


# Datestrings used in filenames
def get_datestring():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y at %H:%M:%S")
    dt_string2 = now.strftime('%d%m%Y%H%M%S')

    return dt_string, dt_string2


# Standalone functions
def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False


def clean_directory(dir_path):
    [f.unlink() for f in Path(dir_path).glob("*") if f.is_file()]


def clean_replay():
    directory = os.path.dirname(os.path.realpath(__file__))

    filelist = os.listdir(directory)

    for item in filelist:
        if '.rpy' in item or '.rec' in item:
            try:
                os.remove(os.path.join(directory, item))
            except OSError:
                pass


def generate_report(opt_config, means, best_parameters):
    pagewidth = 210
    margin = 50

    dt_string, dt_string2 = get_datestring()

    def create_title(pdf):
        pdf.set_font('Helvetica', 'B', 24)
        pdf.ln(15)
        pdf.write(5, 'Bayesian Optimization Report')
        pdf.ln(10)
        pdf.set_font('Helvetica', '', 16)
        pdf.write(5, f'Generated on {dt_string}')
        pdf.line(10, 45, 200, 45)
        pdf.ln(5)

    def create_section(pdf, section_title):
        pdf.ln(10)
        pdf.set_font('Helvetica', 'BU', 16)
        pdf.write(5, section_title)
        pdf.set_font('Helvetica', '', 12)
        pdf.ln(7)

    def create_subsection(pdf, section_title):
        pdf.set_font('Helvetica', 'B', 14)
        pdf.write(5, section_title)
        pdf.set_font('Helvetica', '', 12)
        pdf.ln(7)

    def create_textblock(pdf, line_dict):
        pdf.set_font('Helvetica', '', 12)
        for k, v in line_dict.items():
            pdf.set_font('Helvetica', 'BI', 12)
            pdf.write(5, f'{k}: ')
            pdf.set_font('Helvetica', '', 12)
            pdf.write(5, f'{v}')
            pdf.ln(5)
        pdf.ln(5)

    def draw_plots(pdf, plot_paths):

        plots = glob.glob(plot_paths)

        for idx, plot in enumerate(plots, start=1):
            plot_name = os.path.basename(plot).split('.')[0]
            create_subsection(pdf, plot_name)
            pdf.image(plot, x=margin / 2, y=None, w=pagewidth - margin)
            if idx % 2 == 0:
                pdf.add_page()
            else:
                pdf.ln(10)

    pdf = FPDF()

    plot_paths = os.path.join(os.getcwd(), 'reports', 'plots', '*.png')
    pdf.add_page()
    create_title(pdf)
    create_section(pdf, 'Configuration Data')
    create_textblock(pdf, opt_config)
    create_section(pdf, 'Optimisation Results')
    create_textblock(pdf, means)
    create_textblock(pdf, best_parameters)
    pdf.add_page()
    create_section(pdf, 'Plots')
    draw_plots(pdf, plot_paths)

    report_name = f'report_{dt_string2}.pdf'
    report_dir = os.path.join(os.getcwd(), 'reports')
    report_path = os.path.join(report_dir, report_name)

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    pdf.output(report_path, 'F')


# Simulation class

class Simulation:

    def __init__(self, model_dir, notebook_dir, exe_path, result_metrics):
        self.model_dir = model_dir

        self.notebook_dir = notebook_dir
        self.exe_path = exe_path
        self.result_metrics = result_metrics

        self.iterator = 0
        
        self.input_dir = os.path.abspath(os.path.join(self.model_dir, 'input_template.json'))
        

    def check_template(self):
        # check if the template exists and if not, create one
        template_filename = os.path.abspath(os.path.join(self.model_dir, 'input_template.json'))

        if not Path(template_filename).is_file():
            print('Generating templates:')
            arguments = [self.exe_path, '-t', self.notebook_dir, '-o', self.model_dir]
            print(' '.join(arguments))
            subprocess.run(arguments)

        self.input_dir = template_filename

    def get_results(self, parametrization):
        # unique filename generated on each call
        unique_filename = f"{uuid.uuid4()}[:10]"
        self.result_dir = os.path.abspath(os.path.join(self.model_dir, f'result_trial_{unique_filename}.txt'))
        # We first open the template
        with open(self.input_dir, 'r') as f:
            data = json.load(f)

        for param_name, param_value in parametrization.items():
            # indices in data dictionary with names corresponding to parameters in parametrization dict
            for x in data['inputs']:
                if x['name'] == param_name:
                    x['value'] = param_value
                    
                if x['name'] == 'result_path':
                    x['value'] = self.result_dir
        
        # We overwrite the input
        with open(self.input_dir, 'w') as f:
            json.dump(data, f, indent=4)

        # nTopCL arguments in a list
        arguments = [self.exe_path, "-j", self.input_dir, self.notebook_dir]

        # call nTopCl from cmd
        # print(" ".join(arguments))
        output, error = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

        # print output and errors from the cmd
        print(output.decode("utf-8"))

        # Read the results from a text file and return as dictionary        

        with open(self.result_dir, mode='r') as inp:
                reader = csv.reader(inp)                
                result_dict = {rows[0].strip('\''):float(rows[1]) for rows in reader}
        
        if set(self.result_metrics)==set(result_dict):
            return result_dict
        else:
            raise ValueError('Mismatch in result metric names from Ax and from nTop output file')
