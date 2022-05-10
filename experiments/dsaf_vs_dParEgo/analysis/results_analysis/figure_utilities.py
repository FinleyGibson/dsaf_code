import sys
import os
import rootpath
sys.path.append(rootpath.detect())

import matplotlib

from testsuite.results import Result, ResultsContainer
from testsuite.analysis_tools import strip_problem_names

PATH_TO_PROCESSED_PAREGO_RESULTS = os.path.join(
    rootpath.detect(),
    "experiments/dsaf_vs_dParEgo/analysis/processing_scripts/dual_hypervolume_processing/processed_results")

PATH_TO_IGD_RESULTS = os.path.join(
    rootpath.detect(),
    "experiments/dsaf_vs_saf/analysis/processing_scripts/igd_reference_points/processed_results")

PATH_TO_REPORT_REPO = "/home/finley/phd/papers/gecco_2022/DSAF_EMO/"

RESULT_FILES_LIST = sorted(os.listdir(PATH_TO_PROCESSED_RESULTS))
UNATTAINABLE_RESULTS_FILE_LIST =  [str(file) for file in RESULT_FILES_LIST if int(file.split("_")[-1][:-5]) in [0, 3]]
PARETO_RESULTS_FILE_LIST =  [str(file) for file in RESULT_FILES_LIST if int(file.split("_")[-1][:-5]) in [1, 4]]
ATTAINABLE_RESULTS_FILE_LIST = [str(file) for file in RESULT_FILES_LIST if int(file.split("_")[-1][:-5]) in [2, 5]]

RESULT_FILES_LIST_2OBJ = [file for file in RESULT_FILES_LIST if strip_problem_names if strip_problem_names(file[:-7])[1] ==2]
RESULT_FILES_LIST_3OBJ = [file for file in RESULT_FILES_LIST if strip_problem_names if strip_problem_names(file[:-7])[1] ==3]
RESULT_FILES_LIST_4OBJ = [file for file in RESULT_FILES_LIST if strip_problem_names if strip_problem_names(file[:-7])[1] ==4]

CMAP = matplotlib.cm.tab10


def load_result(file_name):
    assert file_name in RESULT_FILES_LIST
    path = os.path.join(PATH_TO_PROCESSED_RESULTS, file_name)
    assert os.path.isfile(path)
    return ResultsContainer(path)

def load_igd_result(file_name):
    assert file_name in RESULT_FILES_LIST
    path = os.path.join(PATH_TO_IGD_RESULTS, file_name)
    assert os.path.isfile(path)
    return ResultsContainer(path)

def save_fig(fig, filename):
    pass
    # savedirs = [
    #     os.path.join(rootpath.detect(), "experiments/dsaf_vs_saf/analysis/results_analysis/figures/"),
    #     os.path.join(PATH_TO_REPORT_REPO, "figures/")]
    # for d in savedirs:
    #     fig.savefig(os.path.join(d, filename+".png"), dpi=300, facecolor=None, edgecolor=None,
    #     orientation='portrait', pad_inches=0.12)
    #     fig.savefig(os.path.join(d, filename+".pdf"), dpi=300, facecolor=None, edgecolor=None,
    #     orientation='portrait', pad_inches=0.12)
        
        
def format_figures():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    #     matplotlib.rcParams['font.= 'Bitstream Vera Sans'
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('savefig', bbox='tight', transparent=True)


