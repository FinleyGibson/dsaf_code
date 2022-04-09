import sys
import os
import rootpath
sys.path.append(rootpath.detect())

import matplotlib

from testsuite.results import Result, ResultsContainer
from testsuite.analysis_tools import strip_problem_names

PATH_TO_PROCESSED_RESULTS = os.path.join(
    rootpath.detect(),
    "experiments/dsaf_vs_saf/analysis/processing_scripts/dual_hypervolume_processing/processed_results")

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


class Symbols:
    """
    utility class with all symbols defined for consistency. 
    """
    Pareto_front = r"$\mathcal{F}$"
    attinament_front = r"$\hat{A}_{DSAF}$"
    approx_Pareto_front = r"$\tilde{\mathcal{F}}_{DSAF}$"
    target = r"$\textbf{t}$"
    function = r"$f$"    
    
    attinament_front_ref = attinament_front[:-8]+"_{SAF}$"
    approx_Pareto_front_ref = approx_Pareto_front[:-8]+"_{SAF}$"
    
    lhs_samples = "LHS"
    
    @classmethod
    def target_n(cls, n):
        return cls.target[:-1]+"_{"+str(n)+"}$"
    
    @classmethod
    def function_n(cls, n):
        return cls.function[:-1]+"_{"+str(n)+"}$"
    

class Styles:
    """
    utility class with styles defined for consistency.
    """
    colours = {"result": "C0",
               "reference": "C1",
               "lhs": "C2",
               "target": "magenta"}
    
    points_lhs = {"c": colours["lhs"]}
    points_directed = {"c": colours["result"], "alpha": 0.5, "marker": "v"} 
    points_undirected = {"c": colours["reference"], "alpha": 0.5, "marker": "^"}
    points_target = {"c": colours["target"], "s":75, "marker": "x"}
    points_pareto_approx = {"c": colours["result"], "s": 25, "alpha": 0.5, "marker": "v"}
    points_pareto_approx_ref = points_pareto_approx.copy(); points_pareto_approx_ref.update(c=colours["reference"]); points_pareto_approx_ref.update(marker="^");
    
    line_attinament_front = {"c": colours["result"], "linestyle": "-", "linewidth": 5}
    line_attinament_front_ref = line_attinament_front.copy(); line_attinament_front_ref.update(c=colours["reference"])
    line_Pareto_front = {"c": "k", "linestyle": "-", "linewidth": 2}

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

    savedirs = [
        os.path.join(rootpath.detect(), "experiments/dsaf_vs_saf/analysis/results_analysis/figures/"),
        os.path.join(PATH_TO_REPORT_REPO, "figures/")]
    for d in savedirs:
        fig.savefig(os.path.join(d, filename+".png"), dpi=300, facecolor=None, edgecolor=None,
        orientation='portrait', pad_inches=0.12)
        fig.savefig(os.path.join(d, filename+".pdf"), dpi=300, facecolor=None, edgecolor=None,
        orientation='portrait', pad_inches=0.12)
        
        
def format_figures():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    #     matplotlib.rcParams['font.= 'Bitstream Vera Sans'
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('savefig', bbox='tight', transparent=True)


