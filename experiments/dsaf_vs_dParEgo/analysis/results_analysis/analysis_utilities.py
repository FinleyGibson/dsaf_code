import os
import sys
import rootpath
sys.path.append(rootpath.detect())
import matplotlib

from testsuite.results import Result, ResultsContainer

PATH_TO_REPORT_REPO = "/home/finley/phd/papers/gecco_2022/DSAF_EMO/"
PATH_TO_PROCESSED_PAREGO_RESULTS = os.path.join(
    rootpath.detect(),
    "experiments/dsaf_vs_dParEgo/analysis/processing_scripts/"
    "dual_hypervolume_processing/processed_results")
PATH_TO_PROCESSED_DSAF_RESULTS = os.path.join(
    rootpath.detect(),
    "experiments/dsaf_vs_saf/analysis/processing_scripts/"
    "dual_hypervolume_processing/processed_results")
PATH_TO_PROCESSED_PAREGO_IGD = os.path.join(
    rootpath.detect(),
    "experiments/dsaf_vs_dParEgo/analysis/processing_scripts/"
    "igd_processing/processed_results")
PATH_TO_PROCESSED_DSAF_IGD = os.path.join(
    rootpath.detect(),
    "experiments/dsaf_vs_saf/analysis/processing_scripts/"
    "igd_reference_points/processed_results")

assert os.path.isdir(PATH_TO_REPORT_REPO)
assert os.path.isdir(PATH_TO_PROCESSED_PAREGO_RESULTS)
assert os.path.isdir(PATH_TO_PROCESSED_DSAF_RESULTS)

PAREGO_FILES_LIST = sorted(os.listdir(PATH_TO_PROCESSED_PAREGO_RESULTS))
DSAF_FILES_LIST = sorted(os.listdir(PATH_TO_PROCESSED_DSAF_RESULTS))

PAREGO_IGD_LIST = sorted(os.listdir(PATH_TO_PROCESSED_PAREGO_IGD))
DSAF_IGD_LIST = sorted(os.listdir(PATH_TO_PROCESSED_DSAF_IGD))

UNATTAINABLE_PAREGO_FILE_LIST = [str(file) for file in PAREGO_FILES_LIST if
                                 int(file.split("_")[-1][:-5]) in [0, 3]]
PARETO_PAREGO_FILE_LIST = [str(file) for file in PAREGO_FILES_LIST if
                           int(file.split("_")[-1][:-5]) in [1, 4]]
ATTAINABLE_PAREGO_FILE_LIST = [str(file) for file in PAREGO_FILES_LIST
                               if int(file.split("_")[-1][:-5]) in [2, 5]]


UNATTAINABLE_PAREGO_IGD_LIST = [str(file) for file in PAREGO_IGD_LIST if
                                 int(file.split("_")[-1][:-5]) in [0, 3]]
PARETO_PAREGO_IGD_LIST = [str(file) for file in PAREGO_IGD_LIST if
                           int(file.split("_")[-1][:-5]) in [1, 4]]
ATTAINABLE_PAREGO_IGD_LIST = [str(file) for file in PAREGO_IGD_LIST
                               if int(file.split("_")[-1][:-5]) in [2, 5]]


UNATTAINABLE_DSAF_FILE_LIST = [str(file) for file in DSAF_FILES_LIST if
                               int(file.split("_")[-1][:-5]) in [0, 3]]
PARETO_DSAF_FILE_LIST = [str(file) for file in DSAF_FILES_LIST if
                         int(file.split("_")[-1][:-5]) in [1, 4]]
ATTAINABLE_DSAF_FILE_LIST = [str(file) for file in DSAF_FILES_LIST if
                             int(file.split("_")[-1][:-5]) in [2, 5]]

UNATTAINABLE_DSAF_IGD_LIST = [str(file) for file in DSAF_IGD_LIST if
                               int(file.split("_")[-1][:-5]) in [0, 3]]
PARETO_DSAF_IGD_LIST = [str(file) for file in DSAF_IGD_LIST if
                         int(file.split("_")[-1][:-5]) in [1, 4]]
ATTAINABLE_DSAF_IGD_LIST = [str(file) for file in DSAF_IGD_LIST if
                             int(file.split("_")[-1][:-5]) in [2, 5]]

def load_result(file_name, path_to_results):
    """
    load file from specified directory.
    """
    path = os.path.join(path_to_results, file_name)
    assert os.path.isfile(path)
    return ResultsContainer(path)


def save_table(df, file_name, index=True):
    savedirs = [
        os.path.join(rootpath.detect(), "experiments/dsaf_vs_dParEgo/analysis/results_analysis/tables"),
        os.path.join(PATH_TO_REPORT_REPO, "tables/")]
    for d in savedirs:
        file_path = os.path.join(d, file_name+".tex")
        with open(file_path, "w") as outfile:
            print(df.to_latex(index=index, escape=False), file=outfile)

def format_figures():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    #     matplotlib.rcParams['font.= 'Bitstream Vera Sans'
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('savefig', bbox='tight', transparent=True)

def save_fig(fig, filename):
    savedirs = [
        os.path.join(rootpath.detect(), "experiments/dsaf_vs_dParEgo/analysis/results_analysis/figures/"),
        os.path.join(PATH_TO_REPORT_REPO, "figures/")]
    for d in savedirs:
        fig.savefig(os.path.join(d, filename+".png"), dpi=300, facecolor=None, edgecolor=None,
        orientation='portrait', pad_inches=0.12)
        fig.savefig(os.path.join(d, filename+".pdf"), dpi=300, facecolor=None, edgecolor=None,
        orientation='portrait', pad_inches=0.12)


class Symbols:
    """
    utility class with all symbols defined for consistency.
    """
    Pareto_front = r"$\mathcal{F}$"
    attinament_front = r"$\hat{A}_{DSAF}$"
    approx_Pareto_front = r"$\tilde{\mathcal{F}}_{DSAF}$"
    parego = r"$\tilde{\mathcal{F}}_{ParEGO}$"
    target = r"$\textbf{t}$"
    function = r"$f$"

    attinament_front_ref = attinament_front[:-8] + "_{SAF}$"
    approx_Pareto_front_ref = approx_Pareto_front[:-8] + "_{SAF}$"

    lhs_samples = "LHS"

    @classmethod
    def target_n(cls, n):
        return cls.target[:-1] + "_{" + str(n) + "}$"

    @classmethod
    def function_n(cls, n):
        return cls.function[:-1] + "_{" + str(n) + "}$"


class Styles:
    """
    utility class with styles defined for consistency.
    """
    colours = {"result": "C0",
               "reference": "C1",
               "lhs": "C4",
               "target": "magenta",
               "parego": "C2"}

    points_lhs = {"c": colours["lhs"]}
    points_parego = {"c": colours["parego"], "alpha": 0.5, "marker": ">"}
    points_directed = {"c": colours["result"], "alpha": 0.5, "marker": "v"}
    points_undirected = {"c": colours["reference"], "alpha": 0.5,
                         "marker": "^"}
    points_target = {"c": colours["target"], "s": 75, "marker": "x"}
    points_pareto_approx = {"c": colours["result"], "s": 25, "alpha": 0.5,
                            "marker": "v"}
    points_pareto_approx_ref = points_pareto_approx.copy();
    points_pareto_approx_ref.update(c=colours["reference"]);
    points_pareto_approx_ref.update(marker="^");

    line_attinament_front = {"c": colours["result"], "linestyle": "-",
                             "linewidth": 5}
    line_attinament_front_ref = line_attinament_front.copy();
    line_attinament_front_ref.update(c=colours["reference"])
    line_Pareto_front = {"c": "k", "linestyle": "-", "linewidth": 2}

