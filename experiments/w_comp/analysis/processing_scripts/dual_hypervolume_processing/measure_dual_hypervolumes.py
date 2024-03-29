import os
import rootpath
import numpy as np
from testsuite.utilities import PROBLEM_CONFIGURATIONS, TARGETS_D
from testsuite.results import ResultsContainer
from testsuite.analysis_tools import map_target_to_target_n, strip_problem_names
from multiprocessing import Pool

def get_dir_from_target(target, target_dirs):
    """
    extracts directory from target_dirs if it matches target
    :param target: np.array - shape: (n,)
        target as a numpy array
    :param target_dirs: list(str)
        list of strings containing paths to target specific data directories.
    :return: str
        string from target_dirs
    """

    a = str(target.round(2)).replace("[", "").replace("]", "").replace(".",
                                                                       "p").replace(
        " ", "")
    for target_dir in target_dirs:
        b = target_dir.split("target")[-1]
        b = b.split("rho")[0]
        b = b.replace("_", "")
        b = b.split("w")[0]
        if a == b:
            print("Ping!!")
            return target_dir
    return None


def get_refdir_from_name(problem_name, ref_dirs):
    for rd in ref_dirs:
        rd_formatted = rd.split("/")[-1]
        if rd_formatted == problem_name:
            data_path = os.path.join(rd, "log_data")
            final_dir = os.listdir(data_path)[0]
            return os.path.join(data_path, final_dir)
    return None


def worker(problem_name, target):
    # problem_name, target = arg
    target = target.reshape(1, -1)
    # generate path to save file
    target_n = map_target_to_target_n(target)
    file_name = problem_name + f"_{target_n}"
    save_path = os.path.join(SAVE_DIR, file_name) + ".json"

    path_to_result_parent = os.path.join(
        rootpath.detect(),
        "experiments/data/w_0p95/"+problem_name+"/log_data/")
    assert os.path.isdir(path_to_result_parent)

    path_to_results = [os.path.join(path_to_result_parent, path)
                       for path in os.listdir(path_to_result_parent)]

    if os.path.isfile(save_path):
        print(f"Existing results found for {problem_name}")
    else:
        print(f"No existing results found for {problem_name}")
        # generate targeted results
        result_dir = get_dir_from_target(target, path_to_results)
        assert os.path.isdir(result_dir)
        results = ResultsContainer(result_dir)
        print(f"New results file generated at {result_dir}")

        # # add reference data
        # ref_dir = get_refdir_from_name(problem_name, REF_DIRS)
        # results.add_reference_data(ref_dir)

        # get reference volume samples
        results.compute_dual_hypervolume_history(sample_freq=LOG_INTERVAL)
        print("saving to:\t", save_path)
        results.save(save_path)


PROBLEM_CONFIGURATIONS = [
    "wfg5_2obj_6dim",
    "wfg6_2obj_6dim"
]

SAVE_DIR = "./processed_results/w0p95"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
# LOG_INTERVAL = [10, 25, 50, 100, 150]
LOG_INTERVAL = np.arange(10, 155, 5).tolist()

config_pairs = []
for problem_name in PROBLEM_CONFIGURATIONS:
    problem_name = problem_name if problem_name[-1] == "/" else problem_name+"/"
    problem_name = problem_name.split("/")[-2]

    targets = np.asarray(TARGETS_D[problem_name])
    for t in targets:
        config_pairs.append((problem_name, t))


with Pool(4) as p:
    p.starmap(worker, config_pairs)
