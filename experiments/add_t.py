import os
from testsuite.utilities import TARGETS_D
import numpy as np
import pickle

FROM_DIR = "./data/dParEgo"
TO_DIR = "./data/d_ParEgo_with_t"


def load_pkl(path):
    with open(path, "rb") as infile:
        return pickle.load(infile)


def save_pkl(path, file):
    dir_path = "/".join(path.split("/")[:-1])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(path, "wb") as outfile:
        pickle.dump(file, outfile)

def get_target_from_path(path):
    prob_str = path.split("/")[2]
    target_list = TARGETS_D[prob_str]

    target_str = path.split("target")[1].split("rho")[0].replace("__", "_").split("_")
    target_str = [ts.replace("p", ".") for ts in target_str if ts != ""]
    target_from_str = np.array([float(ts) for ts in target_str])

    for target in target_list:
        if np.all(np.round(target, 2) == target_from_str):
            return target
    print("NO TARGET MATCHED!")
    raise KeyError

from_paths  = []
for dirpath, dirnames, filenames in os.walk(FROM_DIR, topdown=False):
    p = [os.path.join(dirpath, filename) for filename in filenames if filename[-11:] == "results.pkl"]
    from_paths += p

to_paths = [os.path.join(TO_DIR, p.split("dParEgo/")[1]) for p in from_paths]


for from_path, to_path in zip(from_paths, to_paths):

    # check paths match
    for a, b in zip(from_path.split("/")[2:], to_path.split("/")[2:]):
        assert a == b

    t = get_target_from_path(from_path)
    assert isinstance(t, list)

    result = load_pkl(from_path)
    result["targets"] = np.array(t).reshape(1, -1)
    save_pkl(to_path, result)
