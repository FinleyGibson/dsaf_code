import os
import sys

import rootpath
from testsuite.analysis_tools import strip_problem_names, get_factors


# target_dir = os.raw_path.join(rootpath.detect(), "experiments/data/saf_directed/")
# target_dir = os.path.join(rootpath.detect(), "experiments/data/saf_undirected/")
target_dir = sys.argv[1]
assert os.path.isdir(target_dir)

problem_list = [
    'wfg5_2obj_6dim',
    'wfg6_2obj_6dim']

if __name__ == "__main__":
    for folder in problem_list:
        prob, obj, dim = strip_problem_names(folder)
        kf, lf = get_factors(obj, dim)

        with open("problem_setup_template") as infile:
            contents = infile.readlines()

        contents.insert(8, "M = {}".format(obj))
        contents.insert(9+1, "n_dim = {}".format(dim))
        contents.insert(10+2, "kfactor, lfactor = {}, {}".format(kf, lf))
        contents.insert(16+3, "func = getattr(wfg, 'WFG{}')".format(prob))

        os.makedirs(os.path.join(target_dir, folder))
        with open(os.path.join(target_dir, folder, "problem_setup.py"), "w") as f:
            contents = "".join(contents)
            f.write(contents)
