import os
from testsuite.utilities import TARGETS_D
import numpy as np
import shutil

SOURCE_DIR1 = "./data/dParEgo"
SOURCE_DIR2 = "./data/from_server"
TARGET_DIR = "./data/dParEgo_combined"

def get_seed_from_path(path):
    seed = path.split("seed")[1][1:3]
    return seed


for d1, d2 in zip(sorted(os.listdir(SOURCE_DIR1)),
                  sorted(os.listdir(SOURCE_DIR2))):
    assert d1 == d2
    folder1 = os.path.join(SOURCE_DIR1, d1, "log_data/")
    folder2 = os.path.join(SOURCE_DIR2, d2, "log_data/")

    for target_1, target_2 in zip(os.listdir(folder1), os.listdir(folder2)):
        path_1 = os.path.join(folder1, target_1)
        path_2 = os.path.join(folder2, target_2)
        matches = [a == b for a, b in zip(path_1.split("/"), path_2.split("/"))]
        # check strings match
        assert len(matches) == sum(matches)+1

        seeds1 = sorted([int(get_seed_from_path(p)) for p in os.listdir(path_1) if p[-11:] == "results.pkl"])
        seeds2 = sorted([int(get_seed_from_path(p)) for p in os.listdir(path_2) if p[-11:] == "results.pkl"])
        print(seeds1)
        print(seeds2)
        try:
            assert len(seeds1) == len(set(seeds1))
        except AssertionError:
            print(path_1)
            print("ping")

        try:
            assert len(seeds2) == len(set(seeds2))
        except AssertionError:
            print(path_2)
            print("ping")

        try:
            assert len(seeds1) == 11
        except AssertionError:
           print(path_1)
           print("ping")

        try:
            assert len(seeds2) == 20
        except AssertionError:
            print(path_2)
            print("ping")

        a = np.array(sorted(seeds1+seeds2))
        b = np.arange(31)
        assert np.all(a==b)

for dirpath, dirnames, filenames in os.walk(SOURCE_DIR1, topdown=False):
    from_paths = [os.path.join(dirpath, filename) for filename in filenames if filename[-11:] == "results.pkl"]
    to_paths = ["/".join(p.split("/")[:2]+["dParEgo_combined"]+p.split("/")[3:]) for p in from_paths]


    for fr, to in zip(from_paths, to_paths):
        assert os.path.isfile(fr)
        shutil.copyfile(fr, to)

    

#
    #
    #     p1 = [os.path.join(dirpath1, filename) for filename in filenames1 if
    #           filename[-11:] == "results.pkl"]
    #
    #     seeds1 = [get_seed_from_path(pi) for pi in p1]
    #     # check no repeats in first dir
    #     assert len(seeds1) == len(set(seeds1))
    #
    # for dirpath2, dirnames2, filenames2 in os.walk(folder2, topdown=False):
    #     p2 = [os.path.join(dirpath2, filename) for filename in filenames2 if
    #           filename[-11:] == "results.pkl"]
    #
    #     seeds2 = [get_seed_from_path(pi) for pi in p1]
    #     # check no repeats in first dir
    #     assert len(seeds2) == len(set(seeds2))
    #
    # print(seeds1)
    # print(seeds2)
    # print()
