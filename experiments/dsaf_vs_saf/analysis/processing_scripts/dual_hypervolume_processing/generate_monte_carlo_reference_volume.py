"""
generates samples within reference volumes A and B for all targets on
all problems

"""
import os
import json
import numpy as np
import time

from testsuite.analysis_tools import get_target_dict, get_dhv_refpoint_dict,\
    get_igd_refpoint_dict, strip_problem_names
from testsuite.utilities import str_format, dominates
from testsuite.results import ResultsContainer


D_TARGETS = get_target_dict()
D_DHV_REF = get_dhv_refpoint_dict()
D_IGD_REF = get_igd_refpoint_dict()
RESULTS_PATH = '../../../data/directed/'

N_MONTE_SAMPLES = {2: 5000,
                   3: 50000,
                   4: 100000,
}


def check_results(path):
    with open(path, "r") as infile:
        D = json.load(infile)
    return list(sorted(D.keys()))


def draw_samples_between_points(n, ideal, ref):
    """
    draw n uniform samples from the hypersquare between ideal and ref
    :param n:
    :param ideal:
    :param ref:
    :return:
    """
    return np.vstack([np.random.uniform(l, u, n)
                      for l, u in zip(ideal, ref)]).T


def draw_nondom_samples_between_points(n, ideal, ref, P):
    points = np.zeros((0, P.shape[1]))
    n_required = n
    frac_in = 1.
    while points.shape[0] < n:
        print(f"ping {points.shape[0]}/{n}")
        new_points = draw_samples_between_points(int(n_required*frac_in*1.1),
                                                 ideal, ref)
        inds = [dominates(P, pi) for pi in new_points]
        new_points = new_points[inds]
        frac_in = n_required/new_points.shape[0]
        n_required = int(n-new_points.shape[0])
        points = np.vstack((points, new_points))

    return points[:n]

# A = draw_samples_between_points(n_to_draw_a, target, vol_rp)
# dhv_points = D_DHV_REF[problem + f"_{tgmp[t_i]}"]
# if attainable:
#     B = draw_samples_between_points(n_to_draw_b, vol_ip, target)
#     inds = [dominates(dhv_points, pi) for pi in B]
#     B = B[inds]
# else:
#     inds = [dominates(dhv_points, pi) for pi in A]
#     A = A[inds]
#     B = np.zeros(0).reshape(0, n_obj)
#

def calculate_volume_reference(igd_rp, results):
    results_y = np.vstack(results.y)
    reference_p = np.vstack([r.p for r in results.reference])
    targets = np.vstack(results.targets)
    return np.vstack((igd_rp, results_y, reference_p, targets)).max(axis=0)*1.1


def calculate_volume_ideal(dhv_rp):
    return dhv_rp.min(axis=0)


def load_json_from_path(file_path):
    with open(file_path, "r") as infile:
        A, B = json.load(infile)

    A, B = np.asarray(A), np.asarray(B)
    if B.ndim == 1 and len(B) == 0:
        return A, B.reshape(0, A.shape[1])
    else:
        return A, B


def worker(problem):
    n_prob, n_obj, n_dim = strip_problem_names(problem)
    targets = D_TARGETS[problem]
    # check targets are ordered correctly
    assert np.array_equal(targets[:2][np.argsort(targets[:2][:, 0])], targets[:2])
    assert np.array_equal(targets[3:][np.argsort(targets[3:][:, 0])], targets[3:])

    problem_dir = os.path.join(RESULTS_PATH, problem)
    assert os.path.isdir(problem_dir)
    for t_i, target in enumerate(targets):
        tic = time.time()
        outfile_path = f"./dual_hypervolume_volume_monte_samples/wfg{n_prob}_{n_obj}obj_{t_i}.json"
        if os.path.isfile(outfile_path):
            # load if it is already done and check enough samples have been drawn.
            A_, B_ = load_json_from_path(outfile_path)
            print(f"existing file loaded from {outfile_path}")
        else:
            A_ = np.zeros((0, n_obj))
            B_ = np.zeros((0, n_obj))

        print(f"Processing wfg{n_prob}_{n_obj}obj_{t_i}...")
        # only the third and 6th targets are attainable, except for wfg2_4obj,
        # where none are attainable
        if problem == "wfg2_4obj_10dim":
            attainable = False
        else:
            attainable = True if t_i in [2, 5] else False

        if A_.shape[0] < N_MONTE_SAMPLES[n_obj]:
            n_to_draw_a = N_MONTE_SAMPLES[n_obj] - A_.shape[0]
        else:
            n_to_draw_a = 0

        if attainable and B_.shape[0] < N_MONTE_SAMPLES[n_obj]:
            n_to_draw_b = N_MONTE_SAMPLES[n_obj]-B_.shape[0]
        else:
            n_to_draw_b = 0

        if n_to_draw_a == 0 and n_to_draw_b == 0:
            print(f"No points to be added for {problem}")
            continue
        else:
            print(f"{problem} requires {n_to_draw_a}+{n_to_draw_b} further samples")

        # get dir extension for target from target value. This is messy,
        # but works
        target_str = str_format(np.round(target, 2))
        problem_target_dir = [i for i in os.listdir(problem_dir + "/log_data") if "_"+target_str+"__" == i.split("target")[-1].split("w")[0]]
        if problem_target_dir == []:
            raise Exception("Something is wrong with result directories: targets do not match")
        else:
            problem_target_dir = problem_target_dir[0]

        result_path = os.path.join(problem_dir, "log_data/", problem_target_dir)
        assert os.path.isdir(result_path)
        reference_path = os.path.join("../../../data/undirected_comp", problem, "log_data")
        reference_path = os.path.join(reference_path, os.listdir(reference_path)[0])

        results = ResultsContainer(result_path)
        results.add_reference_data(reference_path)

        vol_rp = calculate_volume_reference(D_IGD_REF[problem], results)
        rp =D_IGD_REF[problem].max(axis=0)
        Rp = np.vstack(results.y).max(axis=0)

        tgmp = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}
        if attainable:
            vol_ip = calculate_volume_ideal(D_DHV_REF[problem+f"_{tgmp[t_i]}"])
        else:
            vol_ip = target

        volume_estimate_a = np.prod(vol_rp-target)
        volume_estimate_b = np.prod(target-vol_ip)

        split_frac = volume_estimate_b/volume_estimate_a
        if not attainable:
            assert split_frac == 0.

        dhv_points = D_DHV_REF[problem + f"_{tgmp[t_i]}"]
        A = draw_nondom_samples_between_points(n_to_draw_a, target, vol_rp, dhv_points)
        assert A.shape[0] == n_to_draw_a
        A = np.vstack((A_, A))
        if attainable:
            B = draw_nondom_samples_between_points(n_to_draw_b, vol_ip, target, dhv_points)
            assert B.shape[0] == n_to_draw_b
            B = np.vstack((B_, B))
        else:
            B = B_

        try:
            with open(outfile_path, "w") as outfile:
                json.dump((A.tolist(), B.tolist()), outfile)
            print(f"Saved wfg{n_prob}_{n_obj}obj_{t_i}.json")
            print(f"{A.shape[0]} points in vol A and {B.shape[0]} points in vol B")
        except Exception as e:
            print(f"ERROR {problem}_{t_i} SAVING FAILED")
            print(e)
        print("TIME TAKEN: ", time.time()-tic)
        print()


if __name__ == "__main__":
    from multiprocessing import Pool

    # for k in D_TARGETS.keys():
    #     worker(k)
    with Pool(4) as p:
        p.map(worker, [i for i in D_TARGETS.keys()])
