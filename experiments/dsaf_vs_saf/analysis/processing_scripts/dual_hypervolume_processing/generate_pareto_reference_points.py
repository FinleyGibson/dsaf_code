import numpy as np
import os
import json
import rootpath
import wfg
from tqdm import tqdm
from testsuite.analysis_tools import draw_samples, dominates, \
    strip_problem_names, get_target_dict

D_n_to_draw = {2: 1000,
               3: 5000,
               4: 10000}


D_n_to_draw = {2: 10,
               3: 10,
               4: 10}

t_names = ["1_unattainable", "1_pareto", "1_attainable",
           "2_unattainable", "2_pareto", "2_attainable"]


def draw_points_which_dominate_target(func, n_obj, n_dim, n_to_draw, t):
    total_drawn = 0
    P_x, P_y = np.zeros((0, n_dim)), np.zeros((0, n_obj))
    pbar = tqdm(total=n_to_draw)
    while (P_y.shape[0]<n_to_draw) and (total_drawn < 50000000):
        n_left = n_to_draw-P_y.shape[0]
        pareto_x, pareto_y = draw_samples(func, n_obj, n_dim, 100, bar=False)
        dom_inds = np.asarray([dominates(pi, t) for pi in pareto_y]).reshape(-1)
        P_x = np.vstack([P_x, pareto_x[dom_inds]])
        P_y = np.vstack([P_y, pareto_y[dom_inds]])
        pbar.update(min(sum(dom_inds), n_left))
        total_drawn += 100
    return P_x[:n_to_draw], P_y[:n_to_draw]

D_targets = get_target_dict()

results_dir = os.path.join(rootpath.detect(),
                           "experiments/directed/data/directed")
problems = np.asarray(list(os.listdir(results_dir)))
# problems = ["wfg6_4obj_5dim"]
dims = np.asarray([strip_problem_names(r)[1] for r in problems])

save_path = "./dual_hypervolume_pareto_refpoints.json"
# for name in problems[np.argsort(dims)]:
for name in ["wfg2_4obj_10dim"]:
    print(name)
    n_prob, n_obj, n_dim = strip_problem_names(name)

    targets = D_targets[f"wfg{n_prob}_{n_obj}obj_{n_dim}dim"]

    if n_prob == 6 and n_obj == 4:
        n_dim = 5

    func = getattr(wfg, f"WFG{n_prob}")
    n_to_draw = D_n_to_draw[n_obj]

    for t_i, (t_name, target) in enumerate(zip(t_names, targets)):
        if t_name.split("_")[-1] == "attainable":
            key = name+"_"+t_name.split("_")[0]
            try:
                with open(save_path, "r") as infile:
                    D_save = json.load(infile)
            except:
                D_save = {}

            if key not in D_save.keys():
                if name == "wfg2_4obj_10dim":
                    # wfg2_4obj is unattainable
                    p_x = np.zeros((1, n_dim))
                    p_y = targets[t_i-1].reshape(1, -2)
                else:
                    p_x, p_y = draw_points_which_dominate_target(func, n_obj, n_dim,
                                                             n_to_draw, target)
                D_save[key] = (p_x.tolist(), p_y.tolist())

            with open(save_path, "w") as outfile:
                json.dump(D_save, outfile)