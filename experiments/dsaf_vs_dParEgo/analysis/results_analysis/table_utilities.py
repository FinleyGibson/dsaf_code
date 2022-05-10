import sys
import os
import rootpath
sys.path.append(rootpath.detect())

import numpy as np
import pandas


PATH_TO_REPORT_REPO = "/home/finley/phd/papers/gecco_2022/DSAF_EMO/"

def save_table(df, file_name):
    savedirs = [
        os.path.join(rootpath.detect(), "experiments/directed/analysis/results_tables/paper_figure_notebooks/tables/"),
        os.path.join(PATH_TO_REPORT_REPO, "tables/")]
    for d in savedirs:
        file_path = os.path.join(d, file_name+".tex")
        with open(file_path, "w") as outfile:
            print(df.to_latex(index=True, escape=False), file=outfile)
        
        
def get_comparison_matrices(result, intervals=None, attainable=False):
    """
    
    """
    if attainable:
        at_ind = 1
    else:
        at_ind = 0
        
    if intervals is None:
        r = result.dual_hpv_history[at_ind]
        r_ref = result.dual_hpvref_history[at_ind]
    else:
        assert isinstance(intervals, list), "intervals must be a list. For a single interval n supply [n]"
        assert all([i in result.dual_hpv_hist_x for i in intervals]), "Not all supplied intervals exist in results"
        inds = [np.where(np.asarray(result.dual_hpv_hist_x)==i)[0][0] for i in intervals]
        r = result.dual_hpv_history[at_ind][:, inds]
        r_ref = result.dual_hpvref_history[at_ind][:, inds]
        
    M = (r>r_ref)
    M_ref = (r<r_ref)
    M_draw = (r==r_ref)
    return M, M_ref, M_draw