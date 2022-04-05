"""

"""
import unittest
import matplotlib.pyplot as plt
from parameterized import parameterized, parameterized_class
from itertools import product

from testsuite.analysis_tools import strip_problem_names, \
    get_igd_refpoint_dict, get_target_dict
import wfg

REFERENCE_POINTS = get_igd_refpoint_dict()
TARGETS = get_target_dict()
PROBLEMS = sorted(list(REFERENCE_POINTS.keys()))

pn = 104
@parameterized_class(('problem', 't_i'), list(product(PROBLEMS, range(6)))[pn:pn+1])
class TestTargetDominance(unittest.TestCase):

    def setUp(self) -> None:
        """
        sets up for each test with all unique problem/target combinations
        """
        self.prob, self.obj, self.dim = strip_problem_names(self.problem)
        try:
            target = TARGETS[self.problem][self.t_i:self.t_i+1]
        except KeyError:
            target = TARGETS[f'ellipsoid_{self.obj}obj'][self.t_i:self.t_i+1]

        ref_points = REFERENCE_POINTS[self.problem]

        assert target.shape[0] == 1
        assert ref_points.shape[1] == target.shape[1]

        func = getattr(wfg, f"WFG{self.prob}")

        self.func = func
        self.target = target
        self.ref_points = ref_points

    def test_target_dominance_Pareto_front(self):
        print(f"{self.problem}: {self.prob}{self.obj}{self.dim}")
        print("rp: ", self.ref_points.shape)
        print("t: ", self.target.shape)

        plt.scatter(*self.ref_points[:, 2:].T)
        plt.scatter(*self.target[:,2:].T, c="magenta")
        plt.show()





