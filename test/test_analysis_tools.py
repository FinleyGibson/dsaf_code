"""
tests for functions within utilities/analysis_tools.py
new tests added and updated as a when issues arise.
"""

import unittest
import numpy as np
from parameterized import parameterized_class
from itertools import product
from testsuite.utilities import PROBLEM_CONFIGURATIONS

from testsuite.analysis_tools import get_target_dict, get_dual_hypervolume_refvol_points, get_dhv_refpoint_dict, get_igd_refpoint_dict
@parameterized_class([
    {"fetch_function": get_target_dict},
    {"fetch_function": get_dhv_refpoint_dict},
    {"fetch_function": get_igd_refpoint_dict},

])
class TestGetRefPoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.ans = cls.fetch_function()

    def test_return_formatting(self):
        self.assertIsInstance(self.ans, dict)
        self.assertIsInstance(list(self.ans.values())[0], np.ndarray)


params_TestGetRefVol = [
    {"fetch_function": get_dual_hypervolume_refvol_points, "problem": p, "n_target": t}
    for p, t in product(PROBLEM_CONFIGURATIONS, range(6))
]


@parameterized_class(params_TestGetRefVol)
class TestGetRefVol(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.ans = cls.fetch_function(cls.problem, cls.n_target)

    def test_return_formatting(self):
        self.assertIsInstance(self.ans, tuple)
        # should return a tuple of np arrays with samples for each of
        # volume a and b
        self.assertIsInstance(self.ans[0], np.ndarray)
        self.assertEqual(self.ans[0].ndim, 2)
        self.assertEqual(self.ans[1].ndim, 2)

        if self.n_target not in [2, 5]:
            self.assertEqual(self.ans[1].shape[0], 0.)


if __name__ == "__main__":
    unittest.main()
