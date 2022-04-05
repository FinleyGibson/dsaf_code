import unittest
from testsuite.analysis_tools import get_target_dict, get_dhv_refpoint_dict
from parameterized import parameterized, parameterized_class
from testsuite.utilities import dominates
import numpy as np
import matplotlib.pyplot as plt

@parameterized_class([
    {"name": "wfg1_2obj_3dim"},
    {"name": "wfg1_3obj_4dim"},
    {"name": "wfg1_4obj_5dim"},
    {"name": "wfg2_2obj_6dim"},
    {"name": "wfg2_3obj_6dim"},
    {"name": "wfg2_4obj_10dim"},
    {"name": "wfg3_2obj_6dim"},
    {"name": "wfg3_3obj_10dim"},
    {"name": "wfg3_4obj_10dim"},
    {"name": "wfg4_2obj_6dim"},
    {"name": "wfg4_3obj_8dim"},
    {"name": "wfg4_4obj_8dim"},
    {"name": "wfg5_2obj_6dim"},
    {"name": "wfg5_3obj_8dim"},
    {"name": "wfg5_4obj_10dim"},
    {"name": "wfg6_2obj_6dim"},
    {"name": "wfg6_3obj_8dim"},
    {"name": "wfg6_4obj_10dim"}
])
class TestDominance(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.D_targets = get_target_dict()
        cls.D_refpoints = get_dhv_refpoint_dict()

        cls.target_index_converter = {1: 2,
                                      2: 5}

        cls.targets = np.asarray(cls.D_targets[cls.name])
        a = cls.targets[:2][np.argsort(cls.targets[:2, 0])]
        b = cls.targets[:2]
        # checks all targets are correctly ordered
        assert np.array_equal(a, b)

    @parameterized.expand([[1], [2]])
    def test_t_dominates_refpoints(self, ext):
        """
        check Pareto front reference points are dominated by t
        :param ext: int
            integer indicating which target position (1 or 2) is being
            tested
        :return:
        """
        # get the relevant targets and reference points
        t = self.targets[self.target_index_converter[ext]]
        rp = self.D_refpoints[self.name+f"_{ext}"]
        print(self.name, rp.shape)
        # check app points are dominated by t
        doms = [dominates(pi, t) for pi in rp]
        self.assertTrue(np.all(doms))

    @parameterized.expand([[1], [2]])
    def test_t_dominates_refpoints(self, ext):
        if self.name == "wfg2_4obj_10dim":
            print("Catch ")
        # get the relevant targets and reference points
        targets = np.asarray(self.D_targets[self.name])
        a = targets[:2][np.argsort(targets[:2, 0])]
        b = targets[:2]
        # checks all targets are correctly ordered
        self.assertTrue(np.array_equal(a, b))

        t = targets[self.target_index_converter[ext]-1]
        rp = self.D_refpoints[self.name+f"_{ext}"]
        # check app points are not dominated by t
        doms = [dominates(t, pi) for pi in rp]
        if not np.array_equal(t.reshape(-1), rp.reshape(-1)):
            try:
                self.assertLess(sum(doms) / len(doms), 0.01)
            except:
                self.plot(rp, t, doms)

        t = targets[self.target_index_converter[ext]-2]
        rp = self.D_refpoints[self.name+f"_{ext}"]
        # check app points are not dominated by t
        doms = [dominates(pi, t) for pi in rp]
        self.assertFalse(np.any(doms))

    def plot(self, rp, t, dom_ids):
        fig = plt.figure()
        ax = fig.gca()

        ax.scatter(*rp[dom_ids,:2].T, c="C3", s=5)
        ax.scatter(*rp[np.logical_not(dom_ids),:2].T, c="C0", s=5)
        ax.scatter(*t[:2], c="magenta", s=25)
        plt.show()

if __name__ == '__main__':
    unittest.main()
