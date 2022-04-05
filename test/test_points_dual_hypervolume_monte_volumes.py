import unittest
from parameterized import parameterized, parameterized_class
from testsuite.utilities import dominates
from testsuite.analysis_tools import strip_problem_names
from testsuite.analysis_tools import get_dual_hypervolume_refvol_points, \
    get_dhv_refpoint_dict, get_target_dict
from numpy.random import choice
import matplotlib.pyplot as plt

D_REF = get_dhv_refpoint_dict()
D_TAR = get_target_dict()
obj_montesample_conversion = {
    2: 5000,
    3: 50000,
    4: 100000
}

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
        cls.n_prob, cls.n_obj, cls.n_dim = strip_problem_names(cls.name)
        cls.n_samples = obj_montesample_conversion[cls.n_obj]

    @parameterized.expand([
        ["_2"],
        ["_5"]
    ])
    def test_attainable_targets(self, ext):
        ref_vola, ref_volb = get_dual_hypervolume_refvol_points(
            self.name,
            int(ext.strip("_")))
        self.assertEqual(ref_vola.shape[1], self.n_obj)
        self.assertEqual(ref_volb.shape[1], self.n_obj)

        try:
            self.assertEqual(ref_vola.shape[0],
                             obj_montesample_conversion[self.n_obj])
            self.assertEqual(ref_volb.shape[0],
                             obj_montesample_conversion[self.n_obj])
        except:
            # this instance is not practically attainable
            self.assertEqual(self.name, "wfg2_4obj_10dim")

    @parameterized.expand([
        ["_2", "_1"],
        ["_5", "_2"]
    ])
    def test_attainable_dominance(self, ext, t_pos):
        pareto_points = D_REF[self.name+t_pos]
        ref_vola, ref_volb = get_dual_hypervolume_refvol_points(
            self.name,
            int(ext.strip("_")))
        target = D_TAR[self.name][int(ext.strip("_"))]

        # check all points dominated by Pareto
        # tests only 100 random points for speed
        try:
            dom_p_mc_b = [dominates(pareto_points, p)
                          for p in ref_volb[choice(len(ref_volb), 100, False)]]
            self.assertTrue(all(dom_p_mc_b))
            dom_mc_t_b = [dominates(p, target) for p in
                          ref_volb[choice(len(ref_volb), 100, False)]]
            self.assertTrue(all(dom_mc_t_b))
        except ValueError:
            # this problem has an unattainable target
            self.assertEqual(self.n_prob, 2)
            self.assertEqual(self.n_obj, 4)

        dom_p_mc_a = [dominates(pareto_points, p) for p in
                      ref_vola[choice(len(ref_vola), 100, False)]]
        dom_t_mc_a = [dominates(target, p)
                      for p in ref_vola[choice(len(ref_vola), 100, False)]]
        self.assertTrue(all(dom_t_mc_a))
        self.assertTrue(all(dom_p_mc_a))

    @parameterized.expand([
        ["_0", "_1"],
        ["_1", "_1"],
        ["_3", "_2"],
        ["_4", "_2"]
    ])
    def test_unattainable_targets(self, ext, t_pos):
        ref_vola, ref_volb = get_dual_hypervolume_refvol_points(
            self.name,
            int(ext.strip("_")))
        self.assertEqual(ref_vola.shape[1], self.n_obj)
        self.assertEqual(ref_volb.shape[1], self.n_obj)
        target = D_TAR[self.name][int(ext.strip("_"))]

        self.assertEqual(ref_volb.shape[0], 0)
        if ext in ["_0", "_3"]:
            self.assertEqual(ref_vola.shape[0],
                             obj_montesample_conversion[self.n_obj])

    # @parameterized.expand([
    #     ["_0", "_1"],
    #     ["_1", "_1"],
    #     ["_3", "_2"],
    #     ["_4", "_2"]
    # ])
    # def test_attainable_dominance(self, ext, t_pos):
    #     pareto_points = D_REF[self.name+t_pos]
    #     ref_vola, ref_volb = get_dhv_refvol_points(self.name,
    #                                                int(ext.strip("_")))
    #     target = D_TAR[self.name][int(ext.strip("_"))]
    #
    #     # check all points dominated by Pareto
    #     # tests only 100 random points for speed
    #     dom_p_mc_a = [dominates(pareto_points, p) for p in ref_vola[choice(len(ref_vola), 100, False)]]
    #     dom_p_mc_b = [dominates(pareto_points, p) for p in ref_volb[choice(len(ref_volb), 100, False)]]
    #     self.assertTrue(all(dom_p_mc_a))
    #     self.assertTrue(all(dom_p_mc_b))
    #
    #
    #     dom_t_mc_a = [dominates(target, p) for p in ref_vola[choice(len(ref_vola), 100, False)]]
    #     dom_mc_t_b = [dominates(p, target) for p in ref_volb[choice(len(ref_volb), 100, False)]]
    #     self.assertTrue(all(dom_t_mc_a))
    #     self.assertTrue(all(dom_mc_t_b))


if __name__ == "__main__":
    unittest.main()
