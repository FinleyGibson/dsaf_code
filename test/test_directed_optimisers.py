import unittest
from testsuite.directed_optimisers import DirectedWHedge
from testsuite.surrogates import GP, MultiSurrogate
import numpy as np
import matplotlib.pyplot as plt


def obj_f(x):
    return np.array([sum([xi*((-1)**i)+abs((xi+2)*5) for i, xi in enumerate(x)]),\
           sum([xi*((-1)**i) for i, xi in enumerate(x)])])


class TestDirectedWHedge(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.n_dim = 2
        cls.n_obj = 2
        cls.target = np.random.uniform(0, 1, cls.n_obj).reshape(1, -1)
        cls.w = [0.2, 0.4, 0.6, 0.8]
        cls.eta = 0.5
        cls.limits = [[0]*cls.n_dim, [1]*cls.n_dim]
        cls.opt = DirectedWHedge(
            objective_function=obj_f,
            limits=cls.limits,
            targets=cls.target,
            eta=cls.eta,
            surrogate=MultiSurrogate(GP, scaled=True),
            n_initial=10,
            w=cls.w
            )

        # - step
        # - get_next_x  -> updates surrogate
        #               -> applies cmaes to alpha
        # - alpha       -> makes prediciton
        #               -> applies scalarise_y



    def test_one(self):
        self.opt.step()
    # @classmethod
    # def tearDownClass(cls) -> None:
    #     """
    #     runs once at the end of the tests in this class
    #     :return: None
    #     """
    #     print("All tests done!")
    #
    # def setUp(self) -> None:
    #     """
    #     runs the beginning of every test method below
    #     :return: None
    #     """
    #     print("Setting up test.")
    #
    # def tearDown(self) -> None:
    #     """
    #     runs at the end of each test method
    #     :return:
    #     """
    #     print("Tearing down test.")
    #
    # def test_one(self):
    #     self.assertEqual(self.a, "attribute a")
    #     self.assertEqual(self.b, "attribute b")


if __name__ == "__main__":
    x = np.random.uniform(-2, 5, (20, 2))
    print("Ping")
    print(x.shape)
