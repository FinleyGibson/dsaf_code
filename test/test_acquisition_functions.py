import unittest
from parameterized import parameterized, parameterized_class
from testsuite.acquisition_functions import saf_mu
from testsuite.surrogates import GP, RF
import numpy as np


@parameterized_class([
    {"name": "GP", "surrogate": GP, "args": [],
     "kwargs": {"scaled": True}},
    {"name": "RF", "surrogate": RF, "args": [],
     "kwargs": {"extra_trees": True}},
])
class TestSAFMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surr = cls.surrogate(*cls.args, **cls.kwargs)
        cls.x = np.random.uniform(0, 1, size=[10, 5])
        cls.y = np.random.uniform(0, 5, size=[10, 2])

        cls.surr.update(cls.x, cls.y)

    @parameterized.expand([
        (saf_mu, )
    ])
    def test_returns(self, acq):
        # test single evaluations of acquisition function
        x_put = np.random.uniform(0,1, size=self.x[0:1].shape)
        ans = acq(x_put, self.surr, self.y)
        self.assertIsInstance(ans, np.ndarray)
        self.assertEqual(ans.shape[0], x_put.shape[0])

        # test multiple evaluations of acquisition function
        x_put = np.random.uniform(0,1, size=self.x[0:5].shape)
        ans = acq(x_put, self.surr, self.y)
        self.assertIsInstance(ans, np.ndarray)
        self.assertEqual(ans.shape[0], x_put.shape[0])

if __name__ == '__main__':
    unittest.main()
