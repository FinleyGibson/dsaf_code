import unittest
from parameterized import parameterized

import numpy as np

from testsuite.utilities import dominates
class TestDominates(unittest.TestCase):
    # basic dominated
    case_00 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': True,
               'maximize': False,
               'strict': True
               }

    # basic non-dominated
    case_01 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': False,
               'maximize': False,
               'strict': True
               }
    # edge dominated, strict
    case_02 = {'a': np.array([[1., 3.],
                              [3., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': False,
               'maximize': False,
               'strict': True
               }
    # edge dominated, strict
    case_03 = {'a': np.array([[1., 3.],
                              [3., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': True,
               'maximize': False,
               'strict': False
               }
    # edge non-dominated, beyond scope
    case_04 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[0., 5.]]),
               'dominated': False,
               'maximize': False,
               'strict': True
               }

    # edge dominated, beyond scope
    case_05 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[2., 5.]]),
               'dominated': True,
               'maximize': False,
               'strict': True
               }

    # edge dominated, beyond scope, strict
    case_06 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[1., 5.]]),
               'dominated': False,
               'maximize': False,
               'strict': True
               }

    # edge dominated, beyond scope, non-strict
    case_07 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[1., 5.]]),
               'dominated': True,
               'maximize': False,
               'strict': False
               }

    # inverted
    # basic dominated
    case_10 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }

    # basic non-dominated
    case_11 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }
    # edge dominated, strict
    case_12 = {'a': np.array([[1., 3.],
                              [3., 1.]
                              ]),
               'b': np.array([[1., 1.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }
    # edge dominated, strict
    case_13 = {'a': np.array([[1., 3.],
                              [3., 1.]
                              ]),
               'b': np.array([[1., 1.]]),
               'dominated': True,
               'maximize': True,
               'strict': False
               }
    # edge non-dominated, beyond scope
    case_14 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[0., 5.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }

    # edge dominated, beyond scope
    case_15 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[2., 5.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }

    # edge dominated, beyond scope, strict
    case_16 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[0., 4.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }

    # edge dominated, beyond scope, non-strict
    case_17 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[0., 4.]]),
               'dominated': True,
               'maximize': True,
               'strict': False
               }
    case_20 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 3.], [3., 0.]]),
               'dominated': [True, False],
               'maximize': False,
               'strict': True
               }

    case_21 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 3.], [3., 2.]]),
               'dominated': [True, True],
               'maximize': False,
               'strict': True
               }

    case_22 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 0.], [1., 1.]]),
               'dominated': [False, False],
               'maximize': False,
               'strict': True
               }

    cases = [
        case_00,
        case_01,
        case_02,
        case_03,
        case_04,
        case_05,
        case_06,
        case_07,
        case_10,
        case_11,
        case_12,
        case_13,
        case_14,
        case_15,
        case_16,
        case_17,
        case_20,
        case_21,
        case_22
    ]

    # cases = [case_13]

    def test_multiple_element_shapes(self):
        n = 10; m = 2
        """
        to check for dominance of b by ANY of a do:
            ans = dominates(a, b)
            - a.shape = (n, m)
            - b.shape = (1, m)
            - ans: bool
        """
        a = np.random.uniform(0, 1, size=(n, m))
        b = np.random.uniform(0, 1, size=(1, m))
        ans = dominates(a, b)
        self.assertIsInstance(ans, bool)

        """
        to check for which of b by are dominated by ANY of a do:
            ans = dominates(a, b)
            - a.shape = (n, m)
            - b.shape = (n, m)
            - ans: [bool, bool, ..., bool]
        """
        a = np.random.uniform(0, 1, size=(n, m))
        b = np.random.uniform(0, 1, size=(n, m))
        ans = dominates(a, b)
        self.assertIsInstance(ans, list)
        self.assertIsInstance(ans[0], bool)

        """
        to check which of a dominate b do:
        ans = [dominates(ai, b) for ai in a]
        - a.shape = (n, m)
        - b.shape = (1, m)
        - ans: [bool, bool, ..., bool]
        """
        a = np.random.uniform(0, 1, size=(n, m))
        b = np.random.uniform(0, 1, size=(1, m))
        ans = [dominates(ai, b) for ai in a]
        self.assertIsInstance(ans, list)
        self.assertIsInstance(ans[0], bool)

        """
        to check which of b are dominated by any of a do:
        ans = [dominates(a, bi) for bi in b]
        - a.shape = (n, m)
        - b.shape = (n, m)
        - ans: [bool, bool, ..., bool]
        """
        a = np.random.uniform(0, 1, size=(n, m))
        b = np.random.uniform(0, 1, size=(n, m))
        ans = [dominates(a, bi) for bi in b]
        self.assertIsInstance(ans, list)
        self.assertIsInstance(ans[0], bool)


    @parameterized.expand([[case] for case in cases])
    def test_dominated_case_status(self, case):
        out = dominates(a=case['a'],
                        b=case['b'],
                        maximize=case['maximize'],
                        strict=case['strict'])
        if isinstance(out, bool):
            self.assertEqual(case['dominated'], out)
        elif isinstance(out, np.ndarray):
            tmp = np.array(case['dominated'])
            self.assertTrue(np.array_equal(np.ndarray(case['dominated']), out))

    def test_timing(self):
        import numpy as np
        import time

        a = np.random.randn(10, 4)
        b = np.random.randn(100, 4)

        tic = time.time()
        ans0 = dominates(a, b)
        print(time.time() - tic)
        t1 = time.time()-tic

        tic = time.time()
        ans1 = dominates(b, a)
        t2 = time.time()-tic

#
from testsuite.utilities import Pareto_split
class TestParetoSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # create set of points where alternating points dominate
        yi = np.linspace(0, 1, 50)
        y = np.vstack((yi, 1-yi)).T
        y[::2] = y[::2]+0.05

        cls.y = y
        cls.p, cls.d = Pareto_split(cls.y)
        cls.p_ind, cls.d_ind = Pareto_split(cls.y, return_indices=True)

        cls.pmax, cls.dmax = Pareto_split(cls.y, maximize=True)
        cls.pmax_ind, cls.dmax_ind = Pareto_split(cls.y,
                                                  return_indices=True,
                                                  maximize=True)

    def test_basic_pareto_split(self) -> None:
        self.assertEqual(self.p.shape[0], self.d.shape[0])
        np.testing.assert_array_equal(self.d, self.y[::2])
        np.testing.assert_array_equal(self.p, self.y[1::2])

    def test_indices_pareto_split(self) -> None:
        self.assertEqual(self.p_ind.shape[0], self.d_ind.shape[0])
        np.testing.assert_array_equal(self.y[self.d_ind], self.y[::2])
        np.testing.assert_array_equal(self.y[self.p_ind], self.y[1::2])

    def test_maximise(self) -> None:
        self.assertEqual(self.pmax.shape[0], self.dmax.shape[0])
        np.testing.assert_array_equal(self.dmax, self.y[1::2])
        np.testing.assert_array_equal(self.pmax, self.y[::2])

        np.testing.assert_array_equal(self.y[self.dmax_ind], self.y[1::2])
        np.testing.assert_array_equal(self.y[self.pmax_ind], self.y[::2])
