import unittest
import numpy as np
from run_matched_retrieval import measures, rank, cluster_intervals, is_dev


class ControlsTest(unittest.TestCase):
    def test_metrics_cutoffs(self):
        m = measures([1, 11, 20, 21])
        self.assertEqual(m[0].tolist(), [1]*6)
        self.assertAlmostEqual(m[1,0], 1/11)
        self.assertEqual(m[1,1], 0)
        self.assertEqual(m[1,2], 0)
        self.assertEqual(m[2,5], 1)
        self.assertEqual(m[3,5], 0)

    def test_tie_document_order(self):
        self.assertEqual(rank([0, 1, 1, 0]), [1,2,0,3])

    def test_cluster_ci_identical_and_reproducible(self):
        values = np.ones((5,6))
        first = cluster_intervals(values, ['a','a','b','c','c'])
        np.testing.assert_array_equal(first, np.ones((2,6)))
        np.testing.assert_array_equal(first, cluster_intervals(values, ['a','a','b','c','c']))

    def test_split_source_stability(self):
        self.assertEqual(is_dev('same.json'), is_dev('same.json'))


if __name__ == '__main__':
    unittest.main()
