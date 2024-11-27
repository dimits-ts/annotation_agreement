import unittest
import numpy as np

from .. import ndfu


class TestNDFUFunctions(unittest.TestCase):

    def test_to_hist_basic(self):
        """Test _to_hist with a simple array"""
        scores = [1, 2, 2, 3, 3, 3, 4]
        expected_hist = [1/7, 2/7, 3/7, 1/7]
        result = ndfu._to_hist(scores, bins_num=4, normed=True)
        np.testing.assert_almost_equal(result, expected_hist, decimal=6)

    def test_to_hist_non_normalized(self):
        """Test _to_hist with normalization turned off"""
        scores = [1, 2, 2, 3, 3, 3, 4]
        expected_hist = [1, 2, 3, 1]
        result = ndfu._to_hist(scores, bins_num=4, normed=False)
        np.testing.assert_array_equal(result, expected_hist)

    def test_ndfu_precomputed_histogram(self):
        """Test ndfu with a precomputed histogram"""
        hist = np.array([0.1, 0.3, 0.5, 0.1])
        result = ndfu.ndfu(hist, histogram_input=True, normalised=True)
        self.assertAlmostEqual(result, 0.4 / 0.5, places=6)

    def test_ndfu_raw_data(self):
        """Test ndfu with raw data input"""
        raw_data = [1, 1, 2, 3, 3, 3, 4, 4, 5]
        result = ndfu.ndfu(raw_data, histogram_input=False, normalised=True)
        self.assertAlmostEqual(result, 0.222222, places=6)  # Precomputed

    def test_ndfu_unnormalized(self):
        """Test ndfu with normalization turned off"""
        hist = np.array([0.1, 0.3, 0.5, 0.1])
        result = ndfu.ndfu(hist, histogram_input=True, normalised=False)
        self.assertEqual(result, 0.4)

    def test_ndfu_empty_data(self):
        """Test ndfu with empty input"""
        with self.assertRaises(ValueError):
            ndfu.ndfu([], histogram_input=True)

    def test_to_hist_empty_data(self):
        """Test _to_hist with empty input"""
        with self.assertRaises(ValueError):
            ndfu._to_hist([], bins_num=3)


if __name__ == '__main__':
    unittest.main()
