import unittest
import numpy as np
from ..aposteriori import aposteriori_unimodality, is_significantly_bigger


class TestAposterioriUnimodality(unittest.TestCase):
    def test_aposteriori_unimodality_basic(self):
        """Test with simple grouped annotations where polarization is likely explained."""
        grouped_annotations = np.array([
            [1, 1, 1, 2],
            [2, 3, 2, 4]
        ])
        result = aposteriori_unimodality(grouped_annotations)
        self.assertIsInstance(result, bool, "The result should be a boolean.")
        self.assertTrue(result, "Feature should explain polarization")

    def test_aposteriori_unimodality_no_explanation(self):
        """Test when no group has significantly higher nDFU than the global nDFU."""
        grouped_annotations = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        result = aposteriori_unimodality(grouped_annotations)
        self.assertFalse(result, "Should return False when no group explains the polarization.")
    
    def test_aposteriori_unimodality_one_group_significant(self):
        """Test when one group has significantly higher nDFU than the global."""
        grouped_annotations = np.array([
            [1, 1, 1, 2],
            [2, 4, 2, 2]
        ])
        result = aposteriori_unimodality(grouped_annotations)
        self.assertTrue(result, "Should return True when at least one group explains the polarization.")
    
    def test_aposteriori_unimodality_tolerance_handling(self):
        """Test the tolerance parameter affects the outcome."""
        grouped_annotations = np.array([
            [1, 1, 1, 1],
            [1.1, 1.1, 1.1, 1.1]
        ])
        # Low tolerance: Differences are not significant
        result_low_tol = aposteriori_unimodality(grouped_annotations, tol=2)
        self.assertFalse(result_low_tol, "Low tolerance should result in False.")

        # High tolerance: Differences become significant
        result_high_tol = aposteriori_unimodality(grouped_annotations, tol=10e-10)
        self.assertTrue(result_high_tol, "High tolerance should result in True.")


class TestIsSignificantlyBigger(unittest.TestCase):
    def test_is_significantly_bigger_basic(self):
        """Test basic comparison logic."""
        self.assertTrue(is_significantly_bigger(0.6, 0.5, tol=0.01), "Should return True when `a` > `b` significantly.")
        self.assertFalse(is_significantly_bigger(0.5, 0.6, tol=0.01), "Should return False when `a` < `b`.")
        self.assertFalse(is_significantly_bigger(0.5, 0.5, tol=0.01), "Should return False when `a` == `b` within tolerance.")

    def test_is_significantly_bigger_tolerance(self):
        """Test the effect of tolerance on comparison."""
        self.assertTrue(is_significantly_bigger(0.6, 0.5, tol=0.05), "Should return True for a difference of 0.1 with tol=0.05.")
        self.assertFalse(is_significantly_bigger(0.55, 0.5, tol=0.1), "Should return False when difference is within tolerance.")

    def test_is_significantly_bigger_edge_cases(self):
        """Test edge cases with extreme or equal values."""
        self.assertFalse(is_significantly_bigger(1e-10, 1e-10, tol=1e-12), "Should return False for nearly equal values.")
        self.assertTrue(is_significantly_bigger(1.0, 0.0, tol=0.1), "Should return True for large difference.")


if __name__ == "__main__":
    unittest.main()
