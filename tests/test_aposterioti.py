import unittest
import numpy as np
from ..aposteriori import (
    aposteriori_unimodality,
    level_aposteriori_unit,
    level_aposteriori_whole,
    aposteriori_unit,
)


class TestAposterioriUnimodality(unittest.TestCase):

    def setUp(self):
        # Example annotations and groups for testing
        self.annotations = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
        ]
        self.annotator_group = [
            np.array(["A", "B", "A", "B"]),
            np.array(["A", "A", "B", "B"]),
        ]

    def test_aposteriori_unimodality_valid(self):
        """Test valid input for aposteriori_unimodality."""
        result = aposteriori_unimodality(self.annotations, self.annotator_group)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"A", "B"})
        for pvalue in result.values():
            self.assertGreaterEqual(pvalue, 0)
            self.assertLessEqual(pvalue, 1)

    def test_aposteriori_unimodality_mismatched_lengths(self):
        """Test input where annotations and annotator_group lengths mismatch."""
        with self.assertRaises(ValueError):
            aposteriori_unimodality(self.annotations, self.annotator_group[:-1])

    def test_aposteriori_unimodality_inconsistent_lengths_within(self):
        """Test input where annotations and annotator_group lengths mismatch within a comment."""
        bad_annotator_group = [
            np.array(["A", "B"]),  # Mismatch length
            np.array(["A", "A", "B", "B"]),
        ]
        with self.assertRaises(ValueError):
            aposteriori_unimodality(self.annotations, bad_annotator_group)

    def test_level_aposteriori_whole_zero_difference(self):
        """Test level_aposteriori_whole with zero difference statistics."""
        stats = [0.0, 0.0, 0.0]
        pvalue = level_aposteriori_whole(stats)
        self.assertEqual(pvalue, 1.0)

    def test_level_aposteriori_whole_nonzero_difference(self):
        """Test level_aposteriori_whole with nonzero differences."""
        stats = [0.1, 0.2, 0.3]
        pvalue = level_aposteriori_whole(stats)
        self.assertGreater(pvalue, 0)
        self.assertLessEqual(pvalue, 1)

    def test_level_aposteriori_unit_valid(self):
        """Test level_aposteriori_unit with valid input."""
        annotations = np.array([0.1, 0.2, 0.3, 0.4])
        annotator_group = np.array(["A", "B", "A", "B"])
        level = "A"
        score = level_aposteriori_unit(annotations, annotator_group, level)
        self.assertIsInstance(score, float)

    def test_aposteriori_unit_difference(self):
        """Test aposteriori_unit with valid input to ensure it computes nDFU differences."""
        global_annotations = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        level_annotations = np.array([0.1, 0.2, 0.3])
        result = aposteriori_unit(global_annotations, level_annotations)
        self.assertIsInstance(result, float)

    def test_aposteriori_unit_no_difference(self):
        """Test aposteriori_unit when global and level annotations are identical."""
        annotations = np.array([0.1, 0.2, 0.3, 0.4])
        result = aposteriori_unit(annotations, annotations)
        self.assertEqual(result, 0.0)

    def test_edge_case_empty_annotations(self):
        """Test edge case with empty annotations."""
        annotations = []
        annotator_group = []
        result = aposteriori_unimodality(annotations, annotator_group)
        self.assertEqual(result, {})

    def test_edge_case_single_group(self):
        """Test edge case with a single group for all annotations."""
        annotations = [np.array([0.1, 0.2, 0.3, 0.4])]
        annotator_group = [np.array(["A", "A", "A", "A"])]
        result = aposteriori_unimodality(annotations, annotator_group)
        self.assertEqual(result, {"A": 1.0})


if __name__ == "__main__":
    unittest.main()
