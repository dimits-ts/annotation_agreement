import numpy as np
import math

from .ndfu import ndfu


def aposteriori_unimodality(comments_annotations: np.ndarray, tol: float = 0) -> float:
    """
    Run a statistical test for the aposteriori unimodality for annotations split by a certain feature.
    If pvalue < statistical_level, then the polarization of the annotations can be explained by the selected feature.

    :param comments_annotations: a 3D array of the shape [comment_1, comment_2, ...],
     where comment=[[annotator_group_1], [annotator_group_2], ...], 
     where annotator_group=[annotation_1, annotation_2, ...], 
     where annotation is a factor (integer) 
    For example, [[[1,1,1,2], [2,3,2,4]]] represents a single comment
    where the annotations in the first list were created by male annotators, and the second with female ones.
    :type comments_annotations: np.ndarray
    :param tol: TODO
    :type tol: float
    :return: the pvalue of the test
    :rtype: tuple[float, float]
    """
    # combine into flat array
    if len(comments_annotations) == 0:
        raise ValueError("Need at least two groups for grouped_annotations.")

    aposteriori_units = np.array(
        [
            aposteriori_unit(comment_annotations)
            for comment_annotations in comments_annotations
        ]
    )

    aposteriori_passed = np.array(
        [is_significantly_bigger(x, 0, tol) for x in aposteriori_units]
    )

    return 1 - (aposteriori_passed.sum() / len(aposteriori_passed))


def aposteriori_unit(grouped_annotations: np.ndarray) -> float:
    global_annotations = np.concatenate(grouped_annotations, axis=0)
    global_ndfu = ndfu(global_annotations)

    ndfus = np.array([ndfu(group) for group in grouped_annotations])

    diffs = global_ndfu - ndfus
    return diffs.max()


def is_significantly_bigger(a: float, b: float, tol: float) -> bool:
    return a > b and not math.isclose(a, b, abs_tol=tol)
