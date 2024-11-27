import numpy as np
import math

from .ndfu import ndfu


def aposteriori_unimodality(
    grouped_annotations: np.ndarray, tol: float = 10e-3
) -> bool:
    """
    Run a statistical test for the aposteriori unimodality for annotations divided by a certain feature.
    If global nDFU > 0 and the retuned pvalue is low, then we reject the hypothesis that the
    feature does not explain the observed polarization.

    This test calculates the nDFU of all the annotations, and then the nDFU of each factor of the selected feature.
    If global nDFU < nDFU_{factor}, then we reject the aposteriori unimodality hypothesis,
    and conclude that the feature explains the polarization.

    :param grouped_annotations: a 2D array, containing the annotations, grouped by each factor for the selected feature.
    For example, [[1,1,1,2], [2,3,2,4]] where the first list was created by male annotators, and the second with female ones.
    :type grouped_annotations: np.ndarray
    :return: 1-pvalue that all ndfus are zero
    :rtype: tuple[float, float]
    """
    # combine into flat array
    global_annotations = np.concatenate(grouped_annotations, axis=0)
    global_ndfu = ndfu(global_annotations)

    grouped_ndfus = np.array(
        [ndfu(annotation_group) for annotation_group in grouped_annotations]
    )

    results = [
        is_significantly_bigger(global_ndfu, grouped_ndfu, tol)
        for grouped_ndfu in grouped_ndfus
    ]
    # consider that polarization is explained through a feature, iff
    # at least one factor in the feature exhibits more polarization than the global
    return True in results


def is_significantly_bigger(a: float, b: float, tol: float) -> bool:
    return a > b and not math.isclose(a, b, abs_tol=tol)
