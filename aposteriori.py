import numpy as np
import scipy

from ndfu import ndfu


def aposteriori_unimodality(grouped_annotations: list[np.array]) -> tuple[float, float]:
    """Run a statistical test for the aposteriori unimodality for annotations divided by a certain feature.
    If global nDFU > 0 and the retuned pvalue is low, then we reject the hypothesis that the
    feature does not explain the observed polarization.

    This test calculates the nDFU of all the annotations, and then the nDFU of each factor of the selected feature.
    If global nDFU > 0 but for all factors, nDFU_{factor} == 0, then we reject the aposteriori unimodality hypothesis.
    Instead of returning the individual nDFUs, this function runs a Wilcoxon test to determine if all nDFUs are 0.
    We use a non-parametric test because annotations rarely follow the normal distribution, and are typically few in number.

    :param grouped_annotations: the annotations, grouped by each factor for the selected feature
    :type grouped_annotations: list[np.array]
    :return: the global nDFU, and the 1-pvalue that all ndfus are zero
    :rtype: tuple[float, float]
    """
    # combine into flat array
    global_annotations = np.concatenate(grouped_annotations, axis=0)
    global_ndfu = ndfu(global_annotations)

    grouped_unimodality_pvalue = _groups_are_unimodal(grouped_annotations)
    return global_ndfu, 1 - grouped_unimodality_pvalue


def _groups_are_unimodal(grouped_annotations: list[np.array]) -> float:
    """Test whether the nDFU of each factor of the feature are zero, using a Wilcoxon test.

    :param grouped_annotations: the annotations, grouped by each factor for the selected feature
    :type grouped_annotations: list[np.array]
    :return: the pvalue that all ndfus are zero
    :rtype: float
    """
    grouped_ndfu = np.array(
        [ndfu(annotation_group) for annotation_group in grouped_annotations]
    )
    # Use wilcoxon because we can not assume normality or random sampling
    # (realistically, annotations will be few)
    _, pvalue = scipy.stats.wilcoxon(
        grouped_ndfu, np.zeros_like(grouped_ndfu), alternative="greater"
    )
    # H_0: all ndfus are 0 => feature explains polarization
    # H_a: at least 1 ndfu > 0 => feature does not explain polarization
    # use 1-pvalue to reverse the above. Now, if p is low, we accept that feature explains polarization
    return pvalue
