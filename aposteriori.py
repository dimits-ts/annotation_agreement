from typing import Hashable
import numpy as np
import scipy

from .ndfu import ndfu


def aposteriori_unimodality(
    annotations: list[np.ndarray], annotator_group: list[np.ndarray]
) -> dict[Hashable, float]:
    """
    Conducts a statistical test to evaluate whether the polarization of annotations 
    can be explained by a specific grouping feature. If the p-value is below the 
    significance level, it suggests that the grouping feature contributes to polarization.

    Args:
        annotations (list[np.ndarray]): A list where each element contains the annotations 
            for a single comment in a discussion.
        annotator_group (list[np.ndarray]): A list where each element contains the group 
            assignments of annotators corresponding to the annotations for each comment.

    Returns:
        dict[Hashable, float]: A dictionary mapping each unique group (factor) to the 
            p-value of the statistical test for that group.

    Raises:
        ValueError: If the number of comments in `annotations` and `annotator_group` 
            do not match, or if the lengths of annotations and group assignments 
            are inconsistent for any comment.
    """
    if len(annotations) != len(annotator_group):
        raise ValueError(
            "The number of comments in `annotations` and `annotator_group` must be the same."
        )
        
    if len(annotations) == 0:
        return {}

    for annotation, group in zip(annotations, annotator_group):
        if len(annotation) != len(group) or len(annotation) == 0 or len(group) == 0:
            raise ValueError(
                "Annotations and group assignments must have the same length for each comment."
            )
    
    for annotation, group in zip(annotations, annotator_group):
        if len(annotation) == 0 or len(group) == 0:
            raise ValueError(
                "Comments should have at least one annotation."
            )

    # Initialize statistics for each group level
    aposteriori_unit_statistics: dict[Hashable, list] = {
        key: [] for key in np.unique(np.concatenate(annotator_group))
    }

    # Calculate per-comment statistics for each group level
    for comment_annotations, comment_annotator_group in zip(annotations, annotator_group):
        for level in np.unique(annotator_group):
            aposteriori_stat = level_aposteriori_unit(
                comment_annotations, comment_annotator_group, level
            )
            aposteriori_unit_statistics[level].append(aposteriori_stat)

    # Aggregate statistics for the entire group
    aposteriori_final_statistics: dict[Hashable, float] = {}
    for level, stats in aposteriori_unit_statistics.items():
        aposteriori_final_statistics[level] = level_aposteriori_whole(stats)

    return aposteriori_final_statistics


def level_aposteriori_whole(level_aposteriori_statistics: list[float]) -> float:
    """
    Performs a Wilcoxon signed-rank test to determine the significance of differences 
    in aposteriori statistics for a specific group.

    Args:
        level_aposteriori_statistics (list[float]): A list of aposteriori statistics for a group.

    Returns:
        float: The p-value of the Wilcoxon test. If no difference is detected, returns 1.
    """
    x = level_aposteriori_statistics
    y = np.zeros_like(level_aposteriori_statistics)
    if np.sum(x - y) == 0:
        return 1.0
    else:
        return scipy.stats.wilcoxon(x, y=y, alternative="greater").pvalue


def level_aposteriori_unit(
    annotations: np.ndarray, annotator_group: np.ndarray, level: Hashable
) -> float:
    """
    Computes the aposteriori statistic for a specific group within a single comment.

    Args:
        annotations (np.ndarray): The annotations for a single comment.
        annotator_group (np.ndarray): The group assignments for annotators of the same comment.
        level (Hashable): The group level to compute the statistic for.

    Returns:
        float: The aposteriori statistic for the specified group level.
    """
    level_annotations = annotations[annotator_group == level]
    aposteriori_score = aposteriori_unit(annotations, level_annotations)
    return aposteriori_score


def aposteriori_unit(
    global_annotations: np.ndarray, level_annotations: np.ndarray
) -> float:
    """
    Computes the difference in normalized distance from unimodality (nDFU) between 
    global annotations and annotations from a specific group.

    Args:
        global_annotations (np.ndarray): The full set of annotations for a comment.
        level_annotations (np.ndarray): The subset of annotations corresponding to a specific group.

    Returns:
        float: The difference in nDFU values, which indicates the contribution of the 
        group to polarization.
    """
    global_ndfu = ndfu(global_annotations)
    level_ndfu = ndfu(level_annotations)
    return global_ndfu - level_ndfu
