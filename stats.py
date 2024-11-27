import pandas as pd
import numpy as np
import scipy.stats
import scikit_posthocs
import diptest

from itertools import combinations


def posthoc_dunn(df: pd.DataFrame, val_col: str, group_col: str) -> pd.DataFrame:
    """
    Perform Dunn's post-hoc test with Bonferroni correction for multiple comparisons.
    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    :param val_col: The column name containing the values to be compared between groups.
    :type val_col: str
    :param group_col: The column name containing the group labels.
    :type group_col: str
    :return: A DataFrame containing pairwise p-values after Bonferroni correction.
    :rtype: pd.DataFrame

    :example:
        >>> example_df = pd.DataFrame({
        ...     'value': [1.2, 2.3, 2.5, 3.1],
        ...     'group': ['A', 'B', 'A', 'B']
        ... })
        >>> posthoc_dunn(example_df, val_col='value', group_col='group')
    """
    posthoc = scikit_posthocs.posthoc_dunn(
        df, val_col=val_col, group_col=group_col, p_adjust="bonferroni"
    )
    posthoc_df = posthoc.reset_index().melt(
        id_vars="index", var_name="Comparison", value_name="p-value"
    )
    posthoc_df.columns = ["Group1", "Group2", "p-value"]
    posthoc_df = posthoc_df.pivot(index="Group1", columns="Group2", values="p-value")
    return posthoc_df


# produced by ChatGPT
def pairwise_diffs(
    df: pd.DataFrame, groupby_cols: list[str], value_col: str
) -> pd.DataFrame:
    """
    Calculate pairwise differences in mean values between groups.

    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    :param groupby_cols: The columns to group by in order to calculate mean values.
    :type groupby_cols: list[str]
    :param value_col: The column name containing the values for which pairwise differences will be calculated.
    :type value_col: str
    :return: A DataFrame containing pairwise mean differences between groups.
    :rtype: pd.DataFrame

    :example:
        >>> example_df = pd.DataFrame({
        ...     'group': ['A', 'B', 'A', 'B'],
        ...     'value': [1.2, 2.3, 2.5, 3.1]
        ... })
        >>> pairwise_diffs(example_df, groupby_cols=['group'], value_col='value')
    """
    # calculate the mean toxicity for each combination of annotator_prompt and conv_variant
    mean_values = df.groupby(groupby_cols)[value_col].mean().unstack()

    # create an NxN DataFrame to store the pairwise differences
    annotator_prompts = mean_values.columns
    diff_matrix = pd.DataFrame(
        index=annotator_prompts, columns=annotator_prompts, dtype=float
    )

    # calculate pairwise differences between annotator prompts
    for annotator_prompt_1, annotator_prompt_2 in combinations(annotator_prompts, 2):
        differences = mean_values[annotator_prompt_1] - mean_values[annotator_prompt_2]
        average_diff = differences.mean()

        # Populate the difference matrix symmetrically
        diff_matrix.loc[annotator_prompt_1, annotator_prompt_2] = average_diff
        diff_matrix.loc[annotator_prompt_2, annotator_prompt_1] = -average_diff

    # no difference with itself
    np.fill_diagonal(diff_matrix.values, 0)

    return diff_matrix
