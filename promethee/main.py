import os
import itertools
from pathlib import Path

import click
import numpy as np
import pandas as pd

from promethee.utils import (
    load_dataset,
    load_preference_information,
    display_ranking,
    Relation,
)

ScalarOrNumpy = float | np.ndarray

# TODO
def calculate_marginal_preference_index[T: ScalarOrNumpy, U: ScalarOrNumpy](
    diff: T, q: U, p: U
) -> T:
    """
    Function that calculates the marginal preference index for the given pair of alternatives, according to the formula presented during the classes

    :param diff: difference between compared alternatives either as a float for single parser and alternative pairs, or as numpy array for multiple alternative/parser
    :param q: indifference threshold either as a float if you prefer to calculate for a single parser or as numpy array for multiple parser
    :param p: preference threshold either as a float if you prefer to calculate for a single parser or as numpy array for multiple parser
    :return: marginal preference index either as a float for single parser and alternative pairs, or as numpy array for multiple alternative/parser
    """
    return np.minimum(np.maximum((diff - q) / (p - q), 0), 1)  # pyright: ignore[reportReturnType]


# TODO
def calculate_marginal_preference_matrix(
    dataset: pd.DataFrame, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates the marginal preference matrix all alternatives pairs and criterion available in dataset

    :param dataset: difference between compared alternatives
    :param preference_information: preference information
    :return: 3D numpy array with marginal preference matrix on every parser, Consecutive indices [i, j, k] describe first alternative, second alternative, parser
    """
    dataset = dataset.copy()
    preference_information = preference_information.copy()
    cost_criteria = preference_information[preference_information["type"] == "cost"].index
    dataset[cost_criteria] = -dataset[cost_criteria]
    preference_information[["p", "q"]].loc[cost_criteria] = -preference_information[["p", "q"]].loc[cost_criteria]

    marginal_preferences = np.zeros((
        preference_information.shape[0],
        dataset.shape[0],
        dataset.shape[0],
    ))
    for (i, alt1), (j, alt2) in itertools.permutations(enumerate(dataset.index), 2):
        diff = dataset.loc[alt1] - dataset.loc[alt2]
        p = preference_information["p"]
        q = preference_information["q"]
        marginal_preferences[:, i, j] = calculate_marginal_preference_index(
            diff=diff.to_numpy(),
            q=q.to_numpy(),
            p=p.to_numpy(),
        )
    return marginal_preferences


# TODO
def calculate_comprehensive_preference_index(
    marginal_preference_matrix: np.ndarray, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates comprehensive preference index for the given dataset

    :param marginal_preference_matrix: 3D numpy array with marginal preference matrix on every parser, Consecutive indices [i, j, k] describe first alternative, second alternative, parser
    :param preference_information: Padnas preference information dataframe
    :return: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    """
    # (n_weights) -> (n_weights, 1, 1)
    weights = np.expand_dims(preference_information["k"].to_numpy(), (1, 2))
    weights = weights / weights.sum()
    return (marginal_preference_matrix * weights).sum(0)


# TODO
def calculate_positive_flow(
    comprehensive_preference_matrix: np.ndarray, index: pd.Index
) -> pd.Series:
    """
    Function that calculates the positive flow value for the given preference matrix and corresponding index

    :param comprehensive_preference_matrix: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    :param index: index representing the alternative in the corresponding position in preference matrix
    :return: series representing positive flow values for the given preference matrix
    """
    return pd.Series(comprehensive_preference_matrix.sum(1), index=index)


# TODO
def calculate_negative_flow(
    comprehensive_preference_matrix: np.ndarray, index: pd.Index
) -> pd.Series:
    """
    Function that calculates the negative flow value for the given preference matrix and corresponding index

    :param comprehensive_preference_matrix: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    :param index: index representing the alternative in the corresponding position in preference matrix
    :return: series representing negative flow values for the given preference matrix
    """
    return pd.Series(comprehensive_preference_matrix.sum(0), index=index)


# TODO
def calculate_net_flow(positive_flow: pd.Series, negative_flow: pd.Series) -> pd.Series:
    """
    Function that calculates the net flow value for the given positive and negative flow

    :param positive_flow: series representing positive flow values for the given preference matrix
    :param negative_flow: series representing negative flow values for the given preference matrix
    :return: series representing net flow values for the given preference matrix
    """
    return positive_flow - negative_flow


# TODO
def create_partial_ranking(
    positive_flow: pd.Series, negative_flow: pd.Series
) -> set[tuple[str, str, Relation]]:
    """
    Function that aggregates positive and negative flow to a partial ranking (from Promethee I)

    :param positive_flow: series representing positive flow values for the given preference matrix
    :param negative_flow: series representing negative flow values for the given preference matrix
    :return: list of tuples when entries in a tuple represent first alternative, second alternative and the relation between them respectively
    """
    result = set()
    alternatives = positive_flow.index
    for i, j in itertools.permutations(alternatives, 2):
        better_strength = positive_flow[i] > positive_flow[j]
        better_weakness = negative_flow[i] < negative_flow[j]

        if better_strength and better_weakness:
            relation = Relation.PREFERRED
        elif (
            positive_flow[i] == positive_flow[j] and
            negative_flow[i] == negative_flow[j]
        ):
            relation = Relation.INDIFFERENT
        elif (
            better_strength and not better_weakness or
            not better_strength and better_weakness
        ):
            relation = Relation.INCOMPARABLE
        else:
            continue

        result.add((str(i), str(j), relation))
    return result


# TODO
def create_complete_ranking(net_flow: pd.Series) -> set[tuple[str, str, Relation]]:
    """
    Function that aggregates positive and negative flow to a complete ranking (from Promethee II)
    :param net_flow: series representing net flow values for the given preference matrix
    :return: dataframe with alternatives in both index and columns. Every entry in the dataframe from row i and column j represents relation between alternative i and alternative j:
    1 means that i is preferred over j, or they are indifferent
    0 otherwise
    """
    result = set()
    alternatives = net_flow.index
    for i, j in itertools.permutations(alternatives, 2):
        if net_flow[i] > net_flow[j]:
            relation = Relation.PREFERRED
        elif net_flow[i] == net_flow[j]:
            relation = Relation.INDIFFERENT
        else:
            continue

        result.add((str(i), str(j), relation))
    return result


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def promethee(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)

    dataset = load_dataset(dataset_path)
    preference_information = load_preference_information(dataset_path)

    marginal_preference_matrix = calculate_marginal_preference_matrix(
        dataset, preference_information
    )
    comprehensive_preference_matrix = calculate_comprehensive_preference_index(
        marginal_preference_matrix, preference_information
    )

    positive_flow = calculate_positive_flow(
        comprehensive_preference_matrix, dataset.index
    )
    negative_flow = calculate_negative_flow(
        comprehensive_preference_matrix, dataset.index
    )

    assert positive_flow.index.equals(negative_flow.index)

    partial_ranking = create_partial_ranking(positive_flow, negative_flow)
    display_ranking(partial_ranking, "Promethee I")

    net_flow = calculate_net_flow(positive_flow, negative_flow)
    complete_ranking = create_complete_ranking(net_flow)
    display_ranking(complete_ranking, "Promethee II")


if __name__ == "__main__":
    promethee()
