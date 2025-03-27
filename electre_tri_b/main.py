import itertools
from pathlib import Path

import click
import numpy as np
import pandas as pd

from electre_tri_b.utils import (
    load_dataset,
    load_boundary_profiles,
    load_indifference_thresholds,
    load_preference_thresholds,
    load_veto_thresholds,
    load_criterion_types,
    load_credibility_threshold,
)
from promethee.main import ScalarOrNumpy

# TODO
def calculate_marginal_concordance_index[
    T: ScalarOrNumpy,
    U: ScalarOrNumpy,
](diff: T, q: U, p: U) -> T:
    """
    Function that calculates the marginal concordance index for the given pair of alternatives, according to the formula presented during classes.

    :param diff: difference between compared alternatives either as a float for single criterion and alternative pairs, or as numpy array for multiple alternatives
    :param q: indifference threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :param p: preference threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :return: marginal concordance index either as a float for single criterion and alternative pairs, or as numpy array for multiple criterion
    """
    return np.minimum(np.maximum((p + diff) / (p - q), 0), 1)  # pyright: ignore[reportReturnType]


# TODO
def calculate_marginal_concordance_matrix(
    dataset: pd.DataFrame,
    boundary_profiles: pd.DataFrame,
    indifference_thresholds: pd.DataFrame,
    preference_thresholds: pd.DataFrame,
    criterion_types: pd.DataFrame,
) -> np.ndarray:
    """
    Function that calculates the marginal concordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param boundary_profiles: pandas dataframe with boundary profiles
    :param indifference_thresholds: pandas dataframe representing indifference thresholds for all boundary profiles and criterion
    :param preference_thresholds: pandas dataframe representing preference thresholds for all boundary profiles and criterion
    :param criterion_types: pandas dataframe with a column 'type' representing the type of criterion (either gain or cost)
    :return: 4D numpy array with marginal concordance matrix with shape [2, number of alternatives, number of boundary profiles, number of criterion], where element with index [0, i, j, k] describe marginal concordance index between alternative i and boundary profile j on criterion k, while element with index [1, i, j, k] describe marginal concordance index between boundary profile j and  alternative i on criterion k
    """
    dataset = dataset.copy()
    cost_criteria = criterion_types[criterion_types["type"] == "cost"].index
    dataset[cost_criteria] = -dataset[cost_criteria]
    boundary_profiles[cost_criteria] = -boundary_profiles[cost_criteria]

    marginal_concordance = np.zeros((
        2,                              # alternative-boundary and boundary-alternative
        dataset.shape[0],               # n_alt
        boundary_profiles.shape[0],     # n_bounds
        boundary_profiles.shape[1],     # n_crits
    ))
    for (i, alt), (j, bound) in itertools.product(
        enumerate(dataset.index),
        enumerate(boundary_profiles.index),
    ):
        alt_values = dataset.loc[alt]
        bound_values = boundary_profiles.loc[bound]
        p = preference_thresholds.loc[bound]
        q = indifference_thresholds.loc[bound]

        marginal_concordance[0, i, j, :] = calculate_marginal_concordance_index(
            diff=(alt_values - bound_values).to_numpy(),
            q=q.to_numpy(),
            p=p.to_numpy(),
        )
        marginal_concordance[1, i, j, :] = calculate_marginal_concordance_index(
            diff=(bound_values - alt_values).to_numpy(),
            q=q.to_numpy(),
            p=p.to_numpy(),
        )
    return marginal_concordance


# TODO
def calculate_comprehensive_concordance_matrix(
    marginal_concordance_matrix: np.ndarray, criterion_types: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates comprehensive concordance matrix for the given dataset

    :param marginal_concordance_matrix: 4D numpy array with marginal concordance matrix with shape [2, number of alternatives, number of boundary profiles, number of criterion], where element with index [0, i, j, k] describe marginal concordance index between alternative i and boundary profile j on criterion k, while element with index [1, i, j, k] describe marginal concordance index between boundary profile j and  alternative i on criterion k
    :param criterion_types: dataframe that contains "k" column with criterion weights
    :return: 3D numpy array with comprehensive concordance matrix with shape [2, number of alternatives, number of boundary profiles], where element with index [0, i, j] describe comprehensive concordance index between alternative i and boundary profile j, while element with index [1, i, j] describe comprehensive concordance index between boundary profile j and  alternative i
    """
    weights = criterion_types["k"].to_numpy()
    weights = weights / weights.sum()
    return marginal_concordance_matrix @ weights


# TODO
def calculate_marginal_discordance_index[
    T: ScalarOrNumpy,
    U: ScalarOrNumpy,
](diff: T, p: U, v: U) -> T:
    """
    Function that calculates the marginal concordance index for the given pair of alternatives, according to the formula presented during classes.

    :param diff: difference between compared alternatives either as a float for single criterion and alternative pairs, or as numpy array for multiple alternatives
    :param p: preference threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :param v: veto threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :return: marginal discordance index either as a float for single criterion and alternative pairs, or as numpy array for multiple criterion
    """
    return np.minimum(np.maximum((v + diff) / (v - p), 0), 1)  # pyright: ignore[reportReturnType]


# TODO
def calculate_marginal_discordance_matrix(
    dataset: pd.DataFrame,
    boundary_profiles: pd.DataFrame,
    preference_thresholds,
    veto_thresholds,
    criterion_types,
) -> np.ndarray:
    """
    Function that calculates the marginal discordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param boundary_profiles: pandas dataframe with boundary profiles
    :param preference_thresholds: pandas dataframe representing preference thresholds for all boundary profiles and criterion
    :param veto_thresholds: pandas dataframe representing veto thresholds for all boundary profiles and criterion
    :param criterion_types: pandas dataframe with a column 'type' representing the type of criterion (either gain or cost)
    :return: 4D numpy array with marginal discordance matrix with shape [2, number of alternatives, number of boundary profiles, number of criterion], where element with index [0, i, j, k] describe marginal discordance index between alternative i and boundary profile j on criterion k, while element with index [1, i, j, k] describe marginal discordance index between boundary profile j and  alternative i on criterion k
    """
    dataset = dataset.copy()
    cost_criteria = criterion_types[criterion_types["type"] == "cost"].index
    dataset[cost_criteria] = -dataset[cost_criteria]
    boundary_profiles[cost_criteria] = -boundary_profiles[cost_criteria]

    marginal_discordance = np.zeros((
        2,                              # alternative-boundary and boundary-alternative
        dataset.shape[0],               # n_alt
        boundary_profiles.shape[0],     # n_bounds
        boundary_profiles.shape[1],     # n_crits
    ))
    for (i, alt), (j, bound) in itertools.product(
        enumerate(dataset.index),
        enumerate(boundary_profiles.index)
    ):
        alt_values = dataset.loc[alt]
        bound_values = boundary_profiles.loc[bound]
        v = veto_thresholds.loc[bound]
        p = preference_thresholds.loc[bound]

        marginal_discordance[0, i, j, :] = calculate_marginal_discordance_index(
            diff=(alt_values - bound_values).to_numpy(),
            p=p.to_numpy(),
            v=v.to_numpy(),
        )
        marginal_discordance[1, i, j, :] = calculate_marginal_discordance_index(
            diff=(bound_values - alt_values).to_numpy(),
            p=p.to_numpy(),
            v=v.to_numpy(),
        )
    return marginal_discordance


# TODO
def calculate_credibility_index(
    comprehensive_concordance_matrix: np.ndarray,
    marginal_discordance_matrix: np.ndarray,
) -> np.ndarray:
    """
    Function that calculates the credibility index for the given comprehensive concordance matrix and marginal discordance matrix

    :param comprehensive_concordance_matrix: 3D numpy array with comprehensive concordance matrix. Every entry in the matrix [i, j] represents comprehensive concordance index between alternative i and alternative j
    :param marginal_discordance_matrix: 3D numpy array with marginal discordance matrix, Consecutive indices [i, j, k] describe first alternative, second alternative, criterion
    :return: 3D numpy array with credibility matrix with shape [2, number of alternatives, number of boundary profiles], where element with index [0, i, j] describe credibility index between alternative i and boundary profile j, while element with index [1, i, j] describe credibility index between boundary profile j and  alternative i
    """
    outranking_credibility = comprehensive_concordance_matrix.copy()
    marginal_discordance_matrix = np.nan_to_num(marginal_discordance_matrix)
    for d in [0, 1]:
        for i, j in itertools.product(*map(range, outranking_credibility.shape[1:])):
            factors = np.ones((marginal_discordance_matrix.shape[-1]))
            if (denom := (1 - comprehensive_concordance_matrix[d, i, j])) != 0:
                factors = (
                    (1 - marginal_discordance_matrix[d, i, j]) /
                    denom
                )
                factors = np.minimum(factors, 1)
            outranking_credibility[d, i, j] *= factors.prod()
    return outranking_credibility


# TODO
def calculate_outranking_relation_matrix(
    credibility_index: np.ndarray, credibility_threshold
) -> np.ndarray:
    """
    Function that calculates boolean matrix with information if outranking holds for a given pair

    :param credibility_index: 3D numpy array with credibility matrix with shape [2, number of alternatives, number of boundary profiles], where element with index [0, i, j] describe credibility index between alternative i and boundary profile j, while element with index [1, i, j] describe credibility index between boundary profile j and  alternative i
    :param credibility_threshold: float number
    :return: 3D numpy boolean matrix with information if outranking holds for a given pair
    """
    return credibility_index >= credibility_threshold

RELATION_TO_PREFERENCE = {
    # (alt_bound, bound_alt): preference
    (True, True): "I",
    (True, False): ">",
    (False, True): "<",
    (False, False): "?",
}

# TODO
def calculate_relation(
    outranking_relation_matrix: np.ndarray,
    alternatives: pd.Index,
    boundary_profiles_names: pd.Index,
) -> pd.DataFrame:
    """
    Function that determine relation between alternatives and boundary profiles

    :param outranking_relation_matrix: 3D numpy boolean matrix with information if outranking holds for a given pair
    :param alternatives: names of alternatives
    :param boundary_profiles_names: names of boundary profiles
    :return: pandas dataframe with relation between alternatives as rows and boundary profiles as columns. Use "<" or ">" for preference, "I" for indifference and "?" for incompatibility
    """
    relation = pd.DataFrame(index=alternatives, columns=boundary_profiles_names)
    for i in range(alternatives.shape[0]):
        for j in range(boundary_profiles_names.shape[0]):
            alt_relation = outranking_relation_matrix[:, i, j]
            relation.iloc[i, j] = RELATION_TO_PREFERENCE[tuple(alt_relation)]

    return relation


# TODO
def calculate_pessimistic_assigment(relation: pd.DataFrame) -> pd.DataFrame:
    """
    Function that calculates pessimistic assigment for given relation between alternatives and boundary profiles

    :param relation: pandas dataframe with relation between alternatives as rows and boundary profiles as columns. With "<" or ">" for preference, "I" for indifference and "?" for incompatibility
    :return: dataframe with pessimistic assigment
    """
    classes = []
    for _, row in relation.iterrows():
        boundaries = reversed(row.tolist())
        no_outranking = itertools.takewhile(lambda r: r in ["?", "<"], boundaries)
        cls = len(row) - sum(map(lambda _: 1, no_outranking))
        classes.append(cls)
    return pd.DataFrame({"pessimistic": classes}, index=relation.index)

# TODO
def calculate_optimistic_assigment(relation: pd.DataFrame) -> pd.DataFrame:
    """
    Function that calculates optimistic assigment for given relation between alternatives and boundary profiles

    :param relation: pandas dataframe with relation between alternatives as rows and boundary profiles as columns. With "<" or ">" for preference, "I" for indifference and "?" for incompatibility
    :return: dataframe with optimistic assigment
    """
    classes = []
    for _, row in relation.iterrows():
        boundaries = row.tolist()
        no_outranking = itertools.takewhile(lambda r: r in [">", "I", "?"], boundaries)
        cls = sum(map(lambda _: 1, no_outranking))
        classes.append(cls)
    return pd.DataFrame({"optimistic": classes}, index=relation.index)


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def electre_tri_b(dataset_path: Path) -> None:
    dataset_path = Path(dataset_path)

    dataset = load_dataset(dataset_path)
    boundary_profiles = load_boundary_profiles(dataset_path)
    criterion_types = load_criterion_types(dataset_path)
    indifference_thresholds = load_indifference_thresholds(dataset_path)
    preference_thresholds = load_preference_thresholds(dataset_path)
    veto_thresholds = load_veto_thresholds(dataset_path)
    credibility_threshold = load_credibility_threshold(dataset_path)

    marginal_concordance_matrix = calculate_marginal_concordance_matrix(
        dataset,
        boundary_profiles,
        indifference_thresholds,
        preference_thresholds,
        criterion_types,
    )
    comprehensive_concordance_matrix = calculate_comprehensive_concordance_matrix(
        marginal_concordance_matrix, criterion_types
    )

    marginal_discordance_matrix = calculate_marginal_discordance_matrix(
        dataset,
        boundary_profiles,
        preference_thresholds,
        veto_thresholds,
        criterion_types,
    )

    credibility_index = calculate_credibility_index(
        comprehensive_concordance_matrix, marginal_discordance_matrix
    )
    outranking_relation_matrix = calculate_outranking_relation_matrix(
        credibility_index, credibility_threshold
    )
    relation = calculate_relation(
        outranking_relation_matrix, dataset.index, boundary_profiles.index
    )

    pessimistic_assigment = calculate_pessimistic_assigment(relation)
    optimistic_assigment = calculate_optimistic_assigment(relation)

    assignment = pd.concat([pessimistic_assigment, optimistic_assigment], axis=1)
    print(assignment)

    for assignment_type in assignment.columns:
        print()
        print(assignment_type.capitalize())
        for cls in range(2, -1, -1):
            print(cls, assignment[assignment[assignment_type] == cls].index.tolist())

if __name__ == "__main__":
    electre_tri_b()
