"""
Author: Ronen Huang

CSE 163

This program loads the individual FIFA datasets from 2015 to 2022,
combines the loaded FIFA datasets, and cleans the combined FIFA dataset
from 2015 to 2022 for the machine learning tasks.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# File paths for the individual FIFA datasets from 2015 to 2022 (in order).
PATH_15 = 'data/players_15.csv'
PATH_16 = 'data/players_16.csv'
PATH_17 = 'data/players_17.csv'
PATH_18 = 'data/players_18.csv'
PATH_19 = 'data/players_19.csv'
PATH_20 = 'data/players_20.csv'
PATH_21 = 'data/players_21.csv'
PATH_22 = 'data/players_22.csv'

# File path to save the cleaned, combined FIFA dataset from 2015 to 2022 ready
# to use for machine learning tasks.
SAVE_PATH = 'data/players_15_to_22_data.csv'


def load_data(path_15: str, path_16: str, path_17: str, path_18: str,
              path_19: str, path_20: str, path_21: str, path_22: str) ->\
                pd.DataFrame:
    """
    From the given file paths for individual FIFA datasets from 2015 to 2022,
    returns the combined FIFA data from 2015 to 2022 with irrelevant columns
    (usually repetitive or noticeable amount of missing values) as a
    pandas DataFrame.
    """

    players_15 = pd.read_csv(path_15)
    players_16 = pd.read_csv(path_16)
    players_17 = pd.read_csv(path_17)
    players_18 = pd.read_csv(path_18)
    players_19 = pd.read_csv(path_19)
    players_20 = pd.read_csv(path_20)
    players_21 = pd.read_csv(path_21)
    players_22 = pd.read_csv(path_22)

    players_15['year'] = [2015] * players_15.shape[0]
    players_16['year'] = [2016] * players_16.shape[0]
    players_17['year'] = [2017] * players_17.shape[0]
    players_18['year'] = [2018] * players_18.shape[0]
    players_19['year'] = [2019] * players_19.shape[0]
    players_20['year'] = [2020] * players_20.shape[0]
    players_21['year'] = [2021] * players_21.shape[0]
    players_22['year'] = [2022] * players_22.shape[0]

    players_15_to_22 = pd.concat(
        [players_15, players_16, players_17, players_18, players_19,
         players_20, players_21, players_22],
        ignore_index=True
    )

    irrelevant_cols = ['player_url', 'long_name', 'dob', 'nation_position',
                       'nation_team_id', 'nation_jersey_number', 'body_type',
                       'real_face', 'release_clause_eur', 'player_tags',
                       'mentality_composure', 'ls', 'st', 'rs', 'lw',
                       'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm',
                       'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm',
                       'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
                       'player_face_url', 'club_logo_url', 'club_flag_url',
                       'nation_logo_url', 'nation_flag_url']
    players_15_to_22_relevant = players_15_to_22.drop(columns=irrelevant_cols)

    return players_15_to_22_relevant


def clean_player_positions(player_positions: str) -> str:
    """
    From the given player_positions (a String) with each position separated
    by commas, returns the cleaned player_positions with each position
    separated by spaces as a String.

    If there are no player_positions given (NaN),
    then an empty String is returned.
    """

    if isinstance(player_positions, float):
        return ""
    else:
        player_positions = player_positions.split(',')
        player_positions_clean = ""
        for player_position in player_positions:
            player_positions_clean += player_position
        return player_positions_clean


def clean_work_rate(work_rates: str) -> str:
    """
    From the given work_rates (a String) with the attack work rate
    separated from the defense work rate by a forward slash, returns
    the cleaned work_rates with the attack work rate separated from
    the defense work rate by a space as a String.

    If there are no work_rates given (NaN), then an empty String is returned.
    """

    if isinstance(work_rates, float):
        return ""
    else:
        work_rates = work_rates.split('/')
        work_rate_atk = work_rates[0].lower() + "_work_rate_atk"
        work_rate_def = work_rates[1].lower() + "_work_rate_def"
        work_rate_clean = work_rate_atk + " " + work_rate_def
        return work_rate_clean


def clean_player_traits(player_traits: str) -> str:
    """
    From the given player_traits (a String) with each trait separated
    by commas, returns the cleaned player_traits with
    each trait separated by spaces as a String.

    If there are no player_traits given (NaN),
    then an empty String is returned.
    """

    if isinstance(player_traits, float):
        return ""
    else:
        player_traits = player_traits.split(', ')
        player_traits_clean = ""
        for player_trait in player_traits:
            trait_parts = player_trait.split()
            player_trait_clean = ""
            for trait_part in trait_parts:
                if trait_part != "(AI)":
                    if trait_part == "1-on-1":
                        trait_part = "one_on_one"
                    player_trait_clean += trait_part + "_"
            player_trait_clean = player_trait_clean[:-1]
            player_trait_clean = player_trait_clean.replace('-', '_')
            player_traits_clean += player_trait_clean + " "
        return player_traits_clean[:-1]


def replace_missing_values(players_15_to_22_cleaned: pd.DataFrame) ->\
        pd.DataFrame:
    """
    From the given players_15_to_22_cleaned (a pandas DataFrame) cleaned
    combined FIFA data from 2015 to 2022, returns the combined FIFA data
    from 2015 to 2022 with the missing values (NaN) filled (determined
    by column type) and the removal of any players that are free agents
    as a pandas DataFrame.
    """

    players_15_to_22_cleaned_copy = players_15_to_22_cleaned.copy()
    players_15_to_22_cleaned_copy =\
        players_15_to_22_cleaned_copy.loc[
            players_15_to_22_cleaned['club_team_id'].notnull(), :
        ]
    for column in players_15_to_22_cleaned_copy.columns:
        column_type = players_15_to_22_cleaned_copy[column].dtype
        if column_type == 'int64' or column_type == 'float64':
            players_15_to_22_cleaned_copy[column] = \
                players_15_to_22_cleaned_copy[column].fillna(0).astype('int64')
        else:
            players_15_to_22_cleaned_copy[column] = \
                players_15_to_22_cleaned_copy[column].fillna("")
    return players_15_to_22_cleaned_copy


def player_positions_labels(players_15_to_22_cleaned: pd.DataFrame) ->\
        pd.DataFrame:
    """
    From the given players_15_to_22_cleaned (a pandas DataFrame) cleaned
    combined FIFA data from 2015 to 2022, returns for each player whether
    they play a certain position or not (0 for no, 1 for yes) as a pandas
    DataFrame (each row represents a player and the columns are the positions).
    """

    vectorizer_player_positions = CountVectorizer()
    count_player_positions_matrix = vectorizer_player_positions.fit_transform(
        players_15_to_22_cleaned['player_positions'])
    player_positions_data = pd.DataFrame(
        count_player_positions_matrix.toarray(),
        index=players_15_to_22_cleaned.index,
        columns=vectorizer_player_positions.get_feature_names_out()
    )
    return player_positions_data


def work_rate_labels(players_15_to_22_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    From the given players_15_to_22_cleaned (a pandas DataFrame) cleaned
    combined FIFA data from 2015 to 2022, returns for each player whether
    they have a certain workrate or not (0 for no, 1 for yes) as a pandas
    DataFrame (each row represents a player and the columns are the work rate
    types).
    """

    vectorizer_work_rate = CountVectorizer()
    count_work_rate_matrix = vectorizer_work_rate.fit_transform(
        players_15_to_22_cleaned['work_rate']
    )
    work_rate_data = pd.DataFrame(
        count_work_rate_matrix.toarray(),
        index=players_15_to_22_cleaned.index,
        columns=vectorizer_work_rate.get_feature_names_out()
    )
    return work_rate_data


def player_traits_labels(players_15_to_22_cleaned: pd.DataFrame) ->\
        pd.DataFrame:
    """
    From the given players_15_to_22_cleaned (a pandas DataFrame) cleaned
    combined FIFA data from 2015 to 2022, returns for each player whether they
    have a certain trait or not (0 for no, 1 for yes) as a pandas DataFrame
    (each row represents a player and the columns are the traits).
    """

    vectorizer_player_traits = CountVectorizer()
    count_player_traits_matrix = vectorizer_player_traits.fit_transform(
        players_15_to_22_cleaned['player_traits']
    )
    player_traits_data = pd.DataFrame(
        count_player_traits_matrix.toarray(),
        index=players_15_to_22_cleaned.index,
        columns=vectorizer_player_traits.get_feature_names_out()
    )
    return player_traits_data


def combine_data(players_15_to_22_cleaned: pd.DataFrame,
                 player_positions_data: pd.DataFrame,
                 work_rate_data: pd.DataFrame,
                 player_traits_data: pd.DataFrame) -> pd.DataFrame:
    """
    From the given players_15_to_22_cleaned (a pandas DataFrame) cleaned
    combined FIFA data from 2015 to 2022, the given player_positions_data
    (a pandas DataFrame) data of whether a player plays a certain position
    or not, the given work_rate_data (a pandas DataFrame) data of whether a
    player has a certain work rate or not, and the given player_traits_data
    (a pandas DataFrame) data of whether a player has a certain trait or not,
    returns the cleaned, combined FIFA data from 2015 to 2022 ready to use for
    machine learning tasks as a pandas DataFramee.
    """

    players_15_to_22_data = players_15_to_22_cleaned.join(
        player_positions_data
    )
    players_15_to_22_data = players_15_to_22_data.join(work_rate_data)
    players_15_to_22_data = players_15_to_22_data.join(player_traits_data)
    players_15_to_22_data = players_15_to_22_data.drop(
        columns=['player_positions', 'work_rate', 'player_traits']
    )
    return players_15_to_22_data


def load_clean():
    """
    Load and Clean FIFA data.
    """
    players_15_to_22_relevant = load_data(
        PATH_15, PATH_16, PATH_17, PATH_18, PATH_19, PATH_20, PATH_21, PATH_22
    )

    players_15_to_22_cleaned = players_15_to_22_relevant.copy()

    # Clean the 'player_positions', 'work_rate', and 'player_traits' columns.
    players_15_to_22_cleaned['player_positions'] =\
        players_15_to_22_relevant['player_positions'].apply(
            clean_player_positions
        )
    players_15_to_22_cleaned['work_rate'] =\
        players_15_to_22_relevant['work_rate'].apply(clean_work_rate)
    players_15_to_22_cleaned['player_traits'] =\
        players_15_to_22_relevant['player_traits'].apply(clean_player_traits)

    players_15_to_22_cleaned = replace_missing_values(players_15_to_22_cleaned)

    player_positions_data = player_positions_labels(players_15_to_22_cleaned)
    work_rate_data = work_rate_labels(players_15_to_22_cleaned)
    player_traits_data = player_traits_labels(players_15_to_22_cleaned)

    players_15_to_22_data = combine_data(
        players_15_to_22_cleaned, player_positions_data,
        work_rate_data, player_traits_data
    )

    # Saves the cleaned, combined FIFA data from 2015 to 2022
    # ready to use for machine learning tasks under the
    # 'data' directory as a CSV 'players_15_to_22_data.csv'.
    players_15_to_22_data.to_csv(SAVE_PATH, index=False)
