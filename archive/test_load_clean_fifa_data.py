"""
****, ****, Ronen H

CSE 163

This programs tests the validity
of the functions for the
load_clean_fifa_data module.
"""

import load_clean_fifa_data
from unittest import TestCase
import pandas as pd
import numpy as np

# File paths for the individual FIFA datasets
# from 2015 to 2022 (in order).
PATH_15 = '../data/players_15.csv'
PATH_16 = '../data/players_16.csv'
PATH_17 = '../data/players_17.csv'
PATH_18 = '../data/players_18.csv'
PATH_19 = '../data/players_19.csv'
PATH_20 = '../data/players_20.csv'
PATH_21 = '../data/players_21.csv'
PATH_22 = '../data/players_22.csv'


def test_load_data(path_15, path_16, path_17, path_18,
                   path_19, path_20, path_21, path_22):
    """
    Tests the validity of the load_data function (dimensions
    and columns of the combined FIFA dataset from 2015 to 2022).
    """

    players_15_to_22_relevant = load_clean_fifa_data.load_data(
        path_15, path_16, path_17, path_18, path_19, path_20, path_21, path_22)

    players_15 = pd.read_csv(path_15)
    players_16 = pd.read_csv(path_16)
    players_17 = pd.read_csv(path_17)
    players_18 = pd.read_csv(path_18)
    players_19 = pd.read_csv(path_19)
    players_20 = pd.read_csv(path_20)
    players_21 = pd.read_csv(path_21)
    players_22 = pd.read_csv(path_22)

    players_15['year'] = [2015] * players_15.shape[0]

    irrelevant_cols = ['player_url', 'long_name', 'dob', 'nation_position',
                       'nation_team_id', 'nation_jersey_number', 'body_type',
                       'real_face', 'release_clause_eur', 'player_tags',
                       'mentality_composure', 'ls', 'st', 'rs', 'lw', 'lf',
                       'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm',
                       'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb',
                       'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
                       'player_face_url', 'club_logo_url', 'club_flag_url',
                       'nation_logo_url', 'nation_flag_url']
    players_15_relevant = players_15.drop(columns=irrelevant_cols)

    expected_num_rows = players_15.shape[0] + players_16.shape[0] + \
        players_17.shape[0] + players_18.shape[0] + players_19.shape[0] + \
        players_20.shape[0] + players_21.shape[0] + players_22.shape[0]
    expected_num_cols = players_15_relevant.shape[1]

    TestCase().assertEqual(expected_num_rows,
                           players_15_to_22_relevant.shape[0])
    TestCase().assertEqual(expected_num_cols,
                           players_15_to_22_relevant.shape[1])
    TestCase().assertEqual(list(players_15_relevant.columns),
                           list(players_15_to_22_relevant.columns))

    return players_15_to_22_relevant


def test_clean_player_positions(players_15_to_22_relevant):
    """
    Tests the correctness of the clean_player_positions
    function.
    """

    player_positions_case_one = "CF, LW, RW"
    cleaned_player_positions_case_one = \
        load_clean_fifa_data.clean_player_positions(player_positions_case_one)
    TestCase().assertEqual("CF LW RW",
                           cleaned_player_positions_case_one)

    player_positions_case_two = np.nan
    cleaned_player_positions_case_two = \
        load_clean_fifa_data.clean_player_positions(player_positions_case_two)
    TestCase().assertEqual("", cleaned_player_positions_case_two)

    return players_15_to_22_relevant['player_positions'].apply(
        load_clean_fifa_data.clean_player_positions)


def test_clean_work_rate(players_15_to_22_relevant):
    """
    Tests the correctness of the clean_work_rate
    function.
    """

    work_rate_case_one = "High/Medium"
    cleaned_work_rate_case_one = \
        load_clean_fifa_data.clean_work_rate(work_rate_case_one)
    TestCase().assertEqual("high_work_rate_atk medium_work_rate_def",
                           cleaned_work_rate_case_one)

    work_rate_case_two = np.nan
    cleaned_work_rate_case_two = \
        load_clean_fifa_data.clean_work_rate(work_rate_case_two)
    TestCase().assertEqual("", cleaned_work_rate_case_two)

    return players_15_to_22_relevant['work_rate'].apply(
        load_clean_fifa_data.clean_work_rate)


def test_clean_player_traits(players_15_to_22_relevant):
    """
    Tests the correctness of the clean_player_traits
    function.
    """

    player_traits_case_one = "Playmaker (AI), Injury Prone, 1-on-1 Rush"
    cleaned_player_traits_case_one = \
        load_clean_fifa_data.clean_player_traits(player_traits_case_one)
    TestCase().assertEqual("Playmaker Injury_Prone one_on_one_Rush",
                           cleaned_player_traits_case_one)

    player_traits_case_two = np.nan
    cleaned_player_traits_case_two = \
        load_clean_fifa_data.clean_player_traits(player_traits_case_two)
    TestCase().assertEqual("", cleaned_player_traits_case_two)

    return players_15_to_22_relevant['player_traits'].apply(
        load_clean_fifa_data.clean_player_traits)


def test_replace_missing_values(players_15_to_22_cleaned):
    """
    Tests the correctness of the replace_missing_values
    function.
    """

    players_15_to_22_cleaned = load_clean_fifa_data \
        .replace_missing_values(players_15_to_22_cleaned)
    players_15_to_22_cleaned_copy = players_15_to_22_cleaned.dropna()
    TestCase().assertEqual(players_15_to_22_cleaned.shape[0],
                           players_15_to_22_cleaned_copy.shape[0])

    return players_15_to_22_cleaned


def test_player_positions_labels(players_15_to_22_cleaned):
    """
    Tests the validity of the player_positions_labels
    function (dimensions and columns).
    """

    player_positions_data = load_clean_fifa_data \
        .player_positions_labels(players_15_to_22_cleaned)

    expected_cols = ['cam', 'cb', 'cdm', 'cf', 'cm',
                     'gk', 'lb', 'lm', 'lw', 'lwb', 'rb',
                     'rm', 'rw', 'rwb', 'st']
    expected_num_rows = players_15_to_22_cleaned.shape[0]

    TestCase().assertEqual(expected_cols,
                           list(player_positions_data.columns))
    TestCase().assertEqual(expected_num_rows,
                           player_positions_data.shape[0])

    return player_positions_data


def test_work_rate_labels(players_15_to_22_cleaned):
    """
    Tests the validity of the work_rate_labels
    function (dimensions and columns).
    """

    work_rate_data = load_clean_fifa_data \
        .work_rate_labels(players_15_to_22_cleaned)

    expected_cols = ['high_work_rate_atk', 'high_work_rate_def',
                     'low_work_rate_atk', 'low_work_rate_def',
                     'medium_work_rate_atk', 'medium_work_rate_def']
    expected_num_rows = players_15_to_22_cleaned.shape[0]

    TestCase().assertEqual(expected_cols,
                           list(work_rate_data.columns))
    TestCase().assertEqual(expected_num_rows,
                           work_rate_data.shape[0])

    return work_rate_data


def test_player_traits_labels(players_15_to_22_cleaned):
    """
    Tests the validity of the player_traits_labels
    function (dimensions and columns).
    """

    player_traits_data = load_clean_fifa_data \
        .player_traits_labels(players_15_to_22_cleaned)

    expected_cols = ['avoids_using_weaker_foot', 'backs_into_player',
                     'beat_offside_trap', 'cautious_with_crosses', 'chip_shot',
                     'comes_for_crosses', 'diver', 'dives_into_tackles',
                     'early_crosser', 'finesse_shot', 'flair',
                     'giant_throw_in', 'gk_long_throw', 'gk_up_for_corners',
                     'injury_free', 'injury_prone', 'leadership',
                     'long_passer', 'long_shot_taker', 'long_throw_in',
                     'one_club_player', 'one_on_one_rush', 'outside_foot_shot',
                     'playmaker', 'power_free_kick', 'power_header', 'puncher',
                     'rushes_out_of_goal', 'saves_with_feet', 'selfish',
                     'set_play_specialist', 'solid_player', 'speed_dribbler',
                     'swerve_pass', 'takes_finesse_free_kicks',
                     'target_forward', 'team_player', 'technical_dribbler',
                     'through_ball']
    expected_num_rows = players_15_to_22_cleaned.shape[0]

    TestCase().assertEqual(expected_cols,
                           list(player_traits_data.columns))
    TestCase().assertEqual(expected_num_rows,
                           player_traits_data.shape[0])

    return player_traits_data


def test_combine_data(players_15_to_22_cleaned, player_positions_data,
                      work_rate_data, player_traits_data):
    """
    Tests the validity of the combine_data
    function (dimensions and columns).
    """

    players_15_to_22_data = load_clean_fifa_data \
        .combine_data(players_15_to_22_cleaned, player_positions_data,
                      work_rate_data, player_traits_data)

    # -3 for dropping the 'player_positions', 'work_rate',
    # and 'player_traits' columns.
    expected_num_cols = players_15_to_22_cleaned.shape[1] + \
        player_positions_data.shape[1] + work_rate_data.shape[1] + \
        player_traits_data.shape[1] - 3
    expected_num_rows = players_15_to_22_cleaned.shape[0]

    TestCase().assertEqual(expected_num_cols,
                           players_15_to_22_data.shape[1])
    TestCase().assertEqual(expected_num_rows,
                           players_15_to_22_data.shape[0])


def main():
    players_15_to_22_relevant = test_load_data(PATH_15, PATH_16,
                                               PATH_17, PATH_18,
                                               PATH_19, PATH_20,
                                               PATH_21, PATH_22)

    players_15_to_22_cleaned = players_15_to_22_relevant.copy()
    players_15_to_22_cleaned['player_positions'] = \
        test_clean_player_positions(players_15_to_22_relevant)
    players_15_to_22_cleaned['work_rate'] = \
        test_clean_work_rate(players_15_to_22_relevant)
    players_15_to_22_cleaned['player_traits'] = \
        test_clean_player_traits(players_15_to_22_relevant)

    players_15_to_22_cleaned = test_replace_missing_values(
        players_15_to_22_cleaned)

    player_positions_data = test_player_positions_labels(
        players_15_to_22_cleaned)
    work_rate_data = test_work_rate_labels(
        players_15_to_22_cleaned)
    player_traits_data = test_player_traits_labels(
        players_15_to_22_cleaned)

    test_combine_data(players_15_to_22_cleaned, player_positions_data,
                      work_rate_data, player_traits_data)


if __name__ == '__main__':
    main()
