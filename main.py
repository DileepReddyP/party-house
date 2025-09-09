import argparse
import requests
from os import getenv
from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple, Callable
from itertools import product
from math import floor, log2
from time import sleep

BASE_URL = getenv("BASE_URL")
PLAYER_ID = getenv("PLAYER_ID")
CLUB_CAPACITY = 1000
# unused
MAX_REJECT = 20000


def start_game_url(scenario: int):
    return f"{BASE_URL}/new-game?scenario={scenario}&playerId={PLAYER_ID}"


def decide_and_next_url(
    game_id: str, person_index: int, accept: None | bool = None
) -> str:
    accept_param = f"&accept={str(accept).lower()}" if accept is not None else ""
    return f"{BASE_URL}/decide-and-next?gameId={game_id}&personIndex={person_index}{accept_param}"


@dataclass
class Attribute:
    name: str
    min_count: int
    inverted_dim: int
    relative_frequency: float
    # odds_ratio(rel_freq)
    attribute_value: float
    # TODO: Use this
    correlations: defaultdict[str, float]
    alpha_j: float
    utilization: int
    normalized_utilization: int


@dataclass
class GameState:
    constraints: defaultdict[str, Attribute]
    # m in paper, not needed for exp_rp
    num_dims: int
    theta: float
    alpha: float
    aggregate_capacity: int
    # game stats
    scenario: int
    game_id: str
    accepted: int
    processed: int
    rejected: int
    all_constraints_satisfied: bool
    correlation_matrix: list | None
    value_lookup: dict[Tuple[bool, ...], float] | None
    game_algorithm: Callable[[dict[str, bool], "GameState"], Tuple[bool, "GameState"]]


def value_function(rel_freq: float) -> float:
    # TODO: exponential or not?
    return 1.0 / rel_freq


def calculate_theta(constraints: defaultdict[str, Attribute]) -> float:
    value_list = [constraint.attribute_value for constraint in constraints.values()]
    v_min = min(value_list)
    v_max = sum(value_list)
    return float(v_max / v_min)


def adjust_threshold(threshold: float, current_rejects: int, needed: int) -> float:
    reduction = (current_rejects / MAX_REJECT) ** (1 / ((current_rejects / 100) + 1))
    needed_reduction = 1 - ( 1 / (needed * 10) if needed > 0 else 0)
    return threshold * (1.0 - reduction) * needed_reduction


def read_constraints(
    constraints_array,
    rel_freq,
    correlations,
) -> Tuple[float, int, defaultdict[str, Attribute]]:
    constraints = defaultdict(None)
    capacity_array: list[Tuple[str, int, int]] = []
    for constraint in constraints_array:
        name, min_count = constraint["attribute"], constraint["minCount"]
        inverted_dim = CLUB_CAPACITY - min_count
        capacity_array.append((name, min_count, inverted_dim))
    aggregate_capacity = sum(inv_dim for (_, _, inv_dim) in capacity_array)
    min_capacity = min(inv_dim for (_, _, inv_dim) in capacity_array)
    alpha = aggregate_capacity / min_capacity
    for name, min_count, inv_dim in capacity_array:
        alpha_j = float(aggregate_capacity / inv_dim)
        constraints[name] = Attribute(
            name=name,
            min_count=min_count,
            inverted_dim=inv_dim,
            alpha_j=alpha_j,
            relative_frequency=rel_freq[name],
            correlations=correlations[name],
            attribute_value=value_function(rel_freq[name]),
            utilization=0,
            normalized_utilization=0,
        )
    return alpha, aggregate_capacity, constraints


def norm_util_calc(u_j: int, c_j: int, theta: float, alpha_j: float) -> int:
    return floor((u_j / c_j) * log2(theta * alpha_j))


def exp_rp(
    person_attributes: dict[str, bool], game_state: GameState
) -> Tuple[bool, GameState]:
    accept = False
    value = 0.0
    threshold = 0.0
    can_allow = True
    needed = 0
    for attr, has_attr in person_attributes.items():
        if not has_attr:
            threshold += (2 ** game_state.constraints[attr].normalized_utilization) - 1
            can_allow = can_allow and (
                game_state.constraints[attr].inverted_dim
                > game_state.constraints[attr].utilization
            )
        else:
            needed += (
                1
                if (game_state.accepted - game_state.constraints[attr].utilization)
                < (game_state.constraints[attr].min_count)
                else 0
            )
    accept = (
        value >= adjust_threshold(threshold, game_state.rejected, needed) and can_allow
    )
    if accept:
        for attr in game_state.constraints:
            if not person_attributes[attr]:
                game_state.constraints[attr].utilization += 1
                game_state.constraints[attr].normalized_utilization = norm_util_calc(
                    game_state.constraints[attr].utilization,
                    game_state.constraints[attr].inverted_dim,
                    game_state.theta,
                    game_state.constraints[attr].alpha_j,
                )
    return accept, game_state


def should_accept(
    person_attributes: dict[str, bool], game_state: GameState
) -> Tuple[bool, GameState]:
    if game_state.all_constraints_satisfied:
        return True, game_state
    is_boring = all(not has_attr for _, has_attr in person_attributes.items())
    # too bad should have waited an hour more
    if is_boring:
        return False, game_state
    return game_state.game_algorithm(person_attributes, game_state)


def initialize_game(scenario: int) -> GameState | None:
    try:
        response = requests.get(start_game_url(scenario))
        response.raise_for_status()
        data = response.json()
        alpha, aggregate_capacity, constraints = read_constraints(
            data["constraints"],
            data["attributeStatistics"]["relativeFrequencies"],
            data["attributeStatistics"]["correlations"],
        )
        return GameState(
            scenario=scenario,
            game_id=data["gameId"],
            num_dims=len(constraints),
            constraints=constraints,
            aggregate_capacity=aggregate_capacity,
            alpha=alpha,
            all_constraints_satisfied=False,
            theta=calculate_theta(constraints),
            accepted=0,
            rejected=0,
            processed=0,
            correlation_matrix=None,
            value_lookup=None,
            game_algorithm=exp_rp,
        )
    except requests.exceptions.RequestException as e:
        print(f"Error starting new game: {e}")
        return None


def run_game(game_state: GameState, sleep_time: float):
    # first person, by not specifying accept
    try:
        response = requests.get(decide_and_next_url(game_state.game_id, 0))
        response.raise_for_status()
        data = response.json()
        while data["status"] == "running":
            person = data["nextPerson"]
            person_attributes = person["attributes"]
            accept, game_state = should_accept(person_attributes, game_state)
            if accept:
                game_state.accepted += 1
                if not game_state.all_constraints_satisfied:
                    game_state.all_constraints_satisfied = all(
                        (game_state.accepted - game_state.constraints[attr].utilization)
                        >= (game_state.constraints[attr].min_count)
                        for attr in game_state.constraints.keys()
                    )
            else:
                game_state.rejected += 1
            sleep(sleep_time)
            response = requests.get(
                decide_and_next_url(
                    game_state.game_id,
                    person["personIndex"],
                    accept,
                )
            )
            game_state.processed += 1
            if game_state.processed % 100 == 0:
                print(
                    f"Processed={game_state.processed}; In={game_state.accepted}; Out={game_state.rejected};"
                )
            response.raise_for_status()
            data = response.json()
        if data["status"] == "failed":
            print(data["reason"])
        else:
            print(f"SUCCESS!, Reject Count = {data['rejectedCount']}")
    except requests.exceptions.RequestException as e:
        print(f"Error during scenario {game_state.scenario}: {e}")


# CORRELATIONS BASED LOGIC


def precompute_all_attribute_values(
    rel_freq: dict[str, float],
    correlations: dict[str, dict[str, float]],
    constraints: list[dict[str, int | str]],
) -> Tuple[dict[Tuple[bool, ...], float], float]:
    attribute_names = list(rel_freq.keys())
    n_attributes = len(attribute_names)

    # Generate all possible combinations (2^n)
    all_combinations = list(product([False, True], repeat=n_attributes))

    value_lookup = {}
    all_values = []

    for combination in all_combinations:
        attrs = dict(zip(attribute_names, combination))
        value = calculate_combination_value(attrs, rel_freq, correlations, constraints)
        value_lookup[combination] = value
        all_values.append(value)

    max_value = max(all_values)
    min_value = min([v for v in all_values if v > 0])
    theta = max_value / min_value if min_value > 0 else 1.0

    print(f"Precomputed {len(all_combinations)} combinations")
    print(f"Value range: {min_value:.3f} to {max_value:.3f}")
    print(f"Theta (max/min ratio): {theta:.3f}")

    return value_lookup, theta


def calculate_combination_value(
    attrs: dict[str, bool],
    rel_freq: dict[str, float],
    correlations: dict[str, dict[str, float]],
    constraints: list[dict],
) -> float:
    """
    Calculate value of a specific attribute combination using multiple factors:
    1. Base scarcity (inverse frequency)
    2. Constraint urgency (how critical each attribute is)
    3. Correlation synergy (positive correlations between present attributes)
    4. Correlation penalty (negative correlations between present attributes)

    NOTE: All the numbers here are tunable knobs
    """

    present_attrs = [attr for attr, present in attrs.items() if present]

    if not present_attrs:
        return 0.0

    base_value = sum(value_function(rel_freq[attr]) for attr in present_attrs)

    urgency_multiplier = 1.0

    for constraint in constraints:
        attr, min_count = constraint["attribute"], constraint["minCount"]
        if attr in present_attrs:
            expected_count = 1000 * rel_freq[attr]
            shortage_ratio = min_count / expected_count if expected_count > 0 else 1.0
            urgency_multiplier += max(0, shortage_ratio - 1.0)

    correlation_factor = 1.0

    for i, attr1 in enumerate(present_attrs):
        for attr2 in present_attrs[i + 1 :]:
            correlation = correlations.get(attr1, {}).get(attr2, 0.0)
            correlation_factor -= correlation * 0.1

    multi_axis_bonus = 1.0 + (len(present_attrs) - 1) * 0.1
    final_value = (
        base_value * urgency_multiplier * correlation_factor * multi_axis_bonus
    )

    return max(final_value, 0.1)


def initialize_value_system(game_data):
    rel_freq = game_data["attributeStatistics"]["relativeFrequencies"]
    correlations = game_data["attributeStatistics"]["correlations"]
    constraints = game_data["constraints"]

    value_lookup, theta = precompute_all_attribute_values(
        rel_freq, correlations, constraints
    )

    return theta, value_lookup


# Modified exp_rp function using precomputed values:
def correlation_based_exp_rp(
    person_attributes: dict[str, bool],
    game_state: GameState,
) -> Tuple[bool, GameState]:
    person_value = (
        game_state.value_lookup.get(
            tuple(has_attr for has_attr in person_attributes.values()), 0.0
        )
        if game_state.value_lookup
        else 0.0
    )

    threshold = 0.0
    can_allow = True
    needed = 0
    for attr, has_attr in person_attributes.items():
        if not has_attr:
            threshold += (2 ** game_state.constraints[attr].normalized_utilization) - 1
            can_allow = can_allow and (
                game_state.constraints[attr].inverted_dim
                > game_state.constraints[attr].utilization
            )
        if has_attr:
            needed += (
                1
                if (game_state.accepted - game_state.constraints[attr].utilization)
                < (game_state.constraints[attr].min_count)
                else 0
            )

    adjusted_threshold = adjust_threshold(threshold, game_state.rejected, needed)

    accept = person_value >= adjusted_threshold and can_allow

    if accept:
        for attr_name in game_state.constraints:
            if not person_attributes[attr_name]:
                attr = game_state.constraints[attr_name]
                attr.utilization += 1
                attr.normalized_utilization = norm_util_calc(
                    attr.utilization,
                    attr.inverted_dim,
                    game_state.theta,
                    attr.alpha_j,
                )

    return accept, game_state


def correl_initialize_game(scenario: int):
    try:
        response = requests.get(start_game_url(scenario))
        response.raise_for_status()
        data = response.json()

        theta, value_lookup = initialize_value_system(data)
        alpha, aggregate_capacity, constraints = read_constraints(
            data["constraints"],
            data["attributeStatistics"]["relativeFrequencies"],
            data["attributeStatistics"]["correlations"],
        )

        game_state = GameState(
            scenario=scenario,
            game_id=data["gameId"],
            num_dims=len(constraints),
            constraints=constraints,
            aggregate_capacity=aggregate_capacity,
            alpha=alpha,
            all_constraints_satisfied=False,
            theta=theta,
            accepted=0,
            rejected=0,
            processed=0,
            correlation_matrix=data["attributeStatistics"]["correlations"],
            # Add value lookup to game state
            value_lookup=value_lookup,
            game_algorithm=correlation_based_exp_rp,
        )

        return game_state

    except requests.exceptions.RequestException as e:
        print(f"Error starting new game: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Berghain Challenge Solver: Online Multi-dimensional Knapsack"
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Scenario [1/2/3]",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["base", "correl"],
        default="base",
        help="Modification to OMKP Algorithm",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.002,
        help="Add a sleep if API throws an error",
    )
    args = parser.parse_args()
    scenario = args.scenario
    sleep_time = args.sleep
    algo = args.algo
    print(
        f"Starting scenario {scenario} with algorithm {algo} and sleep {sleep_time * 1000}ms:"
    )
    new_game = (
        initialize_game(scenario)
        if algo == "base"
        else correl_initialize_game(scenario)
    )
    if new_game is not None:
        run_game(new_game, sleep_time)


if __name__ == "__main__":
    main()
