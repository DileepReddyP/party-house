import argparse
import requests
from os import getenv
from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple
import math
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
    # unused
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


def odds_ratio(rel_freq: float) -> float:
    return (1.0 - rel_freq) / rel_freq


def calculate_theta(constraints: defaultdict[str, Attribute]) -> float:
    value_list = [constraint.attribute_value for constraint in constraints.values()]
    v_min = min(value_list)
    v_max = sum(value_list)
    return float(v_max / v_min)


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
            attribute_value=odds_ratio(rel_freq[name]),
            utilization=0,
            normalized_utilization=0,
        )
    return alpha, aggregate_capacity, constraints


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
        )
    except requests.exceptions.RequestException as e:
        print(f"Error starting new game: {e}")
        return None


def norm_util_calc(u_j: int, c_j: int, theta: float, alpha_j: float) -> int:
    return math.floor((u_j / c_j) * math.log2(theta * alpha_j))


def exp_rp(
    person_attributes: dict[str, bool], game_state: GameState
) -> Tuple[bool, GameState]:
    accept = False
    value = 0.0
    threshold = 0.0
    can_allow = True
    for attr, has_attr in person_attributes.items():
        if not has_attr:
            value += game_state.constraints[attr].attribute_value
            threshold += (2 ** game_state.constraints[attr].normalized_utilization) - 1
            can_allow = (
                can_allow
                and game_state.constraints[attr].inverted_dim
                > game_state.constraints[attr].utilization
            )
    if value >= threshold and can_allow:
        accept = True
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
    return exp_rp(person_attributes, game_state)


def run_game(game_state: GameState):
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
            sleep(0.05)
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
                    f"Processed: {game_state.processed}; Accepted: {game_state.accepted}; Rejected {game_state.rejected}"
                )
            response.raise_for_status()
            data = response.json()
        if data["status"] == "failed":
            print(data["reason"])
        else:
            print(f"SUCCESS!, Reject Count = {data['rejectedCount']}")
    except requests.exceptions.RequestException as e:
        print(f"Error during scenario {game_state.scenario}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Berghain Challenge Solver: Online Multi-dimensional Knapsack")
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Scenario [1/2/3]",
    )
    args = parser.parse_args()
    scenario = args.scenario
    print(f"Starting scenario {scenario}:")
    new_game = initialize_game(scenario)
    if new_game is not None:
        run_game(new_game)


if __name__ == "__main__":
    main()
