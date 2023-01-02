import heapq
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, List

Ore = int
Clay = int
Geode = int
Obsidian = int


@dataclass
class BluePrint:

    id_: int
    ore_robot_cost: Ore
    clay_robot_cost: Ore
    geode_robot_cost: Tuple[Ore, Obsidian]
    obsidian_robot_cost: Tuple[Ore, Clay]


@dataclass
class State:

    minutes_left: int = 0

    amount_of_ore: Ore = 0
    amount_of_clay: Clay = 0
    amount_of_geodes: Geode = 0
    amount_of_obsidian: Obsidian = 0

    ore_collection_rate: Ore = 0
    clay_collection_rate: Clay = 0
    geode_collection_rate: Geode = 0
    obsidian_collection_rate: Obsidian = 0

    def __hash__(self):
        return hash(
            (
                self.minutes_left,
                self.amount_of_ore,
                self.amount_of_clay,
                self.amount_of_geodes,
                self.amount_of_obsidian,
                self.ore_collection_rate,
                self.clay_collection_rate,
                self.geode_collection_rate,
                self.obsidian_collection_rate,
            )
        )

    def __lt__(self, other):
        return self.amount_of_geodes < other.amount_of_geodes

    def step(self):
        if self.minutes_left == 0:
            raise ValueError("Can't take a step - no time left!")

        return State(
            minutes_left=self.minutes_left - 1,
            amount_of_ore=self.amount_of_ore + self.ore_collection_rate,
            amount_of_clay=self.amount_of_clay + self.clay_collection_rate,
            amount_of_geodes=self.amount_of_geodes + self.geode_collection_rate,
            amount_of_obsidian=self.amount_of_obsidian + self.obsidian_collection_rate,
            ore_collection_rate=self.ore_collection_rate,
            clay_collection_rate=self.clay_collection_rate,
            geode_collection_rate=self.geode_collection_rate,
            obsidian_collection_rate=self.obsidian_collection_rate,
        )

    def can_build_ore_robot(self, blueprint: BluePrint) -> bool:
        return self.amount_of_ore >= blueprint.ore_robot_cost

    def can_build_clay_robot(self, blueprint: BluePrint) -> bool:
        return self.amount_of_ore >= blueprint.clay_robot_cost

    def can_build_geode_robot(self, blueprint: BluePrint) -> bool:
        ore_cost, obsidian_cost = blueprint.geode_robot_cost
        return (
            self.amount_of_ore >= ore_cost and self.amount_of_obsidian >= obsidian_cost
        )

    def can_build_obsidian_robot(self, blueprint: BluePrint) -> bool:
        ore_cost, clay_cost = blueprint.obsidian_robot_cost
        return self.amount_of_ore >= ore_cost and self.amount_of_clay >= clay_cost

    def build_ore_robot(self, blueprint: BluePrint):
        neighbor = self.step()
        neighbor.amount_of_ore -= blueprint.ore_robot_cost
        neighbor.ore_collection_rate += 1
        return neighbor

    def build_clay_robot(self, blueprint: BluePrint):
        neighbor = self.step()
        neighbor.amount_of_ore -= blueprint.clay_robot_cost
        neighbor.clay_collection_rate += 1
        return neighbor

    def build_geode_robot(self, blueprint: BluePrint):
        neighbor = self.step()
        ore_cost, obsidian_cost = blueprint.geode_robot_cost
        neighbor.amount_of_ore -= ore_cost
        neighbor.amount_of_obsidian -= obsidian_cost
        neighbor.geode_collection_rate += 1
        return neighbor

    def build_obsidian_robot(self, blueprint: BluePrint):
        ore_cost, clay_cost = blueprint.obsidian_robot_cost
        neighbor = self.step()
        neighbor.amount_of_ore -= ore_cost
        neighbor.amount_of_clay -= clay_cost
        neighbor.obsidian_collection_rate += 1
        return neighbor


def parse_single_cost(text: str, expected_units: List[str]) -> Tuple[int, ...]:
    text = text.split("costs ")[1]
    parts = text.split(" ")

    if "and" in parts:
        parts.remove("and")

    costs = []

    for i in range(0, len(parts), 2):
        cost = parts[i]
        unit = parts[i + 1]
        exp_unit = expected_units.pop(0)
        assert unit == exp_unit, f"Expected {exp_unit} but got {unit}"
        costs.append(int(cost))

    return tuple(costs)


def parse_blueprint(line: str) -> BluePrint:
    id_, costs = line.split(":")
    id_ = int(id_.split(" ")[1])

    ore_robot, clay_robot, obsidian_robot, geode_robot, _ = costs.split(".")

    ore_robot_cost = parse_single_cost(ore_robot, ["ore"])[0]
    clay_robot_cost = parse_single_cost(clay_robot, ["ore"])[0]
    geode_robot_cost: Tuple[Ore, Obsidian] = parse_single_cost(
        geode_robot, ["ore", "obsidian"]
    )
    obsidian_robot_cost: Tuple[Ore, Clay] = parse_single_cost(
        obsidian_robot, ["ore", "clay"]
    )

    return BluePrint(
        id_=id_,
        ore_robot_cost=ore_robot_cost,
        clay_robot_cost=clay_robot_cost,
        geode_robot_cost=geode_robot_cost,
        obsidian_robot_cost=obsidian_robot_cost,
    )


def read():
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        return [parse_blueprint(line) for line in lines]


def get_neighbors(state: State, blueprint: BluePrint) -> List[State]:
    neighbors = []

    if state.minutes_left == 0:
        return neighbors

    # Do nothing
    neighbors.append(state.step())

    if state.can_build_ore_robot(blueprint):
        neighbors.append(state.build_ore_robot(blueprint))

    if state.can_build_clay_robot(blueprint):
        neighbors.append(state.build_clay_robot(blueprint))

    if state.can_build_geode_robot(blueprint):
        neighbors.append(state.build_geode_robot(blueprint))

    if state.can_build_obsidian_robot(blueprint):
        neighbors.append(state.build_obsidian_robot(blueprint))

    return neighbors


def heuristic(state: State, blueprint: BluePrint) -> int:
    """Get an overly optimistic estimate of how many geodes can be cracked in the remaining time."""
    current = deepcopy(state)

    while current.minutes_left > 0:
        if current.can_build_geode_robot(blueprint):
            current = current.build_geode_robot(blueprint)
        else:  # Assume we can build a whole lot more than we actually can
            current.ore_collection_rate += 1
            current.clay_collection_rate += 1
            current.obsidian_collection_rate += 1
            current = current.step()

    return current.amount_of_geodes - state.amount_of_geodes


def best_first_search(blueprint: BluePrint, minutes_left: int) -> int:
    queue = []
    explored = set()

    heapq.heappush(queue, (0, State(minutes_left=minutes_left, ore_collection_rate=1)))

    while len(queue) > 0:
        _, state = heapq.heappop(queue)

        if state in explored:
            continue

        explored.add(state)

        if state.minutes_left == 0:
            return state.amount_of_geodes

        for neighbor in get_neighbors(state, blueprint):
            priority = -(neighbor.amount_of_geodes + heuristic(neighbor, blueprint))
            heapq.heappush(queue, (priority, neighbor))

    raise ValueError("Couldn't find a solution")


def main():
    blueprints = read()
    sum_of_quality_levels = 0

    for b in blueprints:
        print(b)
        start = time.time()
        num_geodes = best_first_search(b, minutes_left=24)
        print(f"Num geodes cracked: {num_geodes}")
        print(f"Runtime: {time.time() - start}")
        quality_level = b.id_ * num_geodes
        sum_of_quality_levels += quality_level
        print("----------------------")

    print(f"Sum of quality levels: {sum_of_quality_levels}")

    product_of_num_geodes = 1
    num_blueprints = 3

    for b in blueprints[:num_blueprints]:
        print(b)
        start = time.time()
        num_geodes = best_first_search(b, minutes_left=32)
        print(f"Num geodes cracked: {num_geodes}")
        print(f"Runtime: {time.time() - start}")
        product_of_num_geodes = product_of_num_geodes * num_geodes
        print("----------------------")

    print(
        f"Product of geodes for the first {num_blueprints} blueprints: {product_of_num_geodes}"
    )


if __name__ == "__main__":
    main()
