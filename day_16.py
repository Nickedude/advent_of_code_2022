import heapq
import itertools
import math
import time
from collections import defaultdict
from typing import Tuple, List, Dict

from tqdm import tqdm


class Valve:
    def __init__(self, name: str, rate: int):
        self.name = name
        self.rate = rate
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return True


def parse_valve(line: str) -> Tuple[str, int]:
    valve, _ = line.split(";")
    name, rate = valve.split(" has flow ")
    name = name.split("Valve ")[1]
    rate = int(rate.split("rate=")[1])
    return name, rate


def parse_neighbors(line: str) -> List[str]:
    valve, neighbors = line.split(";")
    neighbors = neighbors.split("to")[1]
    prefix = "valves" if "valves" in neighbors else "valve"
    neighbors = neighbors.split(prefix)[1]
    neighbors = neighbors.split(",")
    return [n.strip(" ") for n in neighbors]


def read() -> Dict[str, Valve]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        valves = {}

        # Create valves
        for line in lines:
            name, rate = parse_valve(line)
            valves[name] = Valve(name, rate)

        # Add neighbors
        for line in lines:
            name, _ = parse_valve(line)

            for neighbor in parse_neighbors(line):
                valves[name].add_neighbor(valves[neighbor])

        return valves


def get_length_of_shortest_path(start: Valve, end: Valve) -> int:
    queue = []
    heapq.heappush(queue, (0, start))
    explored = set()

    while len(queue) > 0:
        distance, valve = heapq.heappop(queue)

        if valve.name in explored:
            continue

        explored.add(valve.name)

        if valve.name == end.name:
            return distance

        for neighbor in valve.neighbors:
            if neighbor.name not in explored:
                heapq.heappush(queue, (distance + 1, neighbor))

    raise ValueError("No path was found!")


def get_distances(valves: Dict[str, Valve]) -> Dict[str, Dict[str, int]]:
    names = sorted(list(valves.keys()))
    distances = defaultdict(dict)

    for i, fst in enumerate(names):
        for snd in names[i + 1 :]:
            distance = get_length_of_shortest_path(valves[fst], valves[snd])
            assert distance > 0

            distances[fst][snd] = distance
            distances[snd][fst] = distance

    return distances


def get_potential(
    time_remaining: int, closed_valves: Tuple[str, ...], valves: Dict[str, Valve]
) -> int:
    potential = 0
    closed_valves = set(closed_valves)
    remaining_rates = [
        valve.rate for name, valve in valves.items() if name in closed_valves
    ]
    remaining_rates = sorted(remaining_rates, reverse=True)

    while time_remaining > 0 and len(remaining_rates) > 0:
        rate = remaining_rates.pop(0)
        potential += (time_remaining - 1) * rate
        time_remaining -= 2

    return potential


def best_first_search(
    total_time: int,
    valves: Dict[str, Valve],
    distances: Dict[str, Dict[str, int]],
):
    queue = []
    explored = set()

    heapq.heappush(queue, (0, 0, "AA", total_time, tuple(valves.keys())))
    best_solution = 0

    while len(queue) > 0:
        state = heapq.heappop(queue)

        if state in explored:
            continue

        explored.add(state)

        _, released_pressure, current_valve, time_remaining, closed_valves = state
        best_solution = max(best_solution, released_pressure)

        if len(closed_valves) == 0:  # All valves open -> nothing more to explore
            continue

        # Generate neighboring states by considering opening each of the valves
        for next_valve in closed_valves:
            distance = distances[current_valve][next_valve]

            if (time_remaining - distance) > 0:
                new_closed_valves = tuple([v for v in closed_valves if v != next_valve])
                new_time_remaining = time_remaining - distance - 1
                new_released_pressure = (
                    released_pressure + new_time_remaining * valves[next_valve].rate
                )
                priority = new_released_pressure + get_potential(
                    new_time_remaining, new_closed_valves, valves
                )

                if priority >= best_solution:
                    heapq.heappush(
                        queue,
                        (
                            priority,
                            new_released_pressure,
                            next_valve,
                            new_time_remaining,
                            new_closed_valves,
                        ),
                    )

    return best_solution


def get_valve_combinations(valves: Dict[str, Valve]):
    names = [name for name, valve in valves.items() if valve.rate > 0]
    size = int(math.floor(len(names) / 2.0))
    combinations = []

    size_i_combinations = list(itertools.combinations(names, size))

    for fst in size_i_combinations:
        fst = list(fst)
        snd = [n for n in names if n not in set(fst)]
        combinations.append((fst, snd))

    return combinations


def main():
    valves = read()

    for name, valve in valves.items():
        print(f"{name} -> {[v.name for v in valve.neighbors]}")

    distances = get_distances(valves)

    start = time.time()
    max_released_pressure = best_first_search(
        total_time=30,
        valves={name: valve for name, valve in valves.items() if valve.rate > 0},
        distances=distances,
    )
    print(f"Runtime: {time.time() - start}")
    print(f"Amount of released pressure: {max_released_pressure}")

    combinations = get_valve_combinations(valves)
    max_released_pressure = 0

    for fst, snd in tqdm(combinations):
        release_by_me = best_first_search(
            total_time=26,
            valves={name: valve for name, valve in valves.items() if name in set(fst)},
            distances=distances,
        )

        release_by_elephant = best_first_search(
            total_time=26,
            valves={name: valve for name, valve in valves.items() if name in set(snd)},
            distances=distances,
        )

        max_released_pressure = max(
            max_released_pressure, release_by_me + release_by_elephant
        )

    print(f"Max released pressure: {max_released_pressure}")


if __name__ == "__main__":
    main()
