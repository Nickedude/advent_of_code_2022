from typing import List, Set, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def read():
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))

        coordinates = []

        for line in lines:
            coordinate = line.split(",")
            coordinate = list(map(int, coordinate))
            coordinates.append(np.array(coordinate))

        return np.stack(coordinates, axis=0)


def get_neighbors(coordinate: np.ndarray) -> List[np.ndarray]:
    neighbors = []

    x_offset = np.array([1, 0, 0])
    neighbors.append(coordinate + x_offset)
    neighbors.append(coordinate - x_offset)

    y_offset = np.array([0, 1, 0])
    neighbors.append(coordinate + y_offset)
    neighbors.append(coordinate - y_offset)

    z_offset = np.array([0, 0, 1])
    neighbors.append(coordinate + z_offset)
    neighbors.append(coordinate - z_offset)

    return neighbors


def is_occupied(coordinate: np.ndarray, occupied: np.ndarray) -> bool:
    return (occupied == coordinate).all(axis=1).any()


def get_num_not_covered_faces(coordinates: np.ndarray):
    num_not_covered = 0

    for i in range(len(coordinates)):
        for neighbor in get_neighbors(coordinates[i, :]):
            if not is_occupied(neighbor, coordinates):
                num_not_covered += 1

    return num_not_covered


def get_num_air_pocket_faces(coordinates: np.ndarray) -> int:
    potential_air_pockets = []

    # Get all neighboring coordinates that are not occupied
    for i in range(len(coordinates)):
        for neighbor in get_neighbors(coordinates[i, :]):
            if not is_occupied(neighbor, coordinates):
                potential_air_pockets.append(neighbor)

    # Get unique coordinates
    potential_air_pockets = np.stack(potential_air_pockets, axis=0)
    potential_air_pockets = np.unique(potential_air_pockets, axis=0)

    # Define a max and min coordinate to limit the BFS
    min_ = coordinates.min(axis=0) - np.ones(3, dtype=int)
    max_ = coordinates.max(axis=0) + np.ones(3, dtype=int)
    print(min_, max_)

    # Do a BFS to find all coordinates on the outside
    outside = bfs(max_, coordinates, max_, min_)

    # Start a BFS from each potential pocket and keep track of all explored nodes
    pockets = set()

    for pocket in tqdm(potential_air_pockets):
        if tuple(pocket) in outside or tuple(pocket) in pockets:
            continue

        explored = bfs(pocket, coordinates, max_, min_)
        pockets = pockets.union(explored)

    num_air_pocket_faces = 0

    # For each pocket coordinate, subtract the number of adjacent coordinates that are occupied
    for pocket in pockets:
        pocket = np.array(pocket)
        for neighbor in get_neighbors(pocket):
            if is_occupied(neighbor, coordinates):
                num_air_pocket_faces += 1

    return num_air_pocket_faces


def bfs(start: np.ndarray, occupied: np.ndarray, max_: np.ndarray, min_: np.ndarray) -> Set[Tuple[int, int, int]]:
    queue = [start]
    explored = set()

    while len(queue) > 0:
        current = queue.pop(0)

        if tuple(current) in explored:
            continue

        explored.add(tuple(current))

        for neighbor in get_neighbors(current):
            if (neighbor < min_).any() or (neighbor > max_).any():
                continue

            if tuple(neighbor) not in explored and not is_occupied(neighbor, occupied):
                queue.append(neighbor)

    return explored


def plot_coordinates(coordinates: np.ndarray):
    ax = plt.axes(projection='3d')
    ax.scatter3D(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    plt.show()


def main():
    coordinates = read()
    plot_coordinates(coordinates)

    num_not_covered = get_num_not_covered_faces(coordinates)
    print(f"Number of not covered faces: {num_not_covered}")

    num_air_pockets = get_num_air_pocket_faces(coordinates)
    print(f"Number of truly not covered faces: {num_not_covered - num_air_pockets}")


if __name__ == "__main__":
    main()
