from typing import List, Tuple

import numpy as np


def read() -> List[np.ndarray]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))

        rocks = []

        for line in lines:
            line = line.split(" -> ")
            rock = np.zeros((len(line), 2), dtype=int)

            for i, coordinate in enumerate(line):
                col, row = coordinate.split(",")
                rock[i] = [row, col]

            rocks.append(rock)

        return rocks


def get_range(start: int, end: int) -> np.ndarray:
    if start > end:
        tmp = end
        end = start
        start = tmp

    return np.arange(start, end + 1, step=1, dtype=int)


def get_coordinates_from_line(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    if start[0] == end[0]:
        cols = get_range(start[1], end[1])
        rows = np.ones(len(cols), dtype=int) * start[0]
    elif start[1] == end[1]:
        rows = get_range(start[0], end[0])
        cols = np.ones(len(rows), dtype=int) * start[1]
    else:
        raise ValueError("Line is not straight!")

    return np.stack([rows, cols], axis=1)


def create_grid(rocks: List[np.ndarray], use_floor: bool) -> np.ndarray:
    max_row = 0
    max_col = 0
    min_col = 500

    for rock in rocks:
        row = np.max(rock[:, 0])
        col = np.max(rock[:, 1])

        if row > max_row:
            max_row = row

        if col > max_col:
            max_col = col

        col = np.min(rock[:, 1])

        if col < min_col:
            min_col = col

    grid = np.zeros((max_row + 1, max_col + 1), dtype=int)

    for rock in rocks:
        for i in range(rock.shape[0] - 1):
            start = rock[i]
            end = rock[i + 1]
            coordinates = get_coordinates_from_line(start, end)
            grid[coordinates[:, 0], coordinates[:, 1]] = 1

    print(grid[:, min_col:])

    if use_floor:
        floor = np.zeros((2, grid.shape[1]), dtype=int)
        floor[1, :] = 1
        grid = np.concatenate([grid, floor])

    return grid


def drop(
    grain: Tuple[int, int], grid: np.ndarray, use_floor: bool
) -> Tuple[bool, np.ndarray]:
    is_moving = True

    while is_moving:
        row, col = grain
        rows_below = grid[row + 1 :, col]

        if (rows_below == 0).all():  # Only empty rows below -> falls into the abyss
            return True, grid  # Falls into the abyss

        occupied_idx = np.argmax(rows_below)  # Get first non zero index

        if occupied_idx > 0:  # There are empty rows below -> move the grain downwards
            grain = (row + occupied_idx, col)
            continue

        # No empty rows below, look left and right
        if row + 1 >= grid.shape[0]:  # Out of bounds -> falls into the abyss
            return True, grid

        # Expand grid to the right if necessary
        if col + 1 >= grid.shape[1]:
            new_columns = np.zeros((grid.shape[0], 1), dtype=int)
            new_columns[-1, 0] = 1
            grid = np.concatenate([grid, new_columns], axis=1)

        if grid[row + 1, col - 1] == 0:
            grain = (row + 1, col - 1)
        elif grid[row + 1, col + 1] == 0:
            grain = (row + 1, col + 1)
        else:  # No space left or right -> settled
            is_moving = False
            grid[row, col] = 1

            if row == 0 and col == 500:
                return True, grid  # Entrance blocked

    return False, grid


def get_num_grains(rocks: List[np.ndarray], use_floor: bool):
    grid = create_grid(rocks, use_floor)
    is_full = False
    grain_count = 0

    while not is_full:
        grain = (0, 500)
        grain_count += 1
        is_full, grid = drop(grain, grid, use_floor)

    if not use_floor:  # Last grain fell into the abyss
        grain_count -= 1

    return grain_count


def main():
    rocks = read()
    num_grains = get_num_grains(rocks, use_floor=False)
    print(f"Num grains settled before falling into the abyss: {num_grains}")

    num_grains = get_num_grains(rocks, use_floor=True)
    print(f"Num grains settled before blocking the entrance: {num_grains}")


if __name__ == "__main__":
    main()
