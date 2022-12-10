from typing import List, Tuple

import numpy as np


def read() -> List[Tuple[str, int]]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        instructions = []

        for line in lines:
            direction, units = line.split(" ")
            units = int(units)
            assert direction in {"U", "D", "L", "R"}
            instructions.append((direction, units))

        return instructions


def get_num_tail_positions(instructions: List[Tuple[str, int]], num_knots: int) -> int:
    knots = np.zeros((num_knots, 2))
    positions = set()

    for direction, units in instructions:
        for _ in range(units):
            head = knots[0]
            tail = knots[1]

            head = _move_head(direction, head)
            knots[0] = head

            if not _is_touching(head, tail):
                knots = _move_tail(knots)

            positions.add(tuple(knots[-1]))

    return len(positions)


def _move_head(direction: str, head: np.ndarray) -> np.ndarray:
    vertical = np.array([0, 1])
    horizontal = np.array([1, 0])

    if direction == "U":
        head += vertical
    elif direction == "D":
        head -= vertical
    elif direction == "R":
        head += horizontal
    elif direction == "L":
        head -= horizontal

    return head


def _is_touching(head: np.ndarray, tail: np.ndarray) -> bool:
    for row in range(3):
        for col in range(3):
            to_check = np.array([head[0] + col - 1, head[1] + row - 1])

            if (to_check == tail).all():
                return True

    return False


def _move_tail(knots: np.ndarray) -> np.ndarray:
    for i in range(knots.shape[0] - 1):
        head = knots[i]
        tail = knots[i + 1]

        if _is_touching(head, tail):
            break

        x_distance = head[0] - tail[0]
        y_distance = head[1] - tail[1]
        x_distance = 1 if x_distance > 0 else - 1
        y_distance = 1 if y_distance > 0 else - 1

        if head[0] == tail[0]:  # same x coordinate -> same column
            tail = tail + np.array([0, y_distance])
        elif head[1] == tail[1]:  # same y coordinate -> same row
            tail = tail + np.array([x_distance, 0])
        else:  # move diagonally
            tail = tail + np.array([x_distance, y_distance])

        knots[i + 1] = tail

    return knots


def main():
    instructions = read()
    num_knots = 2
    num_positions = get_num_tail_positions(instructions, num_knots)
    print(f"Num positions for {num_knots} knots: {num_positions}")

    num_knots = 10
    num_positions = get_num_tail_positions(instructions, num_knots)
    print(f"Num positions for {num_knots} knots: {num_positions}")


if __name__ == "__main__":
    main()
