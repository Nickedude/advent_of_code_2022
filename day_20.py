import math
from collections import defaultdict
from copy import deepcopy
from typing import List

import numpy as np


def read() -> List[int]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        return list(map(int, lines))


def mix(numbers: List[int], num_mixes: int = 1):
    size = len(numbers)
    numbers = np.array(numbers)
    indices = np.arange(size)

    for _ in range(num_mixes):
        for i in range(size):
            num = numbers[i]
            idx = indices[i]

            if num == 0:
                continue

            # Moving a full lap means ending up where we started -> subtract all full laps
            num_laps = num / (size - 1)  # Moving a full lap means passing n-1 numbers
            num_laps = math.floor(num_laps) if num_laps >= 0 else math.ceil(num_laps)
            new_idx = idx + num - num_laps * (size - 1)

            # Apply wrap-around and move one extra step
            # This is to avoid having to handle wrap-around when moving the neighbors of this number
            if new_idx < 0:
                new_idx = new_idx % -size
                new_idx = new_idx + size - 1
            elif new_idx >= size:
                new_idx = new_idx % size
                new_idx = new_idx + 1

            # Move all numbers between the old and new index one step
            distance = new_idx - idx
            direction = -1 if distance < 0 else 1

            to_move = np.arange(abs(distance))
            to_move = to_move * direction
            to_move = to_move + idx + direction  # First index to change is the neighbor of i

            _, mask, _ = np.intersect1d(indices, to_move, return_indices=True)
            indices[mask] = indices[mask] - direction
            indices[i] = new_idx

    zero_idx = indices[numbers == 0]
    groove_coordinate_sum = 0

    for i in range(1, 4):
        offset = i * 1000
        idx = (zero_idx + offset) % size
        idx = np.argmax(indices == idx)
        groove_coordinate_sum += numbers[idx]

    return groove_coordinate_sum


def main():
    numbers = read()
    groove_coordinate_sum = mix(numbers, num_mixes=1)
    print(f"Groove coordinate sum: {groove_coordinate_sum}")

    decryption_key = 811589153
    numbers = np.array(numbers) * decryption_key
    groove_coordinate_sum = mix(numbers, num_mixes=10)
    print(f"Groove coordinate sum: {groove_coordinate_sum}")


if __name__ == "__main__":
    main()
