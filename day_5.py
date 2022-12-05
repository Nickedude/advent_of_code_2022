from copy import deepcopy
from typing import List, Tuple, Dict


def parse_configuration(lines: List[str], num_stacks: int) -> Dict[int, List[str]]:
    crate_configuration = {i + 1: [] for i in range(num_stacks)}
    num_chars_per_crate = 4
    expected_num_chars = num_chars_per_crate * num_stacks - 1
    for line in lines[::-1]:  # Go from bottom to top to build stacks in correct order
        assert len(line) == expected_num_chars

        for i in range(num_stacks):
            lower = i * num_chars_per_crate
            upper = lower + num_chars_per_crate
            crate = line[lower:upper]
            crate = crate.strip(" ")

            if not crate:
                continue

            assert crate[0] == "["
            assert crate[2] == "]"
            crate_id = crate[1]
            crate_configuration[i + 1].append(crate_id)

    return crate_configuration


def parse_moves(lines: List[str]) -> List[Tuple[int, int, int]]:
    moves = []

    for line in lines:
        parts = line.split(" ")
        assert parts[0] == "move"
        num = parts[1]
        assert parts[2] == "from"
        src = parts[3]
        assert parts[4] == "to"
        dest = parts[5]
        move = num, src, dest
        move = tuple(map(int, move))
        moves.append(move)

    return moves


def read():
    with open("input.txt") as file:
        lines = file.readlines()
        idx = -1

        lines = list(map(lambda s: s.strip("\n"), lines))

        # Find line separating moves from configuration
        for idx, line in enumerate(lines):
            if line == "":
                break

        # Get number of stacks
        num_stacks = [num for num in lines[idx - 1].split(" ") if num]
        num_stacks = int(num_stacks[-1])

        crate_configuration = parse_configuration(lines[:idx - 1], num_stacks)
        moves = parse_moves(lines[idx + 1:])
        return crate_configuration, moves, num_stacks


def move_crates(crate_configuration: Dict[int, List[str]], moves: List[Tuple[int, int, int]]) -> Dict[int, List[str]]:
    crate_configuration = deepcopy(crate_configuration)

    for num, src, dst in moves:
        for i in range(num):
            crate = crate_configuration[src].pop(-1)
            crate_configuration[dst].append(crate)

    return crate_configuration


def move_stacks(crate_configuration: Dict[int, List[str]], moves: List[Tuple[int, int, int]]) -> Dict[int, List[str]]:
    crate_configuration = deepcopy(crate_configuration)

    for num, src, dst in moves:
        crates_to_move = crate_configuration[src][-num:]
        crate_configuration[src] = crate_configuration[src][:-num]
        crate_configuration[dst].extend(crates_to_move)

    return crate_configuration


def print_solution(solution: Dict[int, List[str]], num_stacks: int):
    print("".join(solution[i][-1] for i in range(1, num_stacks+1)))


def main():
    crate_configuration, moves, num_stacks = read()

    solution = move_crates(crate_configuration, moves)
    print_solution(solution, num_stacks)

    solution = move_stacks(crate_configuration, moves)
    print_solution(solution, num_stacks)


if __name__ == "__main__":
    main()
