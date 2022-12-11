from typing import Set, Tuple, List


def parse_range(range_: str) -> Tuple[int, int]:
    lower, upper = list(map(int, range_.split("-")))
    return lower, upper


def get_set_from_range(range_: Tuple[int, int]) -> Set[int]:
    return set(range(range_[0], range_[1] + 1))


def read() -> List[Tuple[Set[int], Set[int]]]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        pairs = []

        for line in lines:
            fst, snd = line.split(",")
            fst = get_set_from_range(parse_range(fst))
            snd = get_set_from_range(parse_range(snd))
            pairs.append((fst, snd))

        return pairs


def get_num_pair_subsets(pairs: List[Tuple[Set[int], Set[int]]]) -> int:
    num_subsets = 0

    for fst, snd in pairs:
        if fst.issubset(snd) or snd.issubset(fst):
            num_subsets += 1

    return num_subsets


def get_num_non_empty_pair_intersections(pairs: List[Tuple[Set[int], Set[int]]]) -> int:
    num_non_empty_intersections = 0

    for fst, snd in pairs:
        if fst.intersection(snd):
            num_non_empty_intersections += 1

    return num_non_empty_intersections


def main():
    pairs = read()

    num_pair_subsets = get_num_pair_subsets(pairs)
    print(f"Num pair subsets: {num_pair_subsets}")

    num_pair_intersections = get_num_non_empty_pair_intersections(pairs)
    print(f"Num pair intersections: {num_pair_intersections}")


if __name__ == "__main__":
    main()
