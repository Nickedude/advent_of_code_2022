from typing import Tuple, List


def read() -> List[str]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        return lines


def _get_priority(s: str) -> int:
    base_priority = 1 if s.islower() else 27
    return ord(s.lower()) - ord("a") + base_priority


def get_total_priority(rucksacks: List[Tuple[str, str]]) -> int:
    sum_of_priorities = 0

    for rucksack in rucksacks:
        compartment_size = len(rucksack) // 2
        fst = rucksack[:compartment_size]
        snd = rucksack[compartment_size:]
        assert len(fst) == len(snd)

        in_both = list(set(fst).intersection(snd))
        assert len(in_both) == 1
        sum_of_priorities += _get_priority(in_both[0])

    return sum_of_priorities


def get_total_badge_priority(rucksacks: List[Tuple[str, str]]) -> int:
    sum_of_priorities = 0
    group_size = 3

    for i in range(0, len(rucksacks), group_size):
        badge = set(rucksacks[i])

        for j in range(1, group_size):
            badge = badge.intersection(rucksacks[i + j])

        badge = list(badge)[0]
        sum_of_priorities += _get_priority(badge)

    return sum_of_priorities


def main():
    rucksacks = read()
    total_prio = get_total_priority(rucksacks)
    print(f"Total prio: {total_prio}")

    total_badge_prio = get_total_badge_priority(rucksacks)
    print(f"Total badge prio: {total_badge_prio}")


if __name__ == "__main__":
    main()
