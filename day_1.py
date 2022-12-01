from typing import List
NEW_LINE = "\n"


def get_calories_per_elf(lines: List[str]):
    calories_per_elf = []
    current_calories = 0

    for line in lines:
        line = line.strip(NEW_LINE)

        if line:
            current_calories += int(line)
        else:
            calories_per_elf.append(current_calories)
            current_calories = 0

    return calories_per_elf


def main():
    with open("input.txt") as file:
        lines = file.readlines()
        calories_per_elf = get_calories_per_elf(lines)
        calories_per_elf = sorted(calories_per_elf)
        print(calories_per_elf[-1])
        print(sum(calories_per_elf[-3:]))


if __name__ == "__main__":
    main()