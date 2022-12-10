from typing import Tuple, List, Optional

WIDTH = 40
HEIGHT = 6


def read() -> List[Tuple[str, Optional[int]]]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        instructions = []

        for line in lines:
            instruction = line[:4]

            if len(line) > 4:
                value = line[4:].strip(" ")
                value = int(value)
            else:
                value = 0

            instructions.append((instruction, value))

        return instructions


def run_program(instructions: List[Tuple[str, int]]) -> int:
    x = 1
    cycle = 0

    signal_strength = 0
    num_cycles_until_report = 20
    num_cycles_between_reports = 40

    for instruction, value in instructions:
        if instruction == "addx":
            num_cycles_to_run = 2
        elif instruction == "noop":
            num_cycles_to_run = 1
        else:
            raise ValueError(f"Unexpected instruction: {instruction}")

        for i in range(num_cycles_to_run):
            cycle += 1
            num_cycles_until_report -= 1

            position = (cycle % WIDTH) - 1

            if position == 0:
                print("")

            if (x - 1) <= position <= (x + 1):
                print("#", end="")
            else:
                print(".", end="")

            if num_cycles_until_report == 0:
                num_cycles_until_report = num_cycles_between_reports
                signal_strength += cycle * x

        x += value

    return signal_strength


def main():
    instructions = read()
    signal_strength = run_program(instructions)
    print(f"\nSignal strength: {signal_strength}")


if __name__ == "__main__":
    main()
