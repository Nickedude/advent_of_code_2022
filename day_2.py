from typing import Tuple, List


ROCK = "rock"
PAPER = "paper"
SCISSORS = "scissors"

PARSE_OPPONENT_MOVE = {"A": ROCK, "B": PAPER, "C": SCISSORS}

PARSE_YOUR_MOVE = {"X": ROCK, "Y": PAPER, "Z": SCISSORS}


POINTS_PER_MOVE = {
    ROCK: 1,
    PAPER: 2,
    SCISSORS: 3,
}

WINNER_MOVE = {
    ROCK: PAPER,  # rock loses to paper
    PAPER: SCISSORS,  # paper loses to scissors
    SCISSORS: ROCK,  # scissors lose to rock
}

LOSER_MOVE = {v: k for k, v in WINNER_MOVE.items()}


def read_strategy() -> List[Tuple[str, str]]:
    with open("input.txt") as file:
        strategy = []

        for line in file.readlines():
            line = line.strip("\n")
            fst, snd = line.split(" ")
            strategy.append((fst, snd))

        return strategy


def get_points(opponent_move: str, your_move: str) -> int:
    if opponent_move == your_move:  # Draw ends with three points
        return 3

    if WINNER_MOVE[opponent_move] == your_move:  # Winning earns you six points
        return 6

    return 0  # Losing earns you 0 points


def get_move(opponent_move: str, outcome: str) -> str:
    if outcome == "X":  # lose
        return LOSER_MOVE[opponent_move]
    if outcome == "Y":  # draw
        return opponent_move

    return WINNER_MOVE[opponent_move]  # win


def get_num_points_using_outcomes(
    strategy: List[Tuple[str, str]], parse_move: bool = True
) -> int:
    total_num_points = 0

    for opponent_move, snd in strategy:
        opponent_move = PARSE_OPPONENT_MOVE[opponent_move]

        if parse_move:
            your_move = PARSE_YOUR_MOVE[snd]
        else:
            your_move = get_move(opponent_move, snd)

        total_num_points += POINTS_PER_MOVE[your_move] + get_points(
            opponent_move, your_move
        )

    return total_num_points


def main():
    strategy = read_strategy()
    print(get_num_points_using_outcomes(strategy))
    print(get_num_points_using_outcomes(strategy, parse_move=False))


if __name__ == "__main__":
    main()
