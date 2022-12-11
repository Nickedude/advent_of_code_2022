import numpy as np


def read():
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))

        tree_map = []

        for line in lines:
            tree_map.append(np.array([int(c) for c in line]))

        return np.array(tree_map)


def get_num_visible_trees(tree_map: np.ndarray) -> int:
    num_visible = (
        tree_map.shape[0] * 2 + tree_map.shape[1] * 2 - 4
    )  # All trees on the edges

    for i in range(1, tree_map.shape[0] - 1):
        for j in range(1, tree_map.shape[1] - 1):
            height = tree_map[i, j]

            up = (tree_map[:i, j] < height).all()
            down = (tree_map[i + 1 :, j] < height).all()

            left = (tree_map[i, :j] < height).all()
            right = (tree_map[i, j + 1 :] < height).all()

            if up or down or left or right:
                num_visible += 1

    return num_visible


def get_viewing_distance(tree_heights: np.ndarray, max_height: int) -> int:
    blocked = tree_heights >= max_height

    if not blocked.any():
        return len(tree_heights)

    blocked_idx = np.argmax(blocked)
    return blocked_idx + 1


def get_scenic_score(tree_map: np.ndarray, row: int, col: int):
    height = tree_map[row, col]
    up = get_viewing_distance(tree_map[:row, col][::-1], height)
    down = get_viewing_distance(tree_map[row + 1 :, col], height)

    left = get_viewing_distance(tree_map[row, :col][::-1], height)
    right = get_viewing_distance(tree_map[row, col + 1 :], height)
    return up * down * left * right


def get_max_scenic_score(tree_map: np.ndarray) -> int:
    max_score = 0

    for i in range(tree_map.shape[0]):
        for j in range(tree_map.shape[1]):
            score = get_scenic_score(tree_map, i, j)

            if score > max_score:
                max_score = score

    return max_score


def main():
    tree_map = read()
    print(f"Num visible trees: {get_num_visible_trees(tree_map)}")

    max_scenic_score = get_max_scenic_score(tree_map)
    print(f"Max scenic score: {max_scenic_score}")


if __name__ == "__main__":
    main()
