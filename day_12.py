import heapq
from typing import Tuple, Dict


class Node:

    def __init__(self, id_: Tuple[int, int], height: int):
        self.id_ = id_
        self.height = height
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def __str__(self) -> str:
        i, j = self.id_
        return f"[{i},{j}]"

    def __eq__(self, other):
        return True

    def __lt__(self, other):
        return True

    def __hash__(self):
        return self.id_


def read() -> Tuple[Dict[int, Node], Node, Node]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        graph = {}
        start_index = None
        end_index = None

        for i, line in enumerate(lines):
            for j, c in enumerate(line):
                if c == "S":
                    height = char_to_height("a")
                    start_index = (i, j)
                elif c == "E":
                    height = char_to_height("z")
                    end_index = (i, j)
                else:
                    height = char_to_height(c)

                print(f"{height},", end="")

                graph[(i, j)] = Node((i, j), height)
            print("")

        print(f"Num nodes parsed: {len(graph)}")

        for (row, col), node in graph.items():
            up_idx = (row + 1, col)
            down_idx = (row - 1, col)
            left_idx = (row, col - 1)
            right_idx = (row, col + 1)

            for neighbor_idx in [up_idx, down_idx, left_idx, right_idx]:
                row, col = neighbor_idx
                if row < 0 or col < 0 or row >= len(lines) or col >= len(lines[0]):
                    continue

                if neighbor_idx in graph:
                    neighbor = graph[neighbor_idx]

                    if neighbor.height <= node.height + 1:
                        node.add_neighbor(neighbor)

        return graph, graph[start_index], graph[end_index]


def char_to_height(char: str) -> int:
    return ord(char) - ord("a")


def get_length_of_shortest_path(start: Node, end: Node) -> int:
    queue = []
    heapq.heappush(queue, (0, start))
    explored = set()

    while len(queue) > 0:
        distance, node = heapq.heappop(queue)

        if node.id_ in explored:
            continue

        explored.add(node.id_)

        if node.id_ == end.id_:
            return distance

        for neighbor in node.neighbors:
            if neighbor.id_ not in explored and neighbor.id_:
                heapq.heappush(queue, (distance + 1, neighbor))

    raise ValueError("No path was found!")


def main():
    graph, start, end = read()
    distance = get_length_of_shortest_path(start, end)
    print(f"Min distance from start: {distance}")

    min_distance = float("inf")

    for node in graph.values():
        if node.height == 0:
            try:
                distance = get_length_of_shortest_path(node, end)

                if distance < min_distance:
                    min_distance = distance
            except ValueError:
                pass

    print(f"Global min distance: {min_distance}")


if __name__ == "__main__":
    main()
