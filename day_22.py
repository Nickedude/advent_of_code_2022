from collections import defaultdict
from typing import Tuple, Dict, List, Union, Set, Optional

from tqdm import tqdm

FACING_TO_INT = {"right": 0, "down": 1, "left": 2, "up": 3}
INT_TO_FACING = {v: k for k, v in FACING_TO_INT.items()}
DIRECTIONS = list(FACING_TO_INT.keys())

RECIPROCAL_TO_FACING = {"up": "down", "down": "up", "left": "right", "right": "left"}

VERTICAL = ["up", "down"]
HORIZONTAL = ["left", "right"]

FACING_TO_CHAR = {0: "→", 1: "↓", 2: "←", 3: "↑"}


TwoDimCoordinate = Tuple[int, int]
ThreeDimCoordinate = Tuple[int, int, int]


class CubeSide:
    def __init__(
        self, id_: int, row_start: int, col_start: int, row_end: int, col_end: int
    ):
        self.id_ = id_
        self.row_start = row_start
        self.col_start = col_start

        self.row_end = row_end
        self.col_end = col_end

        # Neighboring sides
        self.up = None
        self.down = None
        self.left = None
        self.right = None

        self.nodes = []

    def __str__(self):
        return str(self.id_)

    def __repr__(self):
        return self.__str__()

    @property
    def top_left(self) -> TwoDimCoordinate:
        return self.row_start, self.col_start

    @property
    def top_right(self) -> TwoDimCoordinate:
        return self.row_start, self.col_end

    @property
    def bot_left(self) -> TwoDimCoordinate:
        return self.row_end, self.col_start

    @property
    def bot_right(self) -> TwoDimCoordinate:
        return self.row_end, self.col_end

    @property
    def corners(self) -> Set[TwoDimCoordinate]:
        return {self.top_left, self.top_right, self.bot_left, self.bot_right}

    def get_neighbor_direction(self, other):
        for direction in DIRECTIONS:
            if getattr(self, direction).id_ == other.id_:
                return direction

        raise ValueError(f"Couldn't get direction of neighbor {other.id_}")


class Cube:
    def __init__(self, sides: List[CubeSide]):
        self.sides = sides
        self.id_to_side = {side.id_: side for side in sides}

    def is_cube(self) -> bool:
        for side in self.sides:
            for direction in DIRECTIONS:
                if getattr(side, direction) is None:
                    return False

        # All sides are connected at all joints!
        return True

    def __str__(self):
        string = ""

        for s in self.sides:
            string += f"\n{s.id_}: "
            for direction in DIRECTIONS:
                string += f"\t{direction} -> {getattr(s, direction)}"

        return string

    def __repr__(self):
        return self.__str__()


class Node:
    def __init__(self, row: int, col: int, is_wall: bool):
        self.row = row
        self.col = col
        self.is_wall = is_wall

        # Neighbors
        self.up = None
        self.down = None
        self.left = None
        self.right = None

        self.three_dim_coordinate = None

    def __str__(self):
        return "#" if self.is_wall else "."

    def __repr__(self):
        return self.__str__()


Graph = Dict[TwoDimCoordinate, Node]


def read(
    use_cube: bool,
) -> Tuple[Graph, List[Union[int, str]], Cube, Dict[TwoDimCoordinate, CubeSide]]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        graph = parse_nodes(lines)

        if not use_cube:
            cube = None
            coordinate_to_side = None
            graph = link_nodes(lines, graph)
        else:
            sides, coordinate_to_side = parse_sides(lines, graph)
            sides = link_sides(sides)

            cube = Cube(sides)
            cube = form_cube(cube, graph, debug=False)

            graph = link_nodes_using_sides(graph, coordinate_to_side)

        instructions = parse_instructions(lines[-1])

        return graph, instructions, cube, coordinate_to_side


def parse_nodes(lines: List[str]) -> Graph:
    graph = {}

    # Create all nodes without any linked neighbors
    for row in range(len(lines)):
        if lines[row] == "":  # Finished parsing the graph
            break

        for col in range(len(lines[row])):
            char = lines[row][col]

            if char == " ":
                continue
            elif char == "#":
                graph[(row, col)] = Node(row, col, is_wall=True)
            elif char == ".":
                graph[(row, col)] = Node(row, col, is_wall=False)
            else:
                raise ValueError(f"Unexpected character: {char}")

    return graph


def link_nodes(lines: List[str], graph: Graph) -> Graph:
    """Link each node to its neighbors using wrap-around."""
    height = len(lines) - 1

    for (row, col), node in graph.items():
        width = len(lines[row])
        right_col = (col + 1) % width

        while is_empty(lines, row, right_col):
            right_col = (right_col + 1) % width

        left_col = width - 1 if col == 0 else col - 1

        while is_empty(lines, row, left_col):
            left_col = width - 1 if left_col == 0 else left_col - 1

        up_row = height - 1 if row == 0 else row - 1

        while col >= len(lines[up_row]) or is_empty(lines, up_row, col):
            up_row = height - 1 if up_row == 0 else up_row - 1

        down_row = (row + 1) % height

        while col >= len(lines[down_row]) or is_empty(lines, down_row, col):
            down_row = (down_row + 1) % height

        node.up = graph[(up_row, col)]
        node.down = graph[(down_row, col)]
        node.left = graph[(row, left_col)]
        node.right = graph[(row, right_col)]

    return graph


def is_empty(lines: List[str], row: int, col: int) -> bool:
    return lines[row][col] == " "


def parse_sides(
    lines: List[str], graph: Graph
) -> Tuple[List[CubeSide], Dict[TwoDimCoordinate, CubeSide]]:
    size = 50
    sides = []
    coordinate_to_side = {}
    current = 0
    num_sides = 6

    for row in range(len(lines)):
        if lines[row] == "":
            break

        for col in range(len(lines[row])):
            if is_empty(lines, row, col):
                continue

            if (row, col) in coordinate_to_side:
                continue

            side = CubeSide(current, row, col, row + size - 1, col + size - 1)
            sides.append(side)

            for i in range(row, row + size):
                for j in range(col, col + size):
                    assert not is_empty(lines, i, j)
                    coordinate_to_side[(i, j)] = side
                    side.nodes.append(graph[(i, j)])

            current += 1

    assert len(set(coordinate_to_side.values())) == num_sides

    return sides, coordinate_to_side


def link_sides(sides: List[CubeSide]) -> List[CubeSide]:
    for side in sides:
        for direction in DIRECTIONS:
            if getattr(side, direction) is None:
                neighbor = get_neighboring_side(side, sides, direction)

                if neighbor is not None:
                    setattr(side, direction, neighbor)

    return sides


def get_neighboring_side(
    side: CubeSide, sides: List[CubeSide], direction: str
) -> Optional[CubeSide]:
    coordinates = set()

    if direction == "right":
        for corner in ["top_right", "bot_right"]:
            row, col = getattr(side, corner)
            coordinates.add((row, col + 1))

    elif direction == "left":
        for corner in ["top_left", "bot_left"]:
            row, col = getattr(side, corner)
            coordinates.add((row, col - 1))

    elif direction == "up":
        for corner in ["top_left", "top_right"]:
            row, col = getattr(side, corner)
            coordinates.add((row - 1, col))

    elif direction == "down":
        for corner in ["bot_left", "bot_right"]:
            row, col = getattr(side, corner)
            coordinates.add((row + 1, col))

    for neighbor in sides:
        if side.id_ == neighbor.id_:
            continue

        if coordinates.issubset(neighbor.corners):
            return neighbor

    return None


def form_cube(cube: Cube, graph: Graph, debug: bool) -> Cube:
    if False:
        _set_connection(cube, 0, "left", 2)
        _set_connection(cube, 2, "up", 0)

        _set_connection(cube, 0, "up", 1)
        _set_connection(cube, 1, "up", 0)

        _set_connection(cube, 0, "right", 5)
        _set_connection(cube, 5, "right", 0)

        _set_connection(cube, 5, "up", 3)
        _set_connection(cube, 3, "right", 5)

        _set_connection(cube, 5, "down", 1)
        _set_connection(cube, 1, "left", 5)

        _set_connection(cube, 4, "left", 2)
        _set_connection(cube, 2, "down", 4)

        _set_connection(cube, 4, "down", 1)
        _set_connection(cube, 1, "down", 4)

        size = cube.id_to_side[0].row_end - cube.id_to_side[0].row_start + 1

        # Side 3 is the top
        top = cube.id_to_side[3]
        for node in top.nodes:
            node.three_dim_coordinate = (
                node.col - top.col_start,
                node.row - top.row_start,
                0,
            )

        # Side 2 is to the left
        side = cube.id_to_side[2]
        for node in side.nodes:
            row_offset = node.row - side.row_start
            col_offset = node.col - side.col_start
            node.three_dim_coordinate = (
                -1,
                row_offset,
                col_offset - size,
            )  # 49 -> -1, 0 -> -50

        # Side 4 is downwards
        side = cube.id_to_side[4]
        for node in side.nodes:
            row_offset = node.row - side.row_start
            col_offset = node.col - side.col_start
            node.three_dim_coordinate = (col_offset, size, -(row_offset + 1))

        # Side 0 is upwards
        side = cube.id_to_side[0]
        for node in side.nodes:
            row_offset = node.row - side.row_start
            col_offset = node.col - side.col_start
            node.three_dim_coordinate = (col_offset, -1, row_offset - size)

        # Side 5 is to the right
        side = cube.id_to_side[5]
        for node in side.nodes:
            row_offset = node.row - side.row_start
            col_offset = node.col - side.col_start
            node.three_dim_coordinate = (
                size + 1,
                size - col_offset - 1,
                -(row_offset + 1),
            )  # y: 50 - 49 - 1 -> 0, z: 49 - 50 -> -1, 0 - 50 -> -50

        # Side 1 is the bottom
        side = cube.id_to_side[1]
        for node in side.nodes:
            row_offset = node.row - side.row_start
            col_offset = node.col - side.col_start
            node.three_dim_coordinate = (
                size - col_offset - 1,
                row_offset,
                -(size + 1),
            )  # y: 50 - 49 - 1 = 0, 50 - 0 - 1 = 49

        counts = defaultdict(int)
        for node in graph.values():
            counts[node.three_dim_coordinate] += 1

        wrong = [k for k, v in counts.items() if v > 1]

        assert len(graph) == len(
            set([node.three_dim_coordinate for node in graph.values()])
        )
        return cube

    # TODO: Form the cube algorithmically :)
    _set_connection(cube, 5, "right", 4)
    _set_connection(cube, 4, "down", 5)

    _set_connection(cube, 2, "left", 3)
    _set_connection(cube, 3, "up", 2)

    _set_connection(cube, 0, "left", 3)
    _set_connection(cube, 3, "left", 0)

    _set_connection(cube, 0, "up", 5)
    _set_connection(cube, 5, "left", 0)

    _set_connection(cube, 1, "up", 5)
    _set_connection(cube, 5, "down", 1)

    _set_connection(cube, 1, "down", 2)
    _set_connection(cube, 2, "right", 1)

    _set_connection(cube, 1, "right", 4)
    _set_connection(cube, 4, "right", 1)

    size = cube.id_to_side[0].row_end - cube.id_to_side[0].row_start + 1

    # Side 4 is the top
    top = cube.id_to_side[4]
    for node in top.nodes:
        node.three_dim_coordinate = (
            node.col - top.col_start,
            node.row - top.row_start,
            0,
        )

    # Side 3 is to the left
    side = cube.id_to_side[3]
    for node in side.nodes:
        row_offset = node.row - side.row_start
        col_offset = node.col - side.col_start
        node.three_dim_coordinate = (
            -1,
            row_offset,
            col_offset - size,
        )  # 49 -> -1, 0 -> -50

    # Side 5 is downwards
    side = cube.id_to_side[5]
    for node in side.nodes:
        row_offset = node.row - side.row_start
        col_offset = node.col - side.col_start
        node.three_dim_coordinate = (row_offset, size, col_offset - size)

    # Side 2 is upwards
    side = cube.id_to_side[2]
    for node in side.nodes:
        row_offset = node.row - side.row_start
        col_offset = node.col - side.col_start
        node.three_dim_coordinate = (col_offset, -1, row_offset - size)

    # Side 1 is to the right
    side = cube.id_to_side[1]
    for node in side.nodes:
        row_offset = node.row - side.row_start
        col_offset = node.col - side.col_start
        node.three_dim_coordinate = (
            size,
            size - row_offset - 1,
            col_offset - size,
        )  # y: 50 - 49 - 1 -> 0, z: 49 - 50 -> -1, 0 - 50 -> -50

        if node.three_dim_coordinate == (48, 48, -49):
            print("k")

    # Side 0 is the bottom
    side = cube.id_to_side[0]
    for node in side.nodes:
        row_offset = node.row - side.row_start
        col_offset = node.col - side.col_start
        node.three_dim_coordinate = (
            col_offset,
            size - row_offset - 1,
            -(size + 1),
        )  # y: 50 - 49 - 1 = 0, 50 - 0 - 1 = 49

    counts = defaultdict(int)
    for node in graph.values():
        counts[node.three_dim_coordinate] += 1

    assert len(graph) == len(
        set([node.three_dim_coordinate for node in graph.values()])
    )

    return cube


def _set_connection(cube: Cube, src: int, direction: str, dst: int):
    current = getattr(cube.id_to_side[src], direction)
    assert current is None
    setattr(cube.id_to_side[src], direction, cube.id_to_side[dst])


def link_nodes_using_sides(
    graph: Graph, coordinate_to_side: Dict[TwoDimCoordinate, CubeSide]
) -> Graph:
    """Link each node to its neighbors within the current cube side."""

    for (row, col), node in graph.items():
        up_row = row - 1
        down_row = row + 1
        left_col = col - 1
        right_col = col + 1

        current_side = coordinate_to_side[(row, col)]

        if current_side.col_start <= right_col <= current_side.col_end:
            node.right = graph[(row, right_col)]

        if current_side.col_start <= left_col <= current_side.col_end:
            node.left = graph[(row, left_col)]

        if current_side.row_start <= up_row <= current_side.row_end:
            node.up = graph[(up_row, col)]

        if current_side.row_start <= down_row <= current_side.row_end:
            node.down = graph[(down_row, col)]

    return graph


def parse_instructions(line: str) -> List[Union[int, str]]:
    instructions = []
    prev = -1
    curr = 0

    while curr < len(line):
        if line[curr] in {"R", "L"}:
            num = int(line[prev + 1 : curr])
            prev = curr
            instructions.append(num)
            instructions.append(line[curr])

        curr += 1

    if line[-1] not in {"R", "L"}:
        num = int(line[prev + 1 :])
        instructions.append(num)

    return instructions


def graph_to_str(graph: Graph, path: List[Tuple[int, int, int]] = None):
    coordinates = list(graph.keys())
    height = max([row for (row, col) in coordinates])
    width = max([col for (row, col) in coordinates])
    graph_as_str = ""

    for i in range(height + 1):
        for j in range(width + 1):
            if (i, j) not in graph:
                graph_as_str += " "
                continue

            node = graph[(i, j)]

            if node.is_wall:
                graph_as_str += "#"
                continue

            if path is not None:
                in_path = False

                for (row, col, facing) in path[::-1]:
                    if row == i and col == j:
                        graph_as_str += FACING_TO_CHAR[facing]
                        in_path = True
                        break

                if in_path:
                    continue

            graph_as_str += "."

        graph_as_str += "\n"

    return graph_as_str


def test_top(graph_3d, coordinate_to_side):
    node = graph_3d[(0, 0, 0)]

    facing = FACING_TO_INT["left"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (-1, 0, -1)
    assert (neighbor.row, neighbor.col) == (node.row, node.col - 1), (neighbor.row, neighbor.col)

    facing = FACING_TO_INT["up"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (0, -1, -1)
    assert (neighbor.row, neighbor.col) == (node.row - 1, node.col), (neighbor.row, neighbor.col)

    node = graph_3d[(0, 49, 0)]
    facing = FACING_TO_INT["down"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (0, 50, -1)
    assert (neighbor.row, neighbor.col) == (node.row + 1, node.col - 1), (neighbor.row, neighbor.col)

    node = graph_3d[(0, 49, 0)]
    facing = FACING_TO_INT["left"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (-1, 49, -1)
    assert (neighbor.row, neighbor.col) == (node.row, node.col - 1), (neighbor.row, neighbor.col)

    node = graph_3d[(49, 49, 0)]
    facing = FACING_TO_INT["right"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (50, 49, -1), neighbor.three_dim_coordinate
    assert (neighbor.row, neighbor.col) == (0, 149), (neighbor.row, neighbor.col)


def test_bot(graph, coordinate_to_side):
    # Bottom
    node = graph[(0, 50)]
    assert node.three_dim_coordinate == (0, 49, -51), node.three_dim_coordinate
    facing = FACING_TO_INT["left"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (-1, 49, -50), neighbor.three_dim_coordinate
    assert (neighbor.row, neighbor.col) == (149, 0), (neighbor.row, neighbor.col)

    facing = FACING_TO_INT["up"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (0, 50, -50), neighbor.three_dim_coordinate
    assert (neighbor.row, neighbor.col) == (150, 0), (neighbor.row, neighbor.col)
    assert neighbor.is_wall

    node = graph[(0, 99)]
    assert node.three_dim_coordinate == (49, 49, -51), node.three_dim_coordinate
    facing = FACING_TO_INT["right"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (50, 49, -50), neighbor.three_dim_coordinate
    assert (neighbor.row, neighbor.col) == (0, 100), (neighbor.row, neighbor.col)

    node = graph[(49, 50)]
    assert node.three_dim_coordinate == (0, 0, -51), node.three_dim_coordinate
    facing = FACING_TO_INT["left"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (-1, 0, -50), neighbor.three_dim_coordinate
    assert (neighbor.row, neighbor.col) == (100, 0), (neighbor.row, neighbor.col)

    facing = FACING_TO_INT["down"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (0, -1, -50), neighbor.three_dim_coordinate
    assert (neighbor.row, neighbor.col) == (50, 50), (neighbor.row, neighbor.col)

    node = graph[(49, 99)]
    assert node.left is not None
    assert node.three_dim_coordinate == (49, 0, -51), node.three_dim_coordinate
    facing = FACING_TO_INT["right"]
    neighbor, new_facing = get_cube_neighbor(
        node, INT_TO_FACING[facing], coordinate_to_side
    )
    assert neighbor.three_dim_coordinate == (50, 0, -50), neighbor.three_dim_coordinate
    assert (neighbor.row, neighbor.col) == (49, 100), (neighbor.row, neighbor.col)


def test(graph: Graph, coordinate_to_side: Dict[TwoDimCoordinate, CubeSide]):
    graph_3d = {node.three_dim_coordinate: node for node in graph.values()}
    test_top(graph_3d, coordinate_to_side)
    test_bot(graph, coordinate_to_side)


def move(
    graph: Graph,
    instructions: List[Union[int, str]],
    coordinate_to_side: Dict[TwoDimCoordinate, CubeSide],
) -> List[Tuple[int, int, int]]:
    start = sorted(list(graph.keys()))[0]
    print(f"Starting at {start}")
    node = graph[start]
    facing = FACING_TO_INT["right"]

    path = [(start[0], start[1], facing)]

    for p, inst in enumerate(instructions):
        if isinstance(inst, int):
            while inst > 0:
                neighbor = getattr(node, INT_TO_FACING[facing])

                # We're crossing a side of the cube:
                if coordinate_to_side is not None and neighbor is None:
                    neighbor, new_facing = get_cube_neighbor(
                        node, INT_TO_FACING[facing], coordinate_to_side
                    )
                else:
                    new_facing = facing

                assert neighbor is not None, "Neighbor is None!"

                if neighbor.is_wall:
                    break

                facing = new_facing
                node = neighbor
                inst -= 1
                path.append((neighbor.row, neighbor.col, facing))

        elif isinstance(inst, str):
            if inst == "R":
                facing = (facing + 1) % len(FACING_TO_INT)
            elif inst == "L":
                facing = facing - 1
                facing = len(FACING_TO_INT) - 1 if facing < 0 else facing
            else:
                raise ValueError(f"Unrecognized instruction: {inst}")

            path.append((node.row, node.col, facing))

    return path


def manhattan_distance(fst: Node, snd: Node) -> int:
    dist = 0

    for i in range(3):
        dist += abs(fst.three_dim_coordinate[i] - snd.three_dim_coordinate[i])

    return dist


def get_cube_neighbor(
    node: Node,
    facing: str,
    coordinate_to_side: Dict[TwoDimCoordinate, CubeSide],
) -> Tuple[Node, int]:
    current_side = coordinate_to_side[(node.row, node.col)]
    neighbor_side = getattr(current_side, facing)

    reciprocal = neighbor_side.get_neighbor_direction(current_side)
    new_facing = FACING_TO_INT[RECIPROCAL_TO_FACING[reciprocal]]

    closest = None
    min_dist = float("inf")

    # Find closest coordinate on the neighboring side
    for candidate_node in neighbor_side.nodes:
        dist = manhattan_distance(node, candidate_node)

        if dist < min_dist:
            min_dist = dist
            closest = candidate_node

    return closest, new_facing


def get_password(row: int, col: int, facing: int) -> int:
    return 1000 * (row + 1) + 4 * (col + 1) + facing


def solve(use_cube: bool):
    graph, instructions, cube, coordinate_to_side = read(use_cube)
    print(f"Parsed cube: {cube}")

    print("Parsed graph:")
    print(graph_to_str(graph))
    print("")

    test(graph, coordinate_to_side)
    instructions = instructions[:5000]
    path = move(graph, instructions, coordinate_to_side)

    row, col, facing = path[-1]
    print(graph_to_str(graph, path[-10:]))

    print("Instructions: ", end="")
    for i in instructions:
        print(i, end=" ")

    print("")

    print(f"Final state: row={row}, col={col}, facing={facing}")
    print(f"Password: {get_password(row, col, facing)}")


def main():
    # solve(fold=False)
    solve(use_cube=True)


if __name__ == "__main__":
    main()
