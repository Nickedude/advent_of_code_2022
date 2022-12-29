from typing import Tuple, List, Set

from tqdm import tqdm

Coordinate = Tuple[int, int]


def read() -> Tuple[List[Coordinate], List[Coordinate]]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))

        sensors = []
        beacons = []

        for line in lines:
            sensor, beacon = line.split(":")
            sensor = sensor.strip("Sensor at ")
            beacon = beacon.strip(" closest beacon is at ")

            sensors.append(_get_coordinate(sensor))
            beacons.append(_get_coordinate(beacon))

    return sensors, beacons


def _get_coordinate(s: str) -> Coordinate:
    x, y = s.split(", ")
    x = _get_int_from_str(x, prefix="x=")
    y = _get_int_from_str(y, prefix="y=")
    return x, y


def _get_int_from_str(s: str, prefix: str) -> int:
    return int(s.split(prefix)[1])


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged = False

    while not merged:
        merged = True
        merged_ranges = list()
        merged_indices = set()

        for i, (fst_start, fst_end) in enumerate(ranges):
            if i in merged_indices:
                continue

            merged_indices.add(i)
            candidates = []

            for j, (snd_start, snd_end) in enumerate(ranges):
                if i == j or j in merged_indices:
                    continue

                if fst_start <= snd_end and snd_start <= fst_end:
                    merged_indices.add(j)
                    candidates.append((snd_start, snd_end))

            if len(candidates) > 0:
                merged = False
                candidates.append((fst_start, fst_end))
                start = min([s for s, _ in candidates])
                end = max([e for _, e in candidates])
                merged_ranges.append((start, end))
            else:
                merged_ranges.append((fst_start, fst_end))

        ranges = merged_ranges

    return ranges


def get_covered_column_ranges_at_row(
    sensors: List[Coordinate],
    beacons: List[Coordinate],
    row: int,
    col_limits: Tuple[int, int] = None,
) -> List[Tuple[int, int]]:
    column_ranges = list()

    for (s_col, s_row), (b_col, b_row) in zip(sensors, beacons):
        # All coordinates at this distance from the sensor (except for the beacon) are empty
        distance = abs(s_col - b_col) + abs(s_row - b_row)

        # Distance from the sensor to the given row
        vertical_distance = abs(s_row - row)

        if distance >= vertical_distance:
            # The distance "left" determines how many columns are "covered" by this sensor
            horizontal_distance = distance - vertical_distance

            if col_limits is None:
                min_col = s_col - horizontal_distance
                max_col = s_col + horizontal_distance
            else:
                min_col, max_col = col_limits
                min_col = max(min_col, s_col - horizontal_distance)
                max_col = min(max_col, s_col + horizontal_distance)

            column_ranges.append((min_col, max_col))

    # Merge any overlapping ranges
    column_ranges = merge_ranges(column_ranges)

    return column_ranges


def get_num_columns_without_beacons(
    column_ranges: List[Tuple[int, int]], beacons: List[Coordinate], row: int
) -> int:
    num_columns = 0

    for start, end in column_ranges:
        num_columns += end - start + 1

        # Subtract any beacons at the given row
        for b_col, b_row in set(beacons):
            if b_row == row and start <= b_col <= end:
                num_columns -= 1

    return num_columns


def get_tuning_frequency(coordinate: Coordinate) -> int:
    x, y = coordinate
    return x * 4000000 + y


def main():
    sensors, beacons = read()
    row = 2000000
    column_ranges = get_covered_column_ranges_at_row(sensors, beacons, row)
    num_columns_without_beacons = get_num_columns_without_beacons(
        column_ranges, beacons, row
    )
    print(
        f"Number of positions without a beacon on row {row}: {num_columns_without_beacons}"
    )

    max_range = 4000000
    for row in tqdm(range(max_range), desc="Searching for beacon!"):
        column_ranges = get_covered_column_ranges_at_row(
            sensors, beacons, row, col_limits=(0, max_range)
        )

        if column_ranges != [(0, max_range)]:  # If all columns are not covered
            column_ranges = sorted(column_ranges)
            assert len(column_ranges) <= 2, f"Too many gaps found: {column_ranges}"

            if column_ranges[0][0] != 0:
                col = 0
            elif column_ranges[-1][1] != max_range:
                col = max_range
            else:
                col = column_ranges[0][1] + 1

            tuning_frequency = get_tuning_frequency((col, row))
            print(f"The tuning frequency is: {tuning_frequency}")
            break


if __name__ == "__main__":
    main()
