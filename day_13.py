from typing import List, Tuple, Union


ORDERED = 1
NO_DECISION = 0
UNORDERED = -1


def read() -> List[Tuple]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        packets = []

        for i in range(0, len(lines), 3):
            fst = lines[i]
            snd = lines[i+1]
            packets.append((parse_packet(fst), parse_packet(snd)))

        return packets


def parse_packet(line: str):
    return eval(line)


def is_ordered(left: Union[List, int], right: Union[List, int]) -> int:
    if isinstance(left, int) and isinstance(right, int):
        if left < right:
            return ORDERED
        elif left > right:
            return UNORDERED
        else:
            return NO_DECISION

    if isinstance(left, int) and isinstance(right, list):
        return is_ordered([left], right)

    if isinstance(left, list) and isinstance(right, int):
        return is_ordered(left, [right])

    assert isinstance(left, list)
    assert isinstance(right, list)

    left_idx = 0
    right_idx = 0

    while left_idx < len(left) and right_idx < len(right):
        left_item = left[left_idx]
        right_item = right[right_idx]

        ordered = is_ordered(left_item, right_item)

        if ordered in {ORDERED, UNORDERED}:
            return ordered

        left_idx += 1
        right_idx += 1

    remaining_left = len(left) - left_idx
    remaining_right = len(right) - right_idx

    assert remaining_right == 0 or remaining_left == 0

    if remaining_left == 0 and remaining_right > 0:
        return ORDERED
    elif remaining_right == 0 and remaining_left > 0:
        return UNORDERED

    return NO_DECISION


def get_ordered_packet_pair_indices(packets: List[Tuple]) -> List[int]:
    indices = []

    for i, (left, right) in enumerate(packets):
        if is_ordered(left, right) == ORDERED:
            indices.append(i+1)

    return indices


def sort_packets(packets: List[Tuple]) -> List:
    # Flatten list of packets
    flat_packets = []
    for left, right in packets:
        flat_packets.extend([left, right])

    # Add divider packets
    flat_packets.append([[2]])
    flat_packets.append([[6]])

    # Bubble sort
    is_sorted = False
    while not is_sorted:
        is_sorted = True

        for i in range(len(flat_packets) - 1):
            left = flat_packets[i]
            right = flat_packets[i+1]

            if is_ordered(left, right) == UNORDERED:
                flat_packets[i] = right
                flat_packets[i+1] = left
                is_sorted = False

    return flat_packets


def get_decoder_key(packets: List) -> int:
    # Locate divider packets
    fst_idx = -1
    snd_idx = -1

    for i, packet in enumerate(packets):
        if packet == [[2]]:
            fst_idx = i + 1
        if packet == [[6]]:
            snd_idx = i + 1

    return fst_idx * snd_idx


def main():
    packets = read()

    ordered_indices = get_ordered_packet_pair_indices(packets)
    print(f"Sum of correctly ordered indices: {sum(ordered_indices)}")

    packets = sort_packets(packets)
    decoder_key = get_decoder_key(packets)
    print(f"The decoder key is: {decoder_key}")


if __name__ == "__main__":
    main()

