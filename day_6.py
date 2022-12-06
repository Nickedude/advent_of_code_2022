from typing import List


def _buffer_is_full(buffer: List, message_length: int) -> bool:
    return len(buffer) == message_length


def _get_num_unique_chars(buffer: List) -> int:
    return len(set(buffer))


def find_unique_characters(data: str, message_length: int) -> int:
    buffer = []

    for i, c in enumerate(data):
        if _buffer_is_full(buffer, message_length):
            buffer.pop(0)

        buffer.append(c)
        num_unique_chars = _get_num_unique_chars(buffer)

        if _buffer_is_full(buffer, message_length) and num_unique_chars == message_length:
            return i + 1


def read() -> str:
    with open("input.txt") as file:
        lines = file.readlines()
        assert len(lines) == 1
        return lines[0]


def main():
    data = read()
    print(find_unique_characters(data, message_length=4))
    print(find_unique_characters(data, message_length=14))


if __name__ == "__main__":
    main()
