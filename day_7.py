from abc import ABC
from typing import List, Tuple, Optional

COMMAND_CHAR = "$"


class Directory:
    """Class representing a directory."""

    def __init__(self, name: str, parent=None):
        self._name = name
        self.parent_directory = parent

        self.files = []
        self.child_directories = []

    @property
    def name(self) -> str:
        return self._name

    def get_size(self) -> int:
        """Get the size of the directory."""
        size = 0

        for _, file_size in self.files:
            size += file_size

        for directory in self.child_directories:
            size += directory.get_size()

        return size


class Command(ABC):
    """Class representing a command."""


class ChangeDirectory(Command):
    """Class representing a cd command."""

    def __init__(self, directory: str):
        self._directory = directory

    @property
    def directory(self) -> str:
        return self._directory

    def __str__(self) -> str:
        return f"cd {self._directory}"


class ListDirectory(Command):
    """Class representing a ls command."""

    def __init__(self, files: List[Tuple[str, int]], directories: List[str]):
        self._files = files
        self._directories = directories

    @property
    def files(self) -> List[Tuple[str, int]]:
        return self._files

    @property
    def directories(self) -> List[str]:
        return self._directories

    def __str__(self) -> str:
        output = "ls \n"

        for name in self._directories:
            output += f"  directory {name}\n"

        for name, size in self._files:
            output += f"  file {name}: {size} bytes\n"

        return output


def get_directories_by_condition(root: Directory, condition: callable) -> List[Directory]:
    """Get all directories fulfilling a condition."""
    queue = [root]
    found = []

    for directory in queue:
        if condition(directory):
            found.append(directory)

        for child in directory.child_directories:
            queue.append(child)

    return found


def build_file_system(executed_commands: List[Command]) -> Directory:
    """Build the file system from the executed commands and return the root directory."""
    fst = executed_commands[0]
    assert isinstance(fst, ChangeDirectory), "First command is not a cd"
    assert fst.directory == "/", "First cd command is not to root directory"

    root = Directory(fst.directory)
    current = root

    for command in executed_commands[1:]:
        if isinstance(command, ChangeDirectory):
            if command.directory == "..":
                current = current.parent_directory
            else:
                for child in current.child_directories:
                    if child.name == command.directory:
                        current = child
                        break

        elif isinstance(command, ListDirectory):
            current.files.extend(command.files)

            for directory in command.directories:
                child = Directory(directory, parent=current)
                current.child_directories.append(child)

        else:
            raise ValueError(f"Unexpected command: {command}")

    return root


def parse(commands_and_outputs: List[Tuple[List[str], List[str]]]) -> List[Command]:
    """Parse commands and outputs."""
    executed_commands = []

    for command, output in commands_and_outputs:
        if command[0] == "cd":
            _, target_directory = command
            executed_commands.append(ChangeDirectory(target_directory))
            assert len(output) == 0, f"No output expected for cd command: {output}"
        elif command[0] == "ls":
            assert len(command) == 1, f"No parameter expected for ls command: {command}"
            files = []
            directories = []

            for line in output:
                fst, snd = line.split(" ")

                if fst == "dir":
                    directories.append(snd)
                else:
                    size = int(fst)
                    files.append((snd, size))

            executed_commands.append(ListDirectory(files, directories))
        else:
            raise ValueError(f"Unknown command: {command}")

    return executed_commands


def read() -> List[Tuple[List[str], List[str]]]:
    """Read the input and return a list of commands and outputs."""
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda l: l.strip("\n"), lines))
        command_indices = [i for i, line in enumerate(lines) if COMMAND_CHAR in line]

        commands_and_outputs = []

        for i, command_idx in enumerate(command_indices):
            command = lines[command_idx]
            command = command.strip("$ ")
            command = command.split(" ")

            output_end_idx = len(lines) if i + 1 == len(command_indices) else command_indices[i + 1]
            output = lines[command_idx+1:output_end_idx]

            commands_and_outputs.append((command, output))

        return commands_and_outputs


def main():
    commands_and_outputs = read()
    commands_and_outputs = parse(commands_and_outputs)

    for cmd in commands_and_outputs:
        print(cmd)

    root = build_file_system(commands_and_outputs)
    print(f"Total size: {root.get_size()}")

    max_size = 100000
    dirs = get_directories_by_condition(root, lambda directory: directory.get_size() <= max_size)

    print(f"Directories with size <= {max_size}:")
    for d in dirs:
        print(f"\t{d.name}")

    print("")
    print(f"Sum of these sizes are: {sum([d.get_size() for d in dirs])}")

    used_space = root.get_size()
    total_size = 70000000
    free_space = total_size - used_space
    needed_space = 30000000
    missing_space = needed_space - free_space

    print(f"Free space: {free_space}")
    print(f"Missing space: {missing_space}")

    dirs = get_directories_by_condition(root, lambda directory: directory.get_size() >= missing_space)
    dirs = sorted(dirs, key=lambda directory: directory.get_size())
    print(f"Smallest directory that could be deleted: {dirs[0].name} {dirs[0].get_size()}")


if __name__ == "__main__":
    main()
