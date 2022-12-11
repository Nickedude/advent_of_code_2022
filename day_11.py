from functools import partial
from operator import mul, add
from typing import List, Callable, Dict, Union


class BigNumber:
    """Represents each number as the remainder for each divisor."""

    def __init__(self):
        self._remainders = {}

    def remainder(self, divisor: int) -> int:
        return self._remainders[divisor]

    @property
    def divisors(self) -> List[int]:
        return list(self._remainders.keys())

    @classmethod
    def from_integer(cls, x: int, divisors: List[int]):
        big_number = cls()

        if x is not None:
            for divisor in divisors:
                remainder = x % divisor
                big_number._remainders[divisor] = remainder

        return big_number

    @classmethod
    def from_remainders(cls, remainders: Dict[int, int]):
        big_number = cls()
        big_number._remainders = remainders
        return big_number

    def __mul__(self, other):
        remainders = {}

        for divisor in self.divisors:
            x_remainder = self.remainder(divisor)
            y_remainder = other.remainder(divisor)
            remainder = x_remainder * y_remainder
            remainders[divisor] = remainder % divisor

        return BigNumber.from_remainders(remainders)

    def __add__(self, other):
        remainders = {}

        for divisor in self.divisors:
            x_remainder = self.remainder(divisor)
            y_remainder = other.remainder(divisor)
            remainder = x_remainder + y_remainder
            remainders[divisor] = remainder % divisor

        return BigNumber.from_remainders(remainders)

    def __floordiv__(self, other):
        raise NotImplementedError()


class Test:
    def __init__(self, operand: int, true_id: int, false_id: int):
        self._operand = operand
        self._true_id = true_id
        self._false_id = false_id

    @property
    def operand(self) -> int:
        return self._operand

    def run(self, x: Union[int, BigNumber]) -> int:
        if isinstance(x, int):
            remainder = x % self._operand
        else:
            remainder = x.remainder(self._operand)

        if remainder == 0:
            return self._true_id
        else:
            return self._false_id


class Monkey:
    def __init__(
        self, id_: int, items: List[BigNumber], operation: Callable, test: Test
    ):
        self.id_ = id_
        self.items = items
        self._operation = operation
        self.test = test
        self.num_inspections = 0

    def inspect(self, x: BigNumber) -> BigNumber:
        self.num_inspections += 1
        return self._operation(x)


def read(use_remainders: bool):
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        lines = list(filter(lambda s: len(s) > 0, lines))
        num_lines_per_monkey = 6

        divisors = [3]  # Need to be able to divide by thee to solve first task

        for i in range(0, len(lines), num_lines_per_monkey):
            divisors.append(
                get_operand_from_prefix(lines[i + 3], prefix="Test: divisible by ")
            )

        if use_remainders:
            parse_number = partial(BigNumber.from_integer, divisors=divisors)
        else:
            parse_number = int

        monkeys = []

        for i in range(0, len(lines), num_lines_per_monkey):
            id_ = parse_id(lines[i])
            starting_items = parse_starting_items(lines[i + 1], parse_number)
            operation = parse_operation(lines[i + 2], parse_number)
            test = parse_test(lines[i + 3], lines[i + 4], lines[i + 5])
            monkeys.append(Monkey(id_, starting_items, operation, test))

        return monkeys


def parse_id(id_: str) -> int:
    assert id_.startswith("Monkey ")
    id_ = id_[len("Monkey ") :]
    return int(id_.strip(":"))


def parse_starting_items(
    starting_items: str, parse_number: Callable
) -> List[BigNumber]:
    assert "Starting items: " in starting_items
    starting_items = starting_items.split(": ")[1]
    starting_items = starting_items.split(",")
    starting_items = list(map(int, starting_items))
    return [parse_number(x) for x in starting_items]


def parse_operation(operation: str, parse_number: Callable) -> Callable:
    assert "Operation: " in operation
    operation = operation.split(": ")[1]
    operation = operation.split(" = ")[1]

    operators = [("*", mul), ("+", add)]
    assert any([op_name in operation for op_name, _ in operators])

    for op_name, op in operators:
        if op_name in operation:
            left_operand, right_operand = operation.split(op_name)

            left_operand = parse_operand(left_operand, parse_number)
            right_operand = parse_operand(right_operand, parse_number)

            return lambda x: op(left_operand(x), right_operand(x))


def parse_operand(operand: str, parse_number: Callable) -> Callable:
    operand = operand.strip(" ")

    if operand == "old":
        return lambda x: x

    operand = int(operand)
    operand = parse_number(operand)
    return lambda x: operand


def parse_test(test: str, fst_outcome: str, snd_outcome: str) -> Test:
    operand = get_operand_from_prefix(test, prefix="Test: divisible by ")
    fst_id = get_operand_from_prefix(fst_outcome, prefix="If true: throw to monkey ")
    snd_id = get_operand_from_prefix(snd_outcome, prefix="If false: throw to monkey ")

    return Test(operand, fst_id, snd_id)


def get_operand_from_prefix(operand: str, prefix: str) -> int:
    operand = operand.strip(" ")
    assert prefix in operand
    return int(operand[len(prefix) :])


def turn(current_monkey: Monkey, all_monkeys: List[Monkey], divide_by_three: bool):
    if len(current_monkey.items) > 0:
        items = list(map(current_monkey.inspect, current_monkey.items))

        if divide_by_three:
            items = list(map(lambda x: x // 3, items))

        for i in items:
            next_monkey_id = current_monkey.test.run(i)
            all_monkeys[next_monkey_id].items.append(i)

        current_monkey.items = []


def round_(monkeys: List[Monkey], divide_by_three: bool):
    for monkey in monkeys:
        turn(monkey, monkeys, divide_by_three)


def get_monkey_business(monkeys: List[Monkey]) -> int:
    monkeys = sorted(monkeys, key=lambda m: m.num_inspections)
    return monkeys[-1].num_inspections * monkeys[-2].num_inspections


def chase_monkeys(num_rounds: int, divide_by_three: bool):
    monkeys = read(use_remainders=not divide_by_three)
    print("Chasing monkeys!")

    for i in range(num_rounds):
        round_(monkeys, divide_by_three)

    print(f"Monkey business: {get_monkey_business(monkeys)}")
    for j, m in enumerate(monkeys):
        print(f"Monkey {j}: {m.num_inspections}, {m.items}")
    print("")


def main():
    chase_monkeys(num_rounds=20, divide_by_three=True)
    chase_monkeys(num_rounds=10000, divide_by_three=False)


if __name__ == "__main__":
    main()
