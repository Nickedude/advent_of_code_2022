from typing import Dict, Tuple, Union

OPS = ["+", "-", "/", "*", "=="]


class Monkey:

    def __init__(self, id_: str, expression):
        self.id_ = id_
        self.expression = expression

    def eval(self, monkeys: Dict) -> int:
        if isinstance(self.expression, int):
            return self.expression

        lhs, op, rhs = self.expression
        lhs_val = monkeys[lhs].eval(monkeys)
        rhs_val = monkeys[rhs].eval(monkeys)
        return eval(f"{lhs_val} {op} {rhs_val}")

    def lazy_eval(self, monkeys: Dict) -> Union[Tuple, int, str]:
        if isinstance(self.expression, (int, str)):
            return self.expression

        lhs, op, rhs = self.expression
        lhs_val = monkeys[lhs].lazy_eval(monkeys)
        rhs_val = monkeys[rhs].lazy_eval(monkeys)

        return lhs_val, op, rhs_val


def read() -> Dict[str, Monkey]:
    with open("input.txt") as file:
        lines = file.readlines()
        lines = list(map(lambda s: s.strip("\n"), lines))
        monkeys = {}

        for line in lines:
            monkey = parse(line)
            monkeys[monkey.id_] = monkey

        return monkeys


def parse(line: str) -> Monkey:
    id_, operation = line.split(": ")

    if not any([op in operation for op in OPS]):
        return Monkey(id_, int(operation))

    for op in OPS:
        if op in operation:
            lhs, rhs = operation.split(f" {op} ")
            return Monkey(id_, (lhs, op, rhs))

    raise ValueError(f"Couldn't parse Monkey: {line}")


def expr_to_str(expression: Union[int, str, Tuple]):
    if isinstance(expression, (int, str)):
        return str(expression)

    lhs, op, rhs = expression

    lhs_str = expr_to_str(lhs)
    rhs_str = expr_to_str(rhs)

    if any([op in lhs_str for op in OPS]):
        lhs_str = f"({lhs_str})"

    if any([op in rhs_str for op in OPS]):
        rhs_str = f"({rhs_str})"

    return f"{lhs_str} {op} {rhs_str}"


def solve(expression: Union[Tuple, int, str], value: int) -> int:
    if isinstance(expression, int):
        return expression

    if isinstance(expression, str):
        assert expression == "x"
        return int(value)

    print(f"{value} = {expr_to_str(expression)}")
    lhs, op, rhs = expression

    try:
        rhs_val = eval(expr_to_str(rhs))
        new_value = simplify(value, op, operand=rhs_val, operand_side="right")
        return solve(lhs, new_value)

    except NameError:
        pass

    try:
        lhs_val = eval(expr_to_str(lhs))
        new_value = simplify(value, op, operand=lhs_val, operand_side="left")

        return solve(rhs, new_value)

    except NameError:
        raise ValueError("Couldn't evaluate neither left nor right hand side expression!")


def simplify(value: int, op: str, operand: int, operand_side: str):
    if operand_side not in {"left", "right"}:
        raise ValueError(f"Unrecognized side: {operand_side}")

    if op == "/":
        # Handle the fact that division is not commutative
        if operand_side == "right":
            new_value = value * operand
        elif operand_side == "left":
            new_value = 1.0 / (value / operand)
    elif op == "*":
        new_value = value / operand
    elif op == "+":
        new_value = value - operand
    elif op == "-":
        # Handle the fact that subtraction is not commutative
        if operand_side == "right":
            new_value = value + operand
        elif operand_side == "left":
            new_value = -(value - operand)
    else:
        raise ValueError(f"Couldn't handle op {op}")

    return new_value


def main():
    monkeys = read()
    root_value = monkeys["root"].eval(monkeys)
    print(f"Root value: {root_value}")

    # Change root to use the equality operator
    lhs, _, rhs = monkeys["root"].expression
    monkeys["root"].expression = (lhs, "==", rhs)

    # Change human to return a variable called x
    monkeys["humn"].expression = "x"
    root_expr = monkeys["root"].lazy_eval(monkeys)

    lhs, _, rhs = root_expr

    try:
        lhs_val = eval(expr_to_str(lhs))
        x_val = solve(rhs, lhs_val)
        print(f"X={x_val}")
    except NameError:
        pass

    try:
        rhs_val = eval(expr_to_str(rhs))
        x_val = solve(lhs, rhs_val)
        print(f"X={x_val}")
    except NameError:
        raise ValueError("Couldn't evaluate neither left nor right hand side!")


if __name__ == "__main__":
    main()