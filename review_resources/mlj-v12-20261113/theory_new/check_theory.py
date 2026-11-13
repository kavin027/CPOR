"""Exact arithmetic checks of the finite examples accompanying the analytical proofs."""
from fractions import Fraction as F


def envelope(reference, candidate, factor):
    v = max(F(0), (factor * reference - candidate) / (factor - 1))
    return (candidate - v) ** 2 - factor * (reference - v) ** 2


def settle(allowance, reference, release, actual, factor):
    return allowance + factor * (reference - actual) ** 2 - (release - actual) ** 2


def run():
    reference = F(1)
    factor = F(2)
    initial = settle(F(0), reference, reference, F(0), factor)
    assert initial == 1
    memory = []
    for issue_two in [F(1), F(1, 2)]:
        assert envelope(reference, issue_two, factor) <= initial
        memory.append(settle(initial, reference, issue_two, F(1), factor))
    required = envelope(reference, F(1, 3), factor)
    assert (memory[0], memory[1], required) == (F(1), F(3, 4), F(8, 9))
    assert memory[0] >= required > memory[1]
    print("Third issue: allowances 1 and 3/4; cost of 1/3 equals 8/9.")

    for factor in [F(101, 100)**2, F(201, 200)**2, F(2), F(5, 2), F(100)]:
        step = 1 - 1 / factor
        for actual in [F(0), F(2)]:
            informed = 1 + (actual - 1) * step
            assert envelope(reference, informed, factor) == step <= factor - 1
            assert (informed - actual) ** 2 == 1 / factor**2
            assert settle(0, reference, informed, actual, factor) == factor - 1/factor**2 > 0
        for causal in [F(0), F(3, 4), F(1), F(5, 2)]:
            mean_loss = (causal**2 + (causal - 2)**2) / 2
            assert mean_loss == 1+(causal-1)**2 >= 1
        print(f"R={factor}: first allowance {factor-1}, reservation {step}, expected gap per later issue >= {1-1/factor**2}")


if __name__ == "__main__":
    run()
