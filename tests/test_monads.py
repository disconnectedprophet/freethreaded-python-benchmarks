"""
Monad-law verification for the Writer implementation.

Run:  python -m tests.test_monads
"""

from bench.monads import Writer, merge_writers


def f(x: int) -> Writer:
    return Writer(x + 1, [f"f({x})"])


def g(x: int) -> Writer:
    return Writer(x * 2, [f"g({x})"])


def test_left_identity() -> None:
    a = 41
    assert Writer.pure(a).bind(f) == f(a)


def test_right_identity() -> None:
    m = Writer(7, ["start"])
    assert m.bind(Writer.pure) == m


def test_associativity() -> None:
    m = Writer(3, ["m"])
    assert m.bind(f).bind(g) == m.bind(lambda x: f(x).bind(g))


def test_log_monoid_identity_and_assoc() -> None:
    # (list, ++, []) — identity and associativity on the log itself.
    w = Writer(1, ["a"])
    assert w.bind(Writer.pure).log == ["a"]  # right identity
    l1, l2, l3 = ["a"], ["b"], ["c"]
    assert (l1 + l2) + l3 == l1 + (l2 + l3)  # associativity


def test_merge_preserves_order() -> None:
    ws = [Writer(i, [f"w{i}"]) for i in range(5)]
    values, log = merge_writers(ws)
    assert values == [0, 1, 2, 3, 4]
    assert log == ["w0", "w1", "w2", "w3", "w4"]

# Main guard
if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all monad-law tests passed")
