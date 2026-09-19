"""
Writer monad used in Experiment 3.

Structure:

  Carrier:    Writer[T] = (value: T, log: list[str])
  Log monoid: (list[str], ++, []) — list concatenation is associative
              with identity [], so partial logs combine under any
              grouping without changing the resulting log.
              Order across workers is not a monoid property. It is
              fixed by merge_writers() to the order of its input list,
              which makes the combined log deterministic for a given
              worker ordering.
  pure(a)   = (a, [])
  map:  (a, w).map(f)   =  (f(a), w)
  bind: (a, w1) >>= f   =  let (b, w2) = f(a) in (b, w1 ++ w2)

Monad laws hold because (list[str], ++, []) is a monoid:

  left identity   pure(a).bind(f) == f(a)          -- [] is a left unit
  right identity  m.bind(pure) == m                -- [] is a right unit
  associativity   m.bind(f).bind(g)                -- ++ is associative
                    == m.bind(lambda x: f(x).bind(g))

map is derivable as ``m.bind(lambda x: Writer.pure(f(x)))`` and is
provided directly only to avoid the intermediate Writer. The functor
laws (identity, composition) follow from that equivalence. All five
laws are verified by tests in the artifact repository.

Immutability: shallow and by construction. No method mutates self and
every operation returns a new Writer; ``__slots__`` prevents new
attributes, and both the constructor and the ``log`` property copy the
list, so no caller shares the internal list object. The wrapped value
is not copied, so a Writer over a mutable T is not deeply immutable —
Experiment 3 uses immutable values only.

Cost model (n = len(w1), m = len(w2); for single-Writer operations n
is that Writer's log length):

  bind             one concatenation plus the constructor's defensive
                   copy: two allocations of size n + m, O(n + m) time
  map              O(n) — the value is transformed, the log is copied
                   unchanged through the constructor
  log (property)   O(n) — defensive copy on every access
  merge_writers    O(k + M) for k writers and M total log entries,
                   reading each log through the copying property, so
                   the M entries are traversed twice

Standard library only, consistent with the rest of the package.
"""

from __future__ import annotations
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class Writer(Generic[T]):
    """Pairs a value with an accumulated log (see module docstring)."""

    __slots__ = ("_value", "_log")

    def __init__(self, value: T,
                 log: Optional[list[str]] = None) -> None:
        self._value = value
        self._log: list[str] = [] if log is None else list(log)

    @property
    def value(self) -> T:
        """The wrapped value. Not copied: a mutable T stays shared."""
        return self._value

    @property
    def log(self) -> list[str]:
        """A copy of the accumulated log, so callers cannot mutate it."""
        return list(self._log)

    @classmethod
    def pure(cls, value: T) -> "Writer[T]":
        """Lift a value into the monad with an empty log."""
        return cls(value, [])

    def map(self, f: Callable[[T], U]) -> "Writer[U]":
        """Apply f to the value, leaving the log contents unchanged."""
        return Writer(f(self._value), self._log)

    def bind(self, f: Callable[[T], "Writer[U]"]) -> "Writer[U]":
        """Apply f to the value and concatenate the two logs."""
        nxt = f(self._value)
        if not isinstance(nxt, Writer):
            raise TypeError(
                "Writer.bind expects a function returning Writer")
        return Writer(nxt._value, self._log + nxt._log)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Writer)
                and self._value == other._value
                and self._log == other._log)

    def __repr__(self) -> str:
        return f"Writer(value={self._value!r}, log={self._log!r})"


def merge_writers(writers: list[Writer]) -> tuple[list, list[str]]:
    """
    Collect values and concatenate logs, preserving input order.

    O(k) in the number of writers and O(M) in total log entries. Each
    log is read through the copying ``log`` property, so the entries
    are traversed twice: once for the defensive copy, once to build the
    combined list. The returned values are the same objects the writers
    hold, so a mutable T stays shared; no list is shared between the
    result and any writer.
    """
    values = [w.value for w in writers]
    combined = [entry for w in writers for entry in w.log]
    return values, combined
