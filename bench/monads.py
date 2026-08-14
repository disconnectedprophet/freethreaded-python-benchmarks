"""
Writer monad used in Experiment 3.

Formal definition:

  Carrier:  Writer[T] = (value: T, log: list[str])
  Log monoid: (list[str], ++, []) — list concatenation is associative
              with identity [], so partial logs combine in any
              grouping without changing the ordering and the result log.  
              (Order across workers is fixed by merge_writers to input order,
              making the combined log deterministic for a given
              worker ordering.)
  pure(a) = (a, [])
  bind: (a, w1) >>= f  =  let (b, w2) = f(a) in (b, w1 ++ w2)

Monad laws hold because list concatenation is associative with
identity []:
  left identity:  pure(a).bind(f) == f(a)
  right identity:  m.bind(pure) == m
  associativity:  m.bind(f).bind(g) == m.bind(lambda x:
                                              f(x).bind(g))

The class is immutable by convention: no method mutates self; every
operation returns a new Writer.  Cost model: bind performs one list
concatenation, O(len(w1) + len(w2)) time and allocation;
merge_writers is O(total log entries).
"""

# Imports
from __future__ import annotations
from typing import Callable, Generic, Optional, TypeVar

# Data types
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
        return self._value

    @property
    def log(self) -> list[str]:
        return list(self._log)

    @classmethod
    def pure(cls, value: T) -> "Writer[T]":
        return cls(value, [])

    def map(self, f: Callable[[T], U]) -> "Writer[U]":
        return Writer(f(self._value), self._log)

    def bind(self, f: Callable[[T], "Writer[U]"]) -> "Writer[U]":
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
    O(n) in writers, O(m) in total log entries.  No shared object.
    """
    values = [w.value for w in writers]
    combined = [entry for w in writers for entry in w.log]
    return values, combined
