from __future__ import division, print_function

import abc

import numpy as np
import pandas as pd


class Interval(object):
    """supports open right/left by None"""
    __metaclass__ = abc.ABCMeta

    inf_left = -np.inf
    inf_right = np.inf

    def __init__(self, start, end):
        if start is None or end is None:
            assert start is None and end is None
        else:
            assert start <= end
        self._start = start
        self._end = end

    @classmethod
    def Inf(cls):
        return cls(cls.inf_left, cls.inf_right)

    @classmethod
    def InfLeft(cls, end):
        return cls(cls.inf_left, end)

    @classmethod
    def InfRight(cls, start):
        return cls(start, cls.inf_right)

    @classmethod
    def Empty(cls):
        return cls(None, None)

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def empty(self):
        return self.start is None

    def to_slice(self):
        if self.empty:
            raise NotImplementedError('')  # TODO?
        start = self.start if self.start != self.inf_left else None
        stop = self.end if self.end != self.inf_right else None

        return slice(start, stop)

    def __getitem__(self, item):
        assert isinstance(item, slice)

        if self.empty:
            return self

        if item.start is None:
            start = self.start
        elif self.start == self.inf_left:
            start = item.start
        else:
            start = max(item.start, self.start)

        if item.stop is None:
            end = self.end
        elif self.end == self.inf_right:
            end = item.stop
        else:
            end = min(item.stop, self.end)

        return self.__class__(start, end)

    def __iter__(self):
        return (t for t in [self.start, self.end])

    def __len__(self):
        return self.end - self.start

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __ne__(self, other):
        return not self == other

    def __contains__(self, item):
        if self.empty:
            return False
        else:
            return self.start <= item <= self.end

    def intersect(self, other):
        if other.empty or self.empty:
            return self.__class__.Empty()
        return self[other.to_slice()]

    @staticmethod
    def latest_defined(t1, t2):
        if t1 is None:
            return t2
        elif t2 is None:
            return t1
        else:
            return max(t1, t2)

    @staticmethod
    def earliest_defined(t1, t2):
        if t1 is None:
            return t2
        elif t2 is None:
            return t1
        else:
            return min(t1, t2)

    def __repr__(self):
        return '{}(start={}, end={})'.format(
            self.__class__.__name__,
            self.start,
            self.end
        )

    def is_inf_left(self):
        return self.start == self.inf_left

    def is_inf_right(self):
        return self.end == self.inf_right


class TimeInterval(Interval):

    def __init__(self, start, end):
        super(TimeInterval, self).__init__(
            self._to_internal(start),
            self._to_internal(end)
        )

    @classmethod
    def from_slice(cls, slice_):
        start = slice_.start if slice_.start is not None else cls.inf_left
        end = slice_.stop if slice_.stop is not None else cls.inf_right
        return cls(start, end)

    @property
    def start(self):
        return self._to_external(super(TimeInterval, self).start)

    @property
    def end(self):
        return self._to_external(super(TimeInterval, self).end)

    def _to_internal(self, ts):
        if isinstance(ts, int) or ts in [self.inf_left, self.inf_right, None]:
            return ts
        else:
            ts = pd.Timestamp(ts)  # try casting
            return int(ts.asm8)

    def _to_external(self, ts_int):
        if ts_int in [None, self.inf_left, self.inf_right]:
            return ts_int
        return pd.Timestamp(ts_int)

    def __contains__(self, item):
        if self.empty:
            return False
        else:
            return self._start <= self._to_internal(item) <= self._end
