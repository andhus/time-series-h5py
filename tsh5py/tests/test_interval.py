from __future__ import division, print_function

import pandas as pd

from nose.tools import assert_equal

from tsh5py.utils import Prototype


class TestInterval(object):
    from tsh5py.interval import Interval
    test_class = Interval

    @property
    def prototype(self):
        return Prototype(self.test_class, start=2, end=10)

    def test_init(self):
        self.prototype()

    @property
    def default_slice_start(self):
        return 5

    @property
    def default_slice_end(self):
        return 7

    @property
    def default_point_outside_right(self):
        return 20

    @property
    def default_point_outside_left(self):
        return 0

    def test_slice(self):
        interval = self.prototype()
        sliced = interval[self.default_slice_start:self.default_slice_end]
        expected = self.test_class(
            self.default_slice_start,
            self.default_slice_end
        )
        assert_equal(sliced, expected)

    def test_slice_open(self):
        interval = self.prototype()

        sliced_open = interval[:]
        sliced_open_expected = interval
        assert_equal(sliced_open, sliced_open_expected)

        sliced_open_right = interval[self.default_slice_start:]
        sliced_open_expected_right = self.test_class(
            self.default_slice_start,
            interval.end
        )
        assert_equal(sliced_open_right, sliced_open_expected_right)

        sliced_open_left = interval[:self.default_slice_end]
        sliced_open_expected_left = self.test_class(
            interval.start,
            self.default_slice_end
        )
        assert_equal(sliced_open_left, sliced_open_expected_left)

    def test_intersect(self):
        interval = self.prototype()
        assert_equal(
            interval.intersect(
                self.test_class(
                    self.default_slice_start,
                    self.default_slice_end
                )
            ),
            self.test_class(self.default_slice_start, self.default_slice_end)
        )

    def test__iter__(self):
        interval = self.prototype()
        start, end = interval
        assert_equal(start, interval.start)
        assert_equal(end, interval.end)

    def test_contains(self):
        interval = self.prototype()
        assert self.default_slice_start in interval
        assert self.default_point_outside_left not in interval


class TestTimeInterval(TestInterval):
    from ..interval import TimeInterval
    test_class = TimeInterval

    @property
    def prototype(self):
        return Prototype(
            self.test_class,
            start=pd.Timestamp('2017-01-01'),
            end=pd.Timestamp('2020-01-01')
        )

    @property
    def default_slice_start(self):
        return pd.Timestamp('2018')

    @property
    def default_slice_end(self):
        return pd.Timestamp('2019')

    @property
    def default_point_outside_left(self):
        return pd.Timestamp('2016')

    @property
    def default_point_outside_right(self):
        return pd.Timestamp('2021')
