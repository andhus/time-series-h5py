from __future__ import division, print_function

import abc

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from interval import TimeInterval, Interval


# MODES OF WRITING

# write: always replaces everything
# append: must be outside existing index (either before or after)
# update: index must be identical to existing
# combine: kwargs: propagate_nans=True (new always overwrites)
# replace: kwargs: range=None
from tsh5py.utils import PartitionedTSData


class Selection(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def to_item(self):
        """Returns object that can be passed to get item"""


class IntervalSelection(Selection, Interval):

    def __init__(self, start, end):
        super(IntervalSelection, self).__init__(start, end)

    def to_item(self):
        return self.to_slice()


class TSIndex(PartitionedTSData):
    """ Wraps hdf5.Group to represent the DatetimeIndex of a TSDataset
    """
    _partition_offset = 'partition_offset'
    _first_index = 'first_index'
    _last_index = 'last_index'

    def __init__(self, group):
        super(TSIndex, self).__init__(
            group=group,
            start=None,
            end=None
        )

    @classmethod
    def create(
        cls,
        group,
        partition_offset='24H'
    ):
        tsindex = cls(group)
        tsindex.partition_offset = to_offset(partition_offset)

        return tsindex

    def tsdata_to_array(self, tsdata):
        return tsdata.asi8

    def write(self, index):
        """ Writes the index to the TSIndexGroup.

        :param index: the index to write
        :type index: pd.DatetimeIndex

        :param sanity_check: do checks or not
        :type sanity_check: bool
        """
        self.sanity_check(index)
        self._replace_affected_partitions(index)
        self._update_first_index(index[0])
        self._update_last_index(index[-1])

    def extend(self, index):
        self.sanity_check(index)
        if index[0] > self.end:
            affected_partition = self.load_partition(
                index[0].floor(self.partition_offset)
            )
            updated_index = affected_partition.append(index)
            self._replace_affected_partitions(updated_index)
            self._update_last_index(index[-1])

        elif index[-1] < self.start:
            affected_partition = self.load_partition(
                index[-1].floor(self.partition_offset)
            )
            updated_index = index.append(affected_partition)
            self._replace_affected_partitions(updated_index)
            self._update_first_index(index[0])
        else:
            raise ValueError(
                'index must be completely before or after existing index'
            )

    def combine(self, index):
        index_affected_partitions = self.load_affected_partitions(index)
        updated_index = index_affected_partitions.union(index)
        self._update_first_index(index[0])
        self._update_last_index(index[-1])
        self._replace_affected_partitions(updated_index)

    def replace(self, index, interval=None):
        if interval is None:
            if not len(index) >= 2:
                raise ValueError('')
            start, end = index[0], index[-1]
            affected_interval = None
        else:
            affected_interval = interval.intersect(self)
            start, end = affected_interval

        first_partition_index = self.load_partition(
            start.floor(self.partition_offset)
        )
        last_partition_index = self.load_partition(
            end.floor(self.partition_offset)
        )
        first_keep = first_partition_index[
            :first_partition_index.get_slice_bound(start, 'left', 'ix')
        ]
        last_keep = last_partition_index[
            last_partition_index.get_slice_bound(end, 'right', 'ix'):
        ]
        updated_index = first_keep.append([index, last_keep])
        self._replace_affected_partitions(
            updated_index,
            interval=affected_interval
        )

        if start < self.start:
            self._update_first_index(updated_index[0], force=True)
        elif interval is not None and interval.is_inf_left():
            self._update_first_index(index[0], force=True)
        if end > self.end:
            self._update_last_index(updated_index[-1], force=True)
        elif interval is not None and interval.is_inf_right():
            self._update_last_index(index[-1], force=True)

    @staticmethod
    def sanity_check(index):
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError('')
        if not index.is_monotonic_increasing:
            raise ValueError('')

    def load(self, interval=None):
        interval = interval if interval is not None else slice(None, None)
        partition_selections = self.get_internal_partition_selections(interval)
        chunks = [
            self._group[self.get_partition_key(partition)][selection]
            for partition, selection in partition_selections
        ]
        index = pd.DatetimeIndex(np.concatenate(chunks))
        return index

    def load_partition(self, partition, selection=None):
        selection = selection or slice(None, None)
        partition_key = self.get_partition_key(partition)
        if partition_key in self._group:
            return pd.DatetimeIndex(self._group[partition_key][selection])
        else:
            return pd.DatetimeIndex([])

    def load_affected_partitions(self, index):
        if len(index) == 0:
            return pd.DatetimeIndex([])
        elif len(index) == 1:
            return self.load_partition(
                index[0].floor(self.partition_offset)
            )
        else:
            partitions = self.get_partitions_for_interval(
                TimeInterval(index[0], index[-1])
            )
            chunks = [
                self._group[self.get_partition_key(partition)][:]
                for partition in partitions
            ]
            index = pd.DatetimeIndex(np.concatenate(chunks))
            return index

    def get_partition_selections(self, selection):
        if isinstance(selection, slice):
            interval = TimeInterval.from_slice(selection)
            return self._get_partition_selection_s_from_interval(interval)
        elif isinstance(selection, TimeInterval):
            return self._get_partition_selection_s_from_interval(selection)
        elif isinstance(selection, pd.DatetimeIndex):
            return self._get_partition_selection_s_from_index(selection)
        else:
            raise NotImplementedError('')

    def get_internal_partition_selections(self, interval):
        """
        :param interval:
        :return: [(<partition key>, selection)]
        """
        if isinstance(interval, slice):
            interval = TimeInterval.from_slice(interval)
            return self._get_internal_partition_selection_s_from_interval(interval)
        elif isinstance(interval, TimeInterval):
            return self._get_internal_partition_selection_s_from_interval(interval)
        else:
            raise TypeError('')

    def get_point_partition_selection(self, ts):
        partition = ts.floor(self.partition_offset)
        partition_index = self.load_partition(partition)
        within_partition_idx = partition_index.get_loc(ts)

        return partition, within_partition_idx

    def _get_partition_selection_s_from_index(self, index):
        """
        :param index:
        :return:
        """
        index_series = pd.Series(data=range(len(index)), index=index)
        grouper = index_series.groupby(pd.TimeGrouper(self.partition_offset))
        selections = []
        for partition, chunk in grouper:
            if chunk.empty:
                continue
            partition_index = self.load_partition(partition)
            partition_series = pd.Series(
                range(len(partition_index)),
                index=partition_index
            )
            selection_series = partition_series[chunk.index]
            if selection_series.isnull().any():
                raise KeyError('')
            selections.append(
                (partition, chunk.index, list(selection_series.values))
            )
        return selections

    def _get_internal_partition_selection_s_from_interval(self, interval):
        """Processes interval (range) queries"""
        interval = interval.intersect(self)
        partitions = self.get_partitions_for_interval(interval)
        if len(partitions) == 0:
            return []
        first_selection = self._get_partition_selection_for_interval(
            partitions[0],
            interval
        )
        if len(partitions) == 1:
            return [(partitions[0], first_selection)]

        last_selection = self._get_partition_selection_for_interval(
            partitions[-1],
            interval
        )

        selections = (
            [first_selection] +
            [slice(None, None) for _ in range(len(partitions) - 2)] +
            [last_selection]
        )
        partition_and_selection_s = zip(partitions, selections)

        return partition_and_selection_s

    def _get_partition_selection_s_from_interval(self, interval):
        internal_ps = self._get_internal_partition_selection_s_from_interval(interval)
        partitions, selections = zip(*internal_ps)
        indexes = [
            self.load_partition(partition, selection)
            for partition, selection in internal_ps
        ]
        return zip(partitions, indexes, selections)

    def get_partitions_for_interval(self, interval):
        start, end = interval.intersect(self)
        partitions = pd.date_range(
            start=start.floor(self.partition_offset),
            end=end.floor(self.partition_offset),
            freq=self.partition_offset
        )
        return [p for p in partitions if self.get_partition_key(p) in self._group]

    def _get_partition_selection_for_interval(self, partition, interval):
        """interval already sliced!"""
        partition_index = self.load_partition(partition)

        if TimeInterval(
            partition,
            partition + self.partition_offset.delta
        ).intersect(interval).empty:
            return slice(0, 0)

        if interval.start > partition_index[0]:
            start = partition_index.get_slice_bound(
                interval.start,
                'left',
                'ix'
            )
        else:
            start = None

        if interval.end < partition_index[-1]:
            stop = partition_index.get_slice_bound(
                interval.end,
                'right',
                'ix'
            )
        else:
            stop = None

        return slice(start, stop)

    @property
    def partition_offset(self):
        return to_offset(self._group.attrs[self._partition_offset])

    @partition_offset.setter
    def partition_offset(self, value):
        if self._partition_offset in self._group.attrs:
            raise ValueError('')
        self._group.attrs[self._partition_offset] = to_offset(value).freqstr

    @property
    def start(self):
        if self._first_index in self._group.attrs:
            return pd.Timestamp(self._group.attrs[self._first_index])
        else:
            return None

    def _update_first_index(self, timestamp, force=False):
        if force:
            new_first = timestamp
        else:
            new_first = self.earliest_defined(self.start, timestamp)
        self._group.attrs[self._first_index] = int(new_first.asm8)

    @property
    def end(self):
        if self._last_index in self._group.attrs:
            return pd.Timestamp(self._group.attrs[self._last_index])
        else:
            return None

    def _update_last_index(self, timestamp, force=False):
        if force:
            new_last = timestamp
        else:
            new_last = self.latest_defined(self.end, timestamp)
        self._group.attrs[self._last_index] = int(new_last.asm8)

    @property
    def dtype(self):
        return np.int64

    @dtype.setter
    def dtype(self, value):
        raise NotImplementedError(
            'dtype can not be set for {}'.format(self.__class__.__name__)
        )