from __future__ import division, print_function

import abc
from functools import partial

import numpy as np
import pandas as pd

from tsh5py.interval import Interval, TimeInterval


class Prototype(object ):
    """Partial for classes (just for clarity)"""
    def __init__(self, cls, *args, **kwargs):
        if isinstance(cls, Prototype):
            args = cls._partial.args + args
            _kwargs = kwargs
            kwargs = cls._partial.keywords
            kwargs.update(_kwargs)
            cls = cls._partial.func
        self._partial = partial(cls, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._partial()

    def __repr__(self):
        return "{}(args={}, kwargs={})".format(
            self.__class__.__name__,
            self._partial.args,
            self._partial.keywords
        )


def groupby_regular(tsdata, partition_offset, interval):
    """ handles index and returns also partitions for which there is no data.
    """
    if isinstance(tsdata, pd.DatetimeIndex):
        tsdata = pd.Series(index=tsdata)
        return (
            (partition_key, data.index)
            for partition_key, data in _groupby_regular(
                tsdata,
                partition_offset,
                interval
            )
        )
    else:
        return _groupby_regular(tsdata, partition_offset, interval)


def _groupby_regular(tsdata, partition_offset, interval=None):
    partition_key_to_data = {
        partition_key: data
        for partition_key, data in tsdata.groupby(pd.TimeGrouper(partition_offset))
    }
    if interval is None:
        start = tsdata.index[0]
        end = tsdata.index[-1]
    else:
        start, end = interval
        # assert start <= tsdata.index[0]
        # assert end >= tsdata.index[-1]
    partition_keys = pd.date_range(
        start=start.floor(partition_offset),
        end=end.floor(partition_offset),
        freq=partition_offset
    )
    for partition_key in partition_keys:
        if partition_key in partition_key_to_data:
            yield partition_key, partition_key_to_data[partition_key]
        else:
            yield partition_key, tsdata[len(tsdata):]  # empty


class PartitionedTSData(TimeInterval):
    __metaclass__ = abc.ABCMeta

    _dtype_str_key = '_dtype'

    @abc.abstractproperty
    def partition_offset(self):
        raise NotImplementedError('')

    def __init__(
        self,
        group,
        start,
        end,
        compression=None,
        compression_opts=None
    ):
        super(PartitionedTSData, self).__init__(start, end)
        self._group = group
        self._compression = compression
        self._compression_opts = compression_opts

    @property
    def dtype(self):
        dtype_str = self._group.attrs.get(self._dtype_str_key, None)
        if dtype_str:
            return np.dtype(dtype_str)
        else:
            return None

    @dtype.setter
    def dtype(self, value):
        if self.dtype:
            raise ValueError('can only set dtype once')
        else:
            self._group.attrs[self._dtype_str_key] = value.str

    @staticmethod
    def get_partition_key(partition_ts):
        return partition_ts.isoformat()

    def has_data(self):
        """not to confuse with empty in baseclass"""
        return len(self._group) > 0

    def _replace_affected_partitions(
        self,
        tsdata,
        interval=None,
    ):
        """
        :param tsdata: object
        :param interval: (pd.Timestamp, pd.Timestamp)
        """
        for partition_ts, partition_data in groupby_regular(
            tsdata,
            self.partition_offset,
            interval
        ):
            partition_key = self.get_partition_key(partition_ts)
            if partition_key in self._group:
                del self._group[partition_key]
            partition_array = self.tsdata_to_array(partition_data)
            if not partition_array.shape[0] == 0:
                # check all nans? no need to write...
                partition_ds = self._group.create_dataset(
                    name=partition_key,
                    shape=partition_array.shape,
                    dtype=partition_array.dtype,
                    compression=self._compression,
                    compression_opts=self._compression_opts
                )
                partition_ds[:] = partition_array

    def tsdata_to_array(self, tsdata):
        return tsdata.astype(self.dtype).values
