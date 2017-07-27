from __future__ import print_function, division

import abc
from warnings import warn

import numpy as np
import pandas as pd

from tsh5py.interval import TimeInterval
from tsh5py.utils import PartitionedTSData


class TSSource(PartitionedTSData):

    _stype_key = '_stype'

    _str_to_stype = {
        'series': pd.Series,
        'data_frame': pd.DataFrame,
    }
    _stype_to_str = {
        pd.Series: 'series',
        pd.DataFrame: 'data_frame'
    }

    @classmethod
    def get_source_class(cls, stype):
        stype_to_source_class = {
            pd.Series: SeriesSource,
            pd.DataFrame: DataFrameSource
        }
        return stype_to_source_class[stype]

    def __init__(self, group, index):
        super(TSSource, self).__init__(
            group,
            start=None,
            end=None
        )
        self.index = index

    @property
    def start(self):
        return self.index.start

    @property
    def end(self):
        return self.index.end

    @property
    def partition_offset(self):
        return self.index.partition_offset

    @classmethod
    def create(cls, group, index, stype):
        if stype not in cls._stype_to_str.keys():
            raise TypeError('')
        cls._set_stype(group, stype)
        return cls.from_group_and_index(group, index)

    @classmethod
    def from_group_and_index(cls, group, index):
        return cls.get_source_class(cls._get_stype(group))(group, index)

    @classmethod
    def _get_stype(cls, group):
        if cls._stype_key not in group.attrs:
            return None
        else:
            return cls._str_to_stype[group.attrs[cls._stype_key]]

    @classmethod
    def _set_stype(cls, group, stype):
        group.attrs[cls._stype_key] = cls._stype_to_str[stype]

    @property
    def stype(self):
        return self._get_stype(self._group)

    def _pre_write(self, tsdata):
        if not self.get_dtype(tsdata).kind == 'f':
            warn(
                'only float data is supported due to missing data encoding '
                '(np.nan), the data will be casted to float64'
            )
            tsdata = tsdata.astype(float)

        if self.has_data():
            self.verify_meta(tsdata)
            self.verify_dtype(tsdata)
        else:
            self.write_meta(tsdata)
            self.write_dtype(tsdata)

        return tsdata

    def write(self, tsdata, sanity_check=True):
        tsdata = self._pre_write(tsdata)
        # TODO sanity check..?
        self._replace_affected_partitions(
            tsdata,
            interval=TimeInterval(self.start, self.end)
        )

    def extend(self, tsdata):
        tsdata = self._pre_write(tsdata)
        index = tsdata.index
        if index[0] > self.end:
            partition = index[0].floor(self.partition_offset)
            affected_partition = self.load_partition(partition)
            updated_tsdata = pd.concat([affected_partition, tsdata])

        elif index[-1] < self.start:
            partition = index[-1].floor(self.partition_offset)
            affected_partition = self.load_partition(partition)
            updated_tsdata = pd.concat([tsdata, affected_partition])
        else:
            raise ValueError(
                'index must be completely before or after existing index'
            )
        self._replace_affected_partitions(updated_tsdata)

    def combine(self, tsdata, propagate_nans=True):
        tsdata = self._pre_write(tsdata)
        affected_partitions_tsdata = self.load_affected_partitions(tsdata)
        updated_tsdata = tsdata.combine_first(affected_partitions_tsdata)
        if propagate_nans:
            self.propagate_nans(updated_tsdata, tsdata)
        self._replace_affected_partitions(updated_tsdata)

    def replace(self, tsdata, interval=None):
        tsdata = self._pre_write(tsdata)
        if interval is None:
            if not len(tsdata) >= 2:
                raise ValueError('')
            start, end = tsdata.index[0], tsdata.index[-1]
        else:
            interval = interval.intersect(self)
            start, end = interval

        first_partition_tsdata = self.load_partition(
            start.floor(self.partition_offset)
        )
        last_partition_tsdata = self.load_partition(
            end.floor(self.partition_offset)
        )
        first_keep = first_partition_tsdata[
            :first_partition_tsdata.index.get_slice_bound(start, 'left', 'ix')
        ]
        last_keep = last_partition_tsdata[
            last_partition_tsdata.index.get_slice_bound(end, 'right', 'ix'):
        ]
        updated_tsdata = pd.concat([first_keep, tsdata, last_keep])
        self._replace_affected_partitions(updated_tsdata, interval=interval)

    @abc.abstractmethod
    def propagate_nans(self, combined_tsdata, new_tsdata):
        raise NotImplementedError('')

    def update(self, tsdata):
        tsdata = self._pre_write(tsdata)
        affected_partitions_tsdata = self.load_affected_partitions(tsdata)
        affected_partitions_tsdata.ix[tsdata.index] = tsdata
        self._replace_affected_partitions(affected_partitions_tsdata)

    def load(self, selection):
        if isinstance(selection, pd.Timestamp):
            partition_selection = self.index.get_point_partition_selection(
                selection
            )
            return self._load_point(partition_selection, selection)

        partition_selections = self.index.get_partition_selections(selection)
        tsdata_partitions = [
            self.load_partition_selection(*ps) for ps in partition_selections
        ]
        return pd.concat(tsdata_partitions)

    def _load_point(self, partition_selection, timestamp):
        """
        timestamp not used by default but used by extending classes
        """
        partition, selection = partition_selection
        partition_ds = self._group[self.index.get_partition_key(partition)]
        return partition_ds[selection]

    def load_partition_selection(self, partition, index, selection):
        if self.has_partition(partition):
            data = self._group[self.index.get_partition_key(partition)][selection]
            tsdata_partition = self.stype(
                data=data,
                index=index,
                **self.load_meta_kwargs()
            )
        else:
            tsdata_partition = self.get_empty_data().ix[index]

        return tsdata_partition

    def load_affected_partitions(self, tsdata):
        if len(tsdata) == 0:
            return self.get_empty_data()
        elif len(tsdata) == 1:
            return self.load_partition(
                tsdata.index[0].floor(self.partition_offset)
            )
        else:
            partitions = self.index.get_partitions_for_interval(
                TimeInterval(tsdata.index[0], tsdata.index[-1])
            )
            if partitions:
                return pd.concat(
                    [self.load_partition(partition) for partition in partitions]
                )
            else:
                return self.get_empty_data()

    def load_partition(self, partition):
        index = self.index.load_partition(partition)
        return self.load_partition_selection(
            partition,
            index,
            slice(None, None)
        )

    def has_partition(self, partition):
        return self.get_partition_key(partition) in self._group

    def write_meta(self, tsdata):
        pass

    def load_meta_kwargs(self):
        return {}

    def verify_meta(self, tsdata):
        pass

    def verify_dtype(self, tsdata):
        if not self.dtype == self.get_dtype(tsdata):
            raise TypeError('')

    def write_dtype(self, tsdata):
        self.dtype = self.get_dtype(tsdata)

    def get_dtype(self, tsdata):
        # needs overriding for DataFrames
        return tsdata.dtype

    def get_empty_data(self):
        return self.stype(index=pd.DatetimeIndex([]), **self.load_meta_kwargs())


class SeriesSource(TSSource):

    _name_key = '_name'

    def __init__(self, group, index):
        super(SeriesSource, self).__init__(group, index)
        assert self.stype is pd.Series

    @property
    def name(self):
        return self._group.attrs.get(self._name_key, None)

    @name.setter
    def name(self, value):
        if self._name_key in self._group.attrs:
            assert self._group.attrs[self._name_key] == value
        else:
            self._group.attrs[self._name_key] = value

    def load_meta_kwargs(self):
        return {'name': self.name}

    def write_meta(self, tsdata):
        if tsdata.name:
            self.name = tsdata.name

    def verify_meta(self, tsdata):
        if not tsdata.name == self.name:
            warn('writing series with different name from original')

    def propagate_nans(self, combined_tsdata, new_tsdata):
        combined_tsdata.ix[new_tsdata[new_tsdata.isnull()].index] = np.nan


class DataFrameSource(TSSource):

    _columns_key = '_name'

    def __init__(self, group, index):
        super(DataFrameSource, self).__init__(group, index)
        assert self.stype is pd.DataFrame

    @property
    def columns(self):
        if self._columns_key in self._group.attrs:
            return list(self._group.attrs[self._columns_key])
        else:
            return None

    @columns.setter
    def columns(self, value):
        if self.columns:
            assert self.columns == list(value)
        else:
            self._group.attrs[self._columns_key] = list(value)

    def load_meta_kwargs(self):
        return {'columns': self.columns}

    def write_meta(self, tsdata):
        self.columns = tsdata.columns

    def verify_meta(self, tsdata):
        if not list(tsdata.columns) == self.columns:
            raise ValueError('')

    def get_dtype(self, tsdata):
        dtypes = set(tsdata.dtypes)
        if not len(dtypes) == 1:
            raise TypeError(
                'only DataFrames with same dtype for all columns supported'
            )
        return dtypes.pop()

    def propagate_nans(self, combined_tsdata, new_tsdata):
        isnull = new_tsdata.isnull()
        combined_tsdata.ix[new_tsdata[isnull.any(axis=1)].index][isnull] = np.nan

    def _load_point(self, partition_selection, timestamp):
        data = super(DataFrameSource, self)._load_point(
            partition_selection, timestamp
        )
        return pd.Series(data, index=self.columns, name=timestamp)
