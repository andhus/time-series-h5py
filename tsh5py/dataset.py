from __future__ import print_function, division

from operator import methodcaller

import numpy as np
import pandas as pd

from tsh5py.index import TSIndex
from tsh5py.tssource import TSSource
from tsh5py.utils import TimeInterval


class TSDataset(object):

    _index_key = 'index'
    _sources_key = 'sources'

    def __init__(
        self,
        group,
        compression=None,
        compression_opts=None
    ):
        self._group = group
        # self._remote_index = remote_index  # TODO...
        self._data_hdf5_opts = dict(
            compression=compression,
            compression_opts=compression_opts
        )

    @classmethod
    def create(
        cls,
        group,
        partition_offset='24H',
        compression='gzip',
        compression_opts=9
    ):
        tsds = TSDataset(group, compression, compression_opts)
        tsds.create_index(partition_offset)
        return tsds

    def create_index(self, partition_offset):
        index_group = TSIndex.create(
            self._group.create_group(self._index_key),
            partition_offset=partition_offset
        )
        return index_group

    def require_index(self):
        index_group = TSIndex(
            self._group.require_group(self._index_key)
        )
        return index_group

    def create_source(self, name, stype):
        source = TSSource.create(
            group=self._group.require_group(self._sources_key).create_group(name),
            index=self.require_index(),
            stype=stype,
            **self._data_hdf5_opts
        )
        return source

    def require_source(self, name, stype=None):
        sources_group = self._group.require_group(self._sources_key)
        if name in sources_group:
            source = TSSource.from_group_and_index(
                group=sources_group[name],
                index=self.require_index(),
                **self._data_hdf5_opts
            )
            if stype:
                assert source.stype is stype
        else:
            source = self.create_source(name, stype)

        return source

    @property
    def source_groups(self):
        return self._group.get(self._sources_key, {})

    @property
    def source_names(self):
        return self.source_groups.keys()

    def write(self, source_to_data):
        if self._sources_key in self._group:
            del self._group[self._sources_key]
        index = self.get_check_index(source_to_data)
        tsindex = self.require_index()
        tsindex.write(index=index)
        for name, tsdata in source_to_data.items():
            tssource = self.require_source(name, type(tsdata))
            tssource.write(tsdata)

    def extend(self, source_to_data):
        if self.source_names:
            # to support doing this as first operation
            self._execute_index_modifying_addition(source_to_data, 'extend')
        else:
            self.write(source_to_data)

    def combine(self, source_to_data):
        if self.source_names:
            # to support doing this as first operation
            self._execute_index_modifying_addition(source_to_data, 'combine')
        else:
            self.write(source_to_data)

    def replace(self, source_to_data, interval=None):
        if self.source_names:
            # to support doing this as first operation
            interval = self._normalize_interval(interval)
            self._execute_index_modifying_addition(
                source_to_data,
                'replace',
                interval=interval
            )
        else:
            self.write(source_to_data)

    def update(self, source_to_data):
        for name, tsdata in source_to_data.items():
            source = self.require_source(name, type(tsdata))
            source.update(tsdata)

    @staticmethod
    def _normalize_interval(interval):
        if isinstance(interval, slice):
            interval = TimeInterval.from_slice(interval)
        elif isinstance(interval, tuple):
            interval = TimeInterval.from_slice(slice(*interval))
        else:
            if not (isinstance(interval, TimeInterval) or interval is None):
                raise TypeError('')

        return interval

    @staticmethod
    def _normalize_selection(selection):
        if isinstance(selection, (pd.DatetimeIndex, pd.Timestamp)):
            pass
        elif isinstance(selection, str):
            selection = pd.Timestamp(selection)
        elif selection is None:
            selection = slice(None, None)
        else:
            try:
                selection = TSDataset._normalize_interval(selection)
            except TypeError:
                raise TypeError(
                    'selection of type {} not supported'.format(
                        type(selection)
                    )
                )

        return selection

    def _execute_index_modifying_addition(
        self,
        source_to_data,
        method,
        **kwargs
    ):
        index = self.get_check_index(source_to_data)
        for name in list(set(self.source_names + source_to_data.keys())):
            if name in source_to_data:
                tsdata = source_to_data[name]
                source = self.require_source(name, type(tsdata))
                methodcaller(method, tsdata, **kwargs)(source)
            else:
                source = self.require_source(name)
                tsdata = source.get_empty_data().ix[index]
                kwargs_copy = kwargs.copy()
                if method == 'combine':
                    kwargs_copy['propagate_nans'] = False
                methodcaller(method, tsdata, **kwargs_copy)(source)
        tsindex = self.require_index()
        methodcaller(method, index, **kwargs)(tsindex)

    def get_check_index(self, source_to_data):
        # TODO check that all sources share same index?
        return source_to_data.values()[0].index

    def load(self, sources=None, selection=None):
        selection = self._normalize_selection(selection)
        if selection is None:  # TODO can not be moved to normalize to to
            selection = slice(None, None)

        index = self.require_index()

        if isinstance(sources, (list, tuple)):
            sources_ = sources
        elif isinstance(sources, str):
            sources_ = [sources]
        elif sources is None:
            sources_ = self.source_names
        else:
            raise TypeError()
        source_to_tsdata = {
            source: TSSource.from_group_and_index(
                self.source_groups[source], index
            ).load(selection) for source in sources_
        }
        if isinstance(sources, str):
            return source_to_tsdata[sources]
        else:
            return source_to_tsdata

    def load_index(self, interval=None):
        interval = self._normalize_selection(interval)
        index = self.require_index()
        return index.load(interval)

    def __getitem__(self, item):
        """Pandas-alike way of loading data

        data = tsds['s1'][:] # gives all data from source 's1'
        datas = tsds[['s1', 's2']][pd.Timestamp('2017'):pd.Timestamp('2018')]
        # gives data in range for si and s2

        TODO proper docs.
        """
        # TODO type checks
        if isinstance(item, (str, list)):
            return TSDataSetView(self, item)
        else:
            return self.load(selection=item)


class TSDataSetView(object):
    # TODO docs
    # TODO repr
    def __init__(
        self,
        tsds,
        source_selection=None
    ):
        self.tsds = tsds
        self.source_selection = source_selection

    def __getitem__(self, item):
        return self.tsds.load(sources=self.source_selection, selection=item)
