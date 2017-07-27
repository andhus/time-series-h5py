from __future__ import print_function, division


import numpy as np
import pandas as pd


class TimestampedNDArray(object):

    def __init__(self, data, timestamp):
        self.data = data
        self.timestamp = timestamp

    def __getitem__(self, item):
        return TimestampedNDArray(
            self.data[item],
            self.timestamp
        )

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return '{}(\n{}\n{})'.format(
            self.__class__.__name__,
            self.timestamp,
            self.data
        )


class IndexedNDArray(object):

    def __init__(self, data, index):
        self.data = data
        self._index_series = pd.Series(
            data=np.arange(len(index)),
            index=index,
        )

    @property
    def index(self):
        return self._index_series.index

    @property
    def values(self):
        return self.data

    def __getitem__(self, item):
        if isinstance(item, pd.Timestamp):
            data_item = self.index.get_loc(item)
            return TimestampedNDArray(
                self.data[data_item],
                item
            )
        if isinstance(item, int):
            return TimestampedNDArray(
                self.data[item],
                self.index[item]
            )

        if isinstance(item, tuple):
            first_item = item[0]
            remainder_item = item[1:]
        else:
            first_item = item
            remainder_item = None

        index_series = self._index_series[first_item]
        first_item_data = slice(
            index_series.values[0],
            index_series.values[-1] + 1
        )
        if remainder_item:
            data_item = (first_item_data, ) + remainder_item
        else:
            data_item = first_item_data

        return IndexedNDArray(
            self.data[data_item],
            index_series.index
        )

    def __setitem__(self, key, value):
        if isinstance(key, pd.Timestamp):
            data_key = self.index.get_loc(key)
            self.data[data_key] = value

        elif isinstance(key, int):
            self.data[key] = value
        else:
            if isinstance(key, tuple):
                first_item = key[0]
                remainder_item = key[1:]
            else:
                first_item = key
                remainder_item = None

            index_series = self._index_series[first_item]
            first_item_data = slice(
                index_series.values[0],
                index_series.values[-1] + 1
            )
            if remainder_item:
                data_item = (first_item_data,) + remainder_item
            else:
                data_item = first_item_data

            self.data[data_item] = value

    def __repr__(self):
        return '{}(\n{}\n{})'.format(
            self.__class__.__name__,
            self.index.__repr__(),
            self.data.__repr__()
        )
