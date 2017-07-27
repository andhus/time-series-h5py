from __future__ import print_function, division

import os
import shutil
import tempfile

import numpy as np
import h5py
import pandas as pd

from nose.tools import assert_equal
from pandas.tseries.frequencies import to_offset
from pandas.util.testing import assert_series_equal

from tsh5py.interval import TimeInterval


class TestTSDataset(object):
    from tsh5py.dataset import TSDataset
    test_class = TSDataset

    def get_root_fh(self, mode='w'):
        fh = h5py.File(name=self.file_path, mode=mode)
        return fh

    def setup(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_filename = '{}.hdf5'.format(self.__class__.__name__)
        self.file_path = os.path.join(self.temp_dir, self.temp_filename)
        self.default_group_name = 'ts_dataset'
        self.default_partition_offset = to_offset('24H')

    def teardown(self):
        shutil.rmtree(self.temp_dir)

    def get_and_write_example_series(self):
        fh = self.get_root_fh()
        tsds = self.test_class.create(
            group=fh.create_group(self.default_group_name),
            partition_offset='24H'
        )
        series = pd.Series(
            [1., 3., 4., 6., 1.],
            index=pd.date_range('2017', periods=5, freq='6H')
        )
        tsds.write({'test': series})
        fh.close()

        return series

    def test_extend(self):
        series = self.get_and_write_example_series()
        extend_series = pd.Series(
            [0., 0.],
            index=pd.DatetimeIndex([
                series.index[-1] + pd.Timedelta('1min'),
                series.index[-1] + pd.Timedelta('3min')
            ])
        )
        with self.get_root_fh(mode='a') as fh:
            tsds = self.test_class(fh[self.default_group_name])
            tsds.extend({'test': extend_series})
            series_total = tsds.load('test')

        assert_series_equal(
            series_total,
            pd.concat([series, extend_series])
        )

    def test_extend_new_partition(self):
        series = self.get_and_write_example_series()
        start_time = series.index[-1] + self.default_partition_offset
        extend_series = pd.Series(
            [0., 0.],
            index=pd.DatetimeIndex([
                start_time + pd.Timedelta('1min'),
                start_time + pd.Timedelta('3min')
            ])
        )
        with self.get_root_fh(mode='a') as fh:
            tsds = self.test_class(fh[self.default_group_name])
            tsds.extend({'test': extend_series})
            series_total = tsds.load('test')

        assert_series_equal(
            series_total,
            pd.concat([series, extend_series])
        )

    def test_combine(self):
        series = self.get_and_write_example_series()
        combine_series = pd.Series(
            [0, 0, np.nan],
            index=pd.DatetimeIndex([
                series.index[0] + pd.Timedelta('1min'),
                series.index[1] + pd.Timedelta('3min'),
                series.index[1] + pd.Timedelta('1H')
            ])
        )
        expected_series = combine_series.combine_first(series)
        expected_series[series.index[1] + pd.Timedelta('1H')] = np.nan

        with self.get_root_fh(mode='a') as fh:
            tsds = self.test_class(fh[self.default_group_name])
            tsds.combine({'test': combine_series})
            series_total = tsds.load('test')

        assert_series_equal(series_total, expected_series)

    def test_replace(self):
        series = self.get_and_write_example_series()
        replace_series = pd.Series(
            [0, np.nan],
            index=series.index[[1, -2]]
        )
        expected_series = pd.concat(
            [series[:1], replace_series, series[-1:]]
        )

        with self.get_root_fh(mode='a') as fh:
            tsds = self.test_class(fh[self.default_group_name])
            tsds.replace({'test': replace_series})
            series_total = tsds.load('test')

        assert_series_equal(series_total, expected_series)

    def test_update(self):
        series = self.get_and_write_example_series()
        update_series = series.copy()[1:3]
        update_series[:] = 3.001
        expected_series = series.copy()
        expected_series.ix[update_series.index] = update_series

        with self.get_root_fh(mode='a') as fh:
            tsds = self.test_class(fh[self.default_group_name])
            tsds.update({'test': update_series})
            series_total = tsds.load('test')

        assert_series_equal(series_total, expected_series)

    def test_interval_selection(self):
        series = self.get_and_write_example_series()
        with self.get_root_fh(mode='r') as fh:
            tsds = self.test_class(fh[self.default_group_name])
            series_rec = tsds.load('test', TimeInterval('2016', '2019'))
        assert_series_equal(series, series_rec)

    def test_point_selection(self):
        series = self.get_and_write_example_series()

        with self.get_root_fh(mode='r') as fh:
            tsds = self.test_class(fh[self.default_group_name])
            ts = pd.Timestamp('2017-01-01 00:00:00')
            value = tsds.load('test', ts)
        assert_equal(value, series[ts])

    def test_index_selection(self):
        series = self.get_and_write_example_series()

        fh = self.get_root_fh(mode='r')
        tsds = self.test_class(fh[self.default_group_name])
        ser_rec = tsds.load(
            'test', series.index
        )
        fh.close()

        assert_series_equal(series, ser_rec)