from __future__ import division, print_function

import os
import shutil
import tempfile

import h5py
import pandas as pd
from nose.tools import assert_equal, assert_raises
from pandas.tseries.frequencies import to_offset
from pandas.util.testing import assert_index_equal

from tsh5py.interval import TimeInterval


class TestTSIndex(object):
    from tsh5py.index import TSIndex
    test_class = TSIndex

    def create_empty_instance(self, fh):
        return self.test_class.create(
            group=fh.create_group(self.default_group_name),
            partition_offset=self.default_partition_offset
        )

    def create_default_instance(self, fh):
        instance = self.create_empty_instance(fh)
        index = self.default_index
        instance.write(index)

        return instance

    def get_root_fh(self, mode='w'):
        fh = h5py.File(name=self.file_path, mode=mode)
        return fh

    def setup(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_filename = '{}.hdf5'.format(self.__class__.__name__)
        self.file_path = os.path.join(self.temp_dir, self.temp_filename)
        self.default_group_name = 'tsindex'
        self.default_start = pd.Timestamp('2017-01-01 00:00')
        self.default_end = pd.Timestamp('2017-01-05 03:00')
        self.default_partition_offset = to_offset('24H')
        self.default_index = pd.date_range(
            start=self.default_start,
            end=self.default_end,
            freq='1H'
        )

    def teardown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_empty(self):
        with self.get_root_fh() as fh:
            tsindex = self.create_empty_instance(fh)
            assert_equal(tsindex.start, None)
            assert_equal(tsindex.end, None)

    def test_start_end(self):
        with self.get_root_fh() as fh:
            tsindex = self.create_default_instance(fh)
            assert_equal(tsindex.start, self.default_start)
            assert_equal(tsindex.end, self.default_end)

    def test_partition_offset(self):
        with self.get_root_fh() as fh:
            tsindex = self.create_default_instance(fh)
            assert_equal(
                tsindex.partition_offset,
                self.default_partition_offset
            )

    def test_load_all(self):
        with self.get_root_fh(mode='w') as fh:
            tsindex = self.create_default_instance(fh)
            index = tsindex.load()
        assert_index_equal(
            index,
            self.default_index
        )

    def test_load_sub_interval_start_end_in_index(self):
        with self.get_root_fh(mode='w') as fh:
            tsindex = self.create_default_instance(fh)
            index = tsindex.load(
                slice(self.default_index[2], self.default_index[5])
            )
        assert_index_equal(
            index,
            self.default_index[2:6]
        )

    def test_load_sub_interval_start_end_not_in_index(self):
        with self.get_root_fh(mode='w') as fh:
            tsindex = self.create_default_instance(fh)
            index = tsindex.load(
                slice(
                    self.default_index[2] - pd.Timedelta('1ns'),
                    self.default_index[5] + pd.Timedelta('1ns')
                )
            )
        assert_index_equal(
            index,
            self.default_index[2:6]
        )

    def test_write_and_load(self):
        with self.get_root_fh() as fh:
            tsindex = self.create_empty_instance(fh)
            index = pd.DatetimeIndex([
                '2017-01-01 00:00',
                '2017-01-01 01:00',
                '2017-01-02 06:00',
                '2017-01-02 07:00:00.002',
            ])
            tsindex.write(index)
        with self.get_root_fh(mode='r') as fh:
            tsindex = self.test_class(group=fh[self.default_group_name])
            index_rec = tsindex.load()
        assert_index_equal(
            index,
            index_rec
        )

    def test_extend_after(self):
        extension_after = pd.date_range(
            self.default_end + pd.Timedelta('2H'),
            periods=3,
            freq='1min'
        )
        with self.get_root_fh(mode='w') as fh:
            tsindex = self.create_default_instance(fh)
            tsindex.extend(extension_after)
            total_index = tsindex.load()
        assert_index_equal(
            total_index,
            self.default_index.append(extension_after)
        )

    def test_extend_before(self):
        extension_before = pd.date_range(
            self.default_start - pd.Timedelta('1H'),
            periods=3,
            freq='1min'
        )
        with self.get_root_fh(mode='w') as fh:
            tsindex = self.create_default_instance(fh)
            tsindex.extend(extension_before)
            total_index = tsindex.load()
        assert_index_equal(
            total_index,
            extension_before.append(self.default_index)
        )

    def test_extend_overlaps_end(self):
        overlapping = pd.date_range(
            self.default_end - pd.Timedelta('1H'),
            periods=3,
            freq='1min'
        )
        with self.get_root_fh(mode='w') as fh:
            tsindex = self.create_default_instance(fh)
            with assert_raises(ValueError):
                tsindex.extend(overlapping)

    def test_extend_overlaps_start(self):
        overlapping = pd.date_range(
            self.default_start - pd.Timedelta('1H'),
            periods=3,
            freq='1H'
        )
        with self.get_root_fh(mode='w') as fh:
            tsindex = self.create_default_instance(fh)
            with assert_raises(ValueError):
                tsindex.extend(overlapping)

    def test_combine(self):
        added = pd.date_range(
            self.default_start + pd.Timedelta('10min'),
            periods=3,
            freq='1H'
        )
        with self.get_root_fh(mode='w') as fh:
            tsindex = self.create_default_instance(fh)
            tsindex.combine(added)
            total_index = tsindex.load()
        assert_index_equal(
            total_index,
            self.default_index.union(added)
        )

    def test_replace_no_interval(self):
        original_index = pd.DatetimeIndex([
            '2017-01-01 00:00',
            '2017-01-01 01:00',
            '2017-01-02 06:00',
            '2017-01-02 07:00:00.002',
        ])
        replace_index = pd.DatetimeIndex([
            '2017-01-01 00:30',
            '2017-01-02 07:00',
        ])
        expected_index = pd.DatetimeIndex([
            '2017-01-01 00:00',
            '2017-01-01 00:30',
            '2017-01-02 07:00',
            '2017-01-02 07:00:00.002',
        ])
        with self.get_root_fh() as fh:
            tsindex = self.create_empty_instance(fh)
            tsindex.write(original_index)
            tsindex.replace(replace_index)
        with self.get_root_fh(mode='r') as fh:
            tsindex = self.test_class(group=fh[self.default_group_name])
            total_index = tsindex.load()
        assert_index_equal(
            total_index,
            expected_index
        )

    def test_replace_no_interval_existing_start_end(self):
        original_index = pd.DatetimeIndex([
            '2017-01-01 00:00',
            '2017-01-01 01:00',
            '2017-01-02 06:00',
            '2017-01-02 07:00:00.002',
        ])
        replace_index = pd.DatetimeIndex([
            '2017-01-01 01:00',
            '2017-01-02 07:00:00.002',
        ])
        expected_index = pd.DatetimeIndex([
            '2017-01-01 00:00',
            '2017-01-01 01:00',
            '2017-01-02 07:00:00.002',
        ])
        with self.get_root_fh() as fh:
            tsindex = self.create_empty_instance(fh)
            tsindex.write(original_index)
            tsindex.replace(replace_index)
        with self.get_root_fh(mode='r') as fh:
            tsindex = self.test_class(group=fh[self.default_group_name])
            total_index = tsindex.load()
        assert_index_equal(
            total_index,
            expected_index
        )

    def test_replace_with_interval(self):
        original_index = pd.DatetimeIndex([
            '2017-01-01 00:00',
            '2017-01-01 01:00',
            '2017-01-02 06:00',
            '2017-01-02 07:00:00.002',
        ])
        replace_index = pd.DatetimeIndex([
            '2017-01-01 01:00',
        ])
        replace_interval = TimeInterval(original_index[0], original_index[-1])
        expected_index = replace_index
        with self.get_root_fh() as fh:
            tsindex = self.create_empty_instance(fh)
            tsindex.write(original_index)
            tsindex.replace(replace_index, interval=replace_interval)
        with self.get_root_fh(mode='r') as fh:
            tsindex = self.test_class(group=fh[self.default_group_name])
            total_index = tsindex.load()
        assert_index_equal(
            total_index,
            expected_index
        )

    def test_replace_with_inf_interval(self):
        original_index = pd.DatetimeIndex([
            '2017-01-01 00:00',
            '2017-01-01 01:00',
            '2017-01-02 06:00',
            '2017-01-02 07:00:00.002',
        ])
        replace_index = pd.DatetimeIndex([
            '2017-01-01 01:00',
        ])
        replace_interval = TimeInterval.Inf()
        expected_index = replace_index
        with self.get_root_fh() as fh:
            tsindex = self.create_empty_instance(fh)
            tsindex.write(original_index)
            tsindex.replace(replace_index, interval=replace_interval)
        with self.get_root_fh(mode='r') as fh:
            tsindex = self.test_class(group=fh[self.default_group_name])
            total_index = tsindex.load()
        assert_index_equal(
            total_index,
            expected_index
        )

    def test_get_partitions_for_interval(self):
        with self.get_root_fh() as fh:
            tsindex = self.create_default_instance(fh)
            interval = TimeInterval(
                '2017-01-01 00:00',
                '2017-01-02 03:00',
            )
            partitions = tsindex.get_partitions_for_interval(interval)
            partitions_expected = list(pd.DatetimeIndex(
                ['2017-01-01', '2017-01-02']
            ))
            assert_equal(partitions, partitions_expected)

    def test_get_partition_selections_from_interval(self):
        with self.get_root_fh() as fh:
            tsindex = self.create_default_instance(fh)
            interval_selection = TimeInterval(
                '2017-01-01 00:00',
                '2017-01-03 03:00',
            )
            ps = tsindex.get_internal_partition_selections(interval_selection)
            ps_expected = [
                (
                    pd.Timestamp('2017-01-01'),
                    slice(None, None)
                ),
                (
                    pd.Timestamp('2017-01-02'),
                    slice(None, None)
                ),
                (
                    pd.Timestamp('2017-01-03'),
                    slice(None, 4)
                )
            ]
            assert_equal(ps, ps_expected)

    def test_get_partition_selections_from_index(self):
        with self.get_root_fh() as fh:
            tsindex = self.create_default_instance(fh)
            index_selection = pd.DatetimeIndex([
                '2017-01-01 00:00',                      # p1, [0]
                '2017-01-02 01:00', '2017-01-02 02:00',  # p2, [1, 2]
                '2017-01-04 03:00'                       # p4, [3]
            ])
            pss = tsindex.get_partition_selections(index_selection)
            pss_expected = [
                (pd.Timestamp('2017-01-01'), pd.DatetimeIndex(['2017-01-01 00:00']), [0]),
                (pd.Timestamp('2017-01-02'), pd.DatetimeIndex(['2017-01-02 01:00', '2017-01-02 02:00']), [1, 2]),
                (pd.Timestamp('2017-01-04'), pd.DatetimeIndex(['2017-01-04 03:00']), [3])
            ]
            assert_equal(len(pss), len(pss_expected))
            for ps, ps_expected in zip(pss, pss_expected):
                assert_equal(ps[0], ps_expected[0])
                assert_index_equal(ps[1], ps_expected[1])
                assert_equal(ps[2], ps_expected[2])
