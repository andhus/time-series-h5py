from __future__ import print_function, division

from pprint import pprint
import h5py
import pandas as pd

from tsh5py.dataset import TSDataset

file_path = '/home/andershuss/Temp/tsds.h5py'
handle = h5py.File(file_path, 'w')

tsds = TSDataset.create(handle)

df = pd.DataFrame(
    data={
        'a': [0.1, 0.2, 0.3],
        'b': [1.0, 2.0, 3.0]
    },
    index=pd.DatetimeIndex([
        '2017-01-01 01:00:00',
        '2017-01-01 02:00:01.120',
        '2017-10-01 00:00:00',
    ])
)

print('\nBASIC OPERATIONS\n')

print('\nWrite - can only be done once')
tsds.write({'df': df[:1]})
print(tsds.load('df'))


print('\nExtend - can only add data disjunct data (completely before or after)')
tsds.extend({'df': df[-1:]})
print(tsds.load('df'))
try:
    tsds.extend({'df': df})
except Exception as e:
    print('{}: {}'.format(type(e), e))


print(
    '\nCombine - can be used instead to do "index modifying" operations that '
    'does not only extend'
)
tsds.combine({'df': df[1:-1]})
print(tsds.load('df'))

print(
    '\nUpdate - should be used to update values for (subset of) "existing '
    'index"'
)
df['a'][:-1] += 10
tsds.update({'df': df})
print(tsds.load('df'))

print(
    '\nReplace - should be used when we want to remove parts of index (with '
    'combine we can only add new points in index)'
)
new_df = pd.DataFrame(
    data=0.0,
    index=pd.DatetimeIndex([
        '2017-01-01 02:00:01',
        '2017-01-01 02:00:02',
    ]),
    columns=['a', 'b']
)
tsds.combine({'df': new_df})
print('\nresult using combine:')
print(tsds.load('df'))

tsds.replace({'df': new_df})
print('\nresult using replace:')
print(tsds.load('df'))

print(
    '\nby default the interval covered by replacing data will be used, however a'
    'custom interval can also be used'
)
tsds.replace({'df': new_df}, interval=(new_df.index[0], None))
# end is "infinite"
print(tsds.load('df'))

print('\n\nMULTIPLE SOURCES\n')
print(
    '\nA single TSDataset can have multiple sources that will _share the same '
    'index_. Any of the operations above can be executed for one or many '
    'sources. If only a subset of sources are specified for "index modifying" '
    'operations, the left out sources will also be modified "as expected"...'
)

existing_df = tsds.load('df')
ser = existing_df['a']
tsds.update({'ser': ser})
pprint(tsds.load())  # defaults to all sources,  same as tsds.load(['df', 'ser'])

print('\nextend one source')
extend_ser = pd.Series(
    0.0,
    index=pd.DatetimeIndex([
        ser.index[-1] + pd.Timedelta('1H'),
        ser.index[-1] + pd.Timedelta('100H')
    ]),
    name='a'
)
tsds.extend({'ser': extend_ser})
pprint(tsds.load())

print('\ncombine other source')
tsds.combine({'df': df})
pprint(tsds.load())

print('\n\nLOADING (QUERYING) DATA\n')

print('\nonly index')
index = tsds.load_index()
print(index)

print('\nby interval')
pprint(tsds.load('ser', selection=(index[3], None)))  # None is "open"

print('\nby (any subset of) index')
pprint(tsds.load('ser', selection=index[::2]))  # "step 2"

print('\nby single index points')
pprint(tsds.load('ser', selection=index[1]))
pprint(tsds.load(selection='2017-01-01 01:00:00'))
