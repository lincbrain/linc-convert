"""
The following script is derived from https://github.com/CBI-PITT/holis_tools/blob/main/holis_tools/zyla_spool_reader.py.

Copyright (c) 2021, Alan M Watson

For more information, see the [`holis_tools` LICENCE](https://github.com/CBI-PITT/holis_tools/blob/main/LICENCE)
"""

# stdlib
import configparser
import io
import os
import warnings
from glob import glob

import numpy as np
from scipy.io import loadmat

from linc_convert.utils.math import ceildiv


class SpoolSetInterpreter:

    def __init__(self, spool_set_path, info_file=None):
        if os.path.isfile(spool_set_path) and \
                os.path.splitext(spool_set_path)[-1] == '.zip':
            raise NotImplementedError("Zip storage is not supported.")
            pass
        elif os.path.isdir(spool_set_path):
            self.type = 'dir'
            self.parent = spool_set_path
            self.file_list = glob(os.path.join(spool_set_path, '*'))
            self.spool_set = tuple([os.path.split(x)[-1] for x in self.file_list])
        else:
            assert False, 'The input data structure is not a ZIP file or Directory'

        self._what_spool_format()
        self.spool_files = tuple(self._get_spool_names_in_order())  # In order
        self._get_acquisitionparameters_str()
        self._get_config()
        self._extract_config_values()

        if info_file is not None:
            if os.path.isfile(info_file):
                self._load_info_file(info_file)
            else:
                warnings.warn('ERROR: Info file not found.')
        self.assembled_spool_shape = (self.numDepths,
                                      self.spool_shape[2],
                                      self.spool_shape[0]*len(self.spool_files),
                                      )
    def _load_info_file(self, info_file):
        loaded_info = loadmat(info_file)
        info = loaded_info.get("info", None)
        if info is None:
            warnings.warn("ERROR: 'info' structure not found in the MAT file.")
            return None
        try:
            num_total_frames = int(info['camera'][0][0]['kineticSeriesLength'])
            num_bg_frames = int(info['camera'][0][0]['backgroundFramesNum'])

        except (KeyError, IndexError, ValueError) as e:
            warnings.warn(
                f"ERROR: Unable to extract frame information from info file. {e}")
            return None
        num_frames_per_spool = int(self.config['multiimage']['ImagesPerFile'])
        num_frames_to_load = num_total_frames - num_bg_frames
        num_spool_files_to_load = ceildiv(num_frames_to_load, num_frames_per_spool)
        self.spool_files = self.spool_files[:num_spool_files_to_load]

    def _make_filename_from_spool_set(self, spool_entry):
        return os.path.join(self.parent, spool_entry)

    @property
    def entries(self):
        return self.spool_set

    def _what_spool_format(self):
        if 'Spooled files.sifx' in self.entries:
            format = 'zyla'
        else:
            raise TypeError("Unknown or unsupported spool file format")

        self.format = format

    def _list_spool_files(self):
        return sorted(
            tuple(
                [x for x in self.entries if '0spool.dat' in x]
            )
        )

    def _get_acquisitionparameters_str(self):
        file = self._make_filename_from_spool_set('acquisitionmetadata.ini')
        with open(file, 'rb') as f:
            ini = f.read()
        ini = ini.decode('UTF-8-sig')  # Encoding for acquisitionmetadata.ini
        self.acquisitionparameters_str = ini

    def _get_config(self):
        buf = io.StringIO(self.acquisitionparameters_str)
        self.config = configparser.ConfigParser()
        self.config.read_file(buf)

    def _extract_config_values(self):
        self.acquisition_metadata = {'height': self.config.getint('data', 'AOIHeight'),
                                     'width': self.config.getint('data', 'AOIWidth'),
                                     'stride': self.config.getint('data', 'AOIStride')}
        # ini info
        dtype = self.config.get('data', 'PixelEncoding')

        if dtype == 'Mono16':
            dtype = np.dtype('uint16')
        elif dtype == 'Mono8':
            dtype = np.dtype('uint8')

        self.acquisition_metadata['dtype'] = dtype
        self.dtype = dtype

        self.spool_nbytes = self.config.getint('data', 'ImageSizeBytes')
        self.acquisition_metadata['nbytes'] = self.spool_nbytes

        self.acquisition_metadata['images'] = self.config.getint('multiimage',
                                                                 'ImagesPerFile')

        numDepths = self.acquisition_metadata['height']
        numColumns = self.acquisition_metadata['stride'] // 2
        imageBytes = self.acquisition_metadata['nbytes']
        numFramesPerSpool = self.acquisition_metadata['images']

        if numDepths % 2:  # if there is an odd number of rows ->  KPEDIT - odd rows means 1 less column for some reason
            numRows = numDepths + 1
        else:
            numRows = numDepths + 2

        self.spool_shape = (numFramesPerSpool, numRows, numColumns)
        self.numDepths = numDepths

    def _load_spool_file(self, spool_file_name):
        file = self._make_filename_from_spool_set(spool_file_name)
        print(f'Reading file {spool_file_name}')
        with open(file, 'rb') as f:
            array = np.frombuffer(f.read(), dtype=self.dtype)
        return np.reshape(array, self.spool_shape)

    def __getitem__(self, key):
        if isinstance(key, str):
            assert key in self.spool_files, 'Must be a spool file in self.spool_files OR integer index in self.spool_files'
            return self._load_spool_file(key)
        elif isinstance(key, int):
            return self._load_spool_file(self.spool_files[key])

    def __iter__(self):
        yield from (self[x] for x in range(len(self)))

    def __contains__(self, item):
        return item in self.spool_files

    def __len__(self):
        return len(self.spool_files)

    def _get_spool_names_in_order(self):
        '''
        Spool files are ordered sequentially in the order they were collected 0,1,2,...,201,202,203,...
        but file names are recorded as the reversed number padded to
        10 digits (0000000000,1000000000,20000000000,...,1020000000,2020000000,3020000000,...) + spool.dat
        '''
        spool_files = self._list_spool_files()
        for idx in range(len(spool_files)):
            # Convert index to string, pad with zeros to 10 digits and reverse
            tmp = str(idx).zfill(10)[::-1]
            tmp = f'{tmp}spool.dat'
            if tmp in spool_files:
                yield tmp
            else:
                warnings.warn(f"{tmp} not located in spool directory")

    def assemble(self):
        axis_0_shape = self.spool_shape[0]
        canvas = np.zeros((axis_0_shape * len(self), *self.spool_shape[1:]),
                          dtype=self.dtype)
        for idx, spool_file in enumerate(self):
            start = idx * axis_0_shape
            stop = start + axis_0_shape
            canvas[start:stop] = spool_file
        return canvas

    # this is the modified version for lsm pipeline
    def assemble_cropped(self):
        return self.assemble().transpose(1, 2, 0)[:self.numDepths,:,:]