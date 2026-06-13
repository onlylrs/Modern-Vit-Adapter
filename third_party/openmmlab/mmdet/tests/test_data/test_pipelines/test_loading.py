# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np
import pytest

from mmdet.datasets.pipelines import (LoadImageFromFile, LoadImageFromWebcam,
                                      LoadMultiChannelImageFromFiles)


class TestLoading:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../../data')

    def test_load_img(self):
        results = dict(
            img_prefix=self.data_prefix, img_info=dict(filename='color.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == osp.join(self.data_prefix, 'color.jpg')
        assert results['ori_filename'] == 'color.jpg'
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (288, 512, 3)
        assert results['ori_shape'] == (288, 512, 3)
        assert repr(transform) == transform.__class__.__name__ + \
            "(to_float32=False, color_type='color', " + \
            "file_client_args={'backend': 'disk'}, local_cache_dir=None)"

        # no img_prefix
        results = dict(
            img_prefix=None, img_info=dict(filename='tests/data/color.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == 'tests/data/color.jpg'
        assert results['ori_filename'] == 'tests/data/color.jpg'
        assert results['img'].shape == (288, 512, 3)

        # to_float32
        transform = LoadImageFromFile(to_float32=True)
        results = transform(copy.deepcopy(results))
        assert results['img'].dtype == np.float32

        # gray image
        results = dict(
            img_prefix=self.data_prefix, img_info=dict(filename='gray.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8

        transform = LoadImageFromFile(color_type='unchanged')
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (288, 512)
        assert results['img'].dtype == np.uint8

    def test_load_img_uses_local_cache(self, tmp_path):
        source_dir = tmp_path / 'source'
        source_dir.mkdir()
        cache_dir = tmp_path / 'cache'
        source_file = source_dir / 'cached.jpg'
        mmcv.imwrite(np.full((8, 8, 3), 127, dtype=np.uint8), str(source_file))
        results = dict(
            img_prefix=str(source_dir), img_info=dict(filename='cached.jpg'))
        transform = LoadImageFromFile(local_cache_dir=str(cache_dir))

        first = transform(copy.deepcopy(results))

        class FailingClient:

            def get(self, filename):
                raise RuntimeError('source should not be read after cache fill')

        transform.file_client = FailingClient()
        second = transform(copy.deepcopy(results))

        assert first['img'].shape == (8, 8, 3)
        assert np.array_equal(second['img'], first['img'])
        assert any(cache_dir.iterdir())

    def test_load_img_rejects_cache_for_non_disk_backend(self):
        with pytest.raises(ValueError):
            LoadImageFromFile(
                local_cache_dir='/tmp/cache',
                file_client_args=dict(backend='memcached'))

    def test_load_multi_channel_img(self):
        results = dict(
            img_prefix=self.data_prefix,
            img_info=dict(filename=['color.jpg', 'color.jpg']))
        transform = LoadMultiChannelImageFromFiles()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == [
            osp.join(self.data_prefix, 'color.jpg'),
            osp.join(self.data_prefix, 'color.jpg')
        ]
        assert results['ori_filename'] == ['color.jpg', 'color.jpg']
        assert results['img'].shape == (288, 512, 3, 2)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (288, 512, 3, 2)
        assert results['ori_shape'] == (288, 512, 3, 2)
        assert results['pad_shape'] == (288, 512, 3, 2)
        assert results['scale_factor'] == 1.0
        assert repr(transform) == transform.__class__.__name__ + \
            "(to_float32=False, color_type='unchanged', " + \
            "file_client_args={'backend': 'disk'})"

    def test_load_webcam_img(self):
        img = mmcv.imread(osp.join(self.data_prefix, 'color.jpg'))
        results = dict(img=img)
        transform = LoadImageFromWebcam()
        results = transform(copy.deepcopy(results))
        assert results['filename'] is None
        assert results['ori_filename'] is None
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (288, 512, 3)
        assert results['ori_shape'] == (288, 512, 3)
