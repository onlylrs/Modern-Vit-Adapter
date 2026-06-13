import numpy as np
import pytest
from mmcv import ConfigDict

from mmdet.apis import train as train_api
from mmdet.datasets.builder import build_dataloader


class ToyDataset:

    CLASSES = ('cell',)

    def __init__(self, length=8):
        self.length = length
        self.flag = np.zeros(length, dtype=np.uint8)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return dict(idx=idx)


def test_build_dataloader_uses_performance_options():
    dataloader = build_dataloader(
        ToyDataset(),
        samples_per_gpu=2,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4)

    assert dataloader.pin_memory is True
    assert dataloader.persistent_workers is True
    assert dataloader.prefetch_factor == 4


def test_train_detector_forwards_dataloader_options(monkeypatch):
    calls = []

    def fake_build_dataloader(*args, **kwargs):
        calls.append(kwargs)
        return object()

    class FakeRunner:

        timestamp = None

        def register_training_hooks(self, *args, **kwargs):
            pass

        def register_hook(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

    monkeypatch.setattr(train_api, 'build_dataloader', fake_build_dataloader)
    monkeypatch.setattr(train_api, 'build_optimizer', lambda *args, **kwargs: object())
    monkeypatch.setattr(train_api, 'build_runner', lambda *args, **kwargs: FakeRunner())
    monkeypatch.setattr(train_api, 'MMDataParallel', lambda model, device_ids: model)
    monkeypatch.setattr(train_api, 'EvalHook', lambda *args, **kwargs: object())

    model = pytest.importorskip('torch').nn.Module()
    cfg = ConfigDict(
        log_level='INFO',
        data=ConfigDict(
            samples_per_gpu=2,
            workers_per_gpu=3,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4),
        gpu_ids=[0],
        seed=7,
        runner=ConfigDict(type='EpochBasedRunner', max_epochs=1),
        optimizer=ConfigDict(type='SGD', lr=0.01),
        optimizer_config=ConfigDict(),
        lr_config=ConfigDict(policy='step'),
        checkpoint_config=ConfigDict(interval=1),
        log_config=ConfigDict(interval=1),
        workflow=[('train', 1)],
        work_dir='.',
        load_from=None,
        resume_from=None)

    train_api.train_detector(
        model,
        ToyDataset(),
        cfg,
        distributed=False,
        validate=False)

    assert calls
    assert calls[0]['pin_memory'] is True
    assert calls[0]['persistent_workers'] is True
    assert calls[0]['prefetch_factor'] == 4


def test_train_detector_forwards_val_dataloader_options(monkeypatch):
    calls = []

    def fake_build_dataloader(*args, **kwargs):
        calls.append(kwargs)
        return object()

    class FakeRunner:

        timestamp = None

        def register_training_hooks(self, *args, **kwargs):
            pass

        def register_hook(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

    monkeypatch.setattr(train_api, 'build_dataset', lambda *args, **kwargs: ToyDataset())
    monkeypatch.setattr(train_api, 'build_dataloader', fake_build_dataloader)
    monkeypatch.setattr(train_api, 'build_optimizer', lambda *args, **kwargs: object())
    monkeypatch.setattr(train_api, 'build_runner', lambda *args, **kwargs: FakeRunner())
    monkeypatch.setattr(train_api, 'MMDataParallel', lambda model, device_ids: model)

    model = pytest.importorskip('torch').nn.Module()
    cfg = ConfigDict(
        log_level='INFO',
        data=ConfigDict(
            samples_per_gpu=2,
            workers_per_gpu=3,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            val=ConfigDict(samples_per_gpu=2, pipeline=[])),
        gpu_ids=[0],
        seed=7,
        runner=ConfigDict(type='EpochBasedRunner', max_epochs=1),
        optimizer=ConfigDict(type='SGD', lr=0.01),
        optimizer_config=ConfigDict(),
        lr_config=ConfigDict(policy='step'),
        checkpoint_config=ConfigDict(interval=1),
        log_config=ConfigDict(interval=1),
        workflow=[('train', 1)],
        work_dir='.',
        load_from=None,
        resume_from=None,
        evaluation=ConfigDict(interval=1))

    train_api.train_detector(
        model,
        ToyDataset(),
        cfg,
        distributed=False,
        validate=True)

    assert len(calls) == 2
    assert calls[1]['pin_memory'] is True
    assert calls[1]['persistent_workers'] is True
    assert calls[1]['prefetch_factor'] == 4
