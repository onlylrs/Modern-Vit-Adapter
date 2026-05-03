# Copyright (c) Shanghai AI Lab. All rights reserved.
from .checkpoint import load_checkpoint

try:
    from .customized_text import CustomizedTextLoggerHook, PrintLrGroupHook
except KeyError:
    CustomizedTextLoggerHook = None
    PrintLrGroupHook = None
from .early_stopping_hook import EarlyStoppingHook
from .layer_decay_optimizer_constructor import LayerDecayOptimizerConstructor
from .layer_decay_optimizer_constructor_split import LayerDecayOptimizerConstructorSplit
from .maskdino_optimizer_constructor import MaskDINOOptimizerConstructor
from .my_checkpoint import my_load_checkpoint
from .reliable_eval_hook import ReliableEvalHook

__all__ = [
    'LayerDecayOptimizerConstructor',
    'LayerDecayOptimizerConstructorSplit',
    'MaskDINOOptimizerConstructor',
    'EarlyStoppingHook',
    'ReliableEvalHook',
    'load_checkpoint', 'my_load_checkpoint'
]

if CustomizedTextLoggerHook is not None:
    __all__.insert(3, 'CustomizedTextLoggerHook')
if PrintLrGroupHook is not None:
    __all__.insert(4, 'PrintLrGroupHook')
