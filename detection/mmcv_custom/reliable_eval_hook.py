import os.path as osp

import torch.distributed as dist
from mmdet.core import DistEvalHook, EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class _ReliableEvalMixin:
    def __init__(self, *args, early_stop_metric=None, early_stop_patience=None, early_stop_min_delta=0.0, early_stop_rule='greater', **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop_metric = early_stop_metric
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_rule = early_stop_rule
        self._best_score = None
        self._bad_epochs = 0

    def _is_improved(self, score):
        if self._best_score is None:
            return True
        if self.early_stop_rule == 'greater':
            return score > self._best_score + self.early_stop_min_delta
        return score < self._best_score - self.early_stop_min_delta

    def _update_early_stop(self, runner):
        metric_name = self.early_stop_metric or self.key_indicator
        if metric_name is None or self.early_stop_patience is None:
            return
        metrics = runner.log_buffer.output
        if metric_name not in metrics:
            return
        score = metrics[metric_name]
        if self._is_improved(score):
            self._best_score = score
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
            if self._bad_epochs >= self.early_stop_patience:
                runner.logger.info(
                    f'Early stopping triggered on {metric_name}: '
                    f'no improvement for {self.early_stop_patience} eval rounds.'
                )
                if hasattr(runner, 'train_loop') and hasattr(runner.train_loop, 'stop_training'):
                    runner.train_loop.stop_training = True
                else:
                    runner._max_epochs = runner.epoch + 1


class ReliableEvalHook(_ReliableEvalMixin, EvalHook):
    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return
        from mmdet.apis import single_gpu_test

        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)
        self._update_early_stop(runner)


class ReliableDistEvalHook(_ReliableEvalMixin, DistEvalHook):
    def _do_evaluate(self, runner):
        if self.broadcast_bn_buffer:
            model = runner.model
            for _, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)

        if runner.rank == 0:
            self.latest_results = results
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
            self._update_early_stop(runner)
