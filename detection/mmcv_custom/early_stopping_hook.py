from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    def __init__(self, monitor='bbox_mAP', rule='greater', patience=30, min_delta=0.0):
        if rule not in ('greater', 'less'):
            raise ValueError(f'unsupported rule: {rule}')
        self.monitor = monitor
        self.rule = rule
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.bad_epochs = 0

    def _is_improved(self, score):
        if self.best_score is None:
            return True
        if self.rule == 'greater':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def _update_from_metrics(self, runner, metrics):
        hook_msgs = runner.meta.setdefault('hook_msgs', {}) if runner.meta is not None else {}
        if self.monitor not in metrics:
            return
        score = metrics[self.monitor]
        if self._is_improved(score):
            self.best_score = score
            self.bad_epochs = 0
            hook_msgs[f'early_stop_best_{self.monitor}'] = score
            return
        self.bad_epochs += 1
        hook_msgs[f'early_stop_bad_epochs_{self.monitor}'] = self.bad_epochs
        if self.bad_epochs >= self.patience:
            runner.logger.info(
                f'Early stopping triggered on {self.monitor}: '
                f'no improvement for {self.patience} epochs.'
            )
            if hasattr(runner, 'train_loop') and hasattr(runner.train_loop, 'stop_training'):
                runner.train_loop.stop_training = True
            else:
                runner._max_epochs = runner.epoch + 1

    def after_val_epoch(self, runner):
        metrics = getattr(runner.log_buffer, 'output', {})
        self._update_from_metrics(runner, metrics)
