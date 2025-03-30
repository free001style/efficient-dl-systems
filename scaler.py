import torch


class Scaler:
    """
    My custom AMP Scaler.
    """

    def __init__(self, scale_coef=2 ** 16, update_coef=2.0, update_interval=2000, is_dynamic=True):
        """
        Args:
            scale_coef (int): initial scale coefficient.
            update_coef (int): coefficient to update scale coefficient.
            update_interval (int): interval to update scale coefficient.
            is_dynamic (bool): whether to use dynamic scaling (change scale coef) or not.
        """
        self._scale_coef = scale_coef
        self._update_coef = update_coef
        self._is_dynamic = is_dynamic
        self._update_interval = update_interval
        self._is_nan_inf = False
        self._step = 0

    def scale(self, loss):
        """
        Multiply loss by scale coefficient.
        """
        return loss * self._scale_coef

    def _unscale(self, optimizer):
        """
        Unscale gradient.
        """
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad /= self._scale_coef

    def step(self, optimizer):
        """
        Unscale gradient and if there isn't Nan or inf, do step. Otherwise do only zero_grad.
        """
        self._unscale(optimizer)
        self._check_nan_inf(optimizer)
        if self._is_nan_inf:
            optimizer.zero_grad()
        else:
            optimizer.step()

    def _check_nan_inf(self, optimizer):
        """
        Check Nan and inf in gradient.
        """
        self._is_nan_inf = False
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        self._is_nan_inf = True
                        return

    def update(self):
        """
        If dynamic is True update scale coefficient, otherwise do nothing.
        """
        if self._is_dynamic:
            if self._is_nan_inf:
                self._scale_coef /= self._update_coef
            elif self._step % self._update_interval == 0:
                self._scale_coef *= self._update_coef
                self._step += 1
