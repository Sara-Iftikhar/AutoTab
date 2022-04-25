
import numpy as np
from skopt.callbacks import EarlyStopper


class DeltaYStopper(EarlyStopper):

    def __init__(self, min_val_loss, patience):
        super(DeltaYStopper, self).__init__()
        self.min_val_loss = min_val_loss
        self.patience =patience
        self.counter = 0
        self.wait = 0
        self.best = 999999999999
        self.best_iter = 0

    def _criterion(self, result):
        self.counter += 1

        diff = abs(np.nanmin(result.func_vals) - self.best)
        if diff > self.min_val_loss:
            self.best_iter = self.counter
            self.best = np.nanmin(result.func_vals)

        if self.counter - self.best_iter > self.patience:
            print(f'early stopping at {self.counter}')
            return True

        return False

class EarlyStopperMinImp(EarlyStopper):
    """Stops optimization of objective function does not shows improvement
    after first `patience` iterations. """
    def __init__(self, min_improvement, patience):
        super(EarlyStopperMinImp, self).__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.counter = 0

    def _criterion(self, result):
        self.counter += 1
        if self.counter>= self.patience:
            return np.nanmin(result.func_vals) > self.min_improvement

        return None

