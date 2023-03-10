
__all__ = ["Callbacks", "data_to_csv", "data_to_h5",
           "EarlyStopperMinImp", "DeltaYStopper"]


import numpy as np
import pandas as pd

try:
    from skopt.callbacks import EarlyStopper
except ModuleNotFoundError:
    class EarlyStopper(object): pass


class Callbacks(object):
    """callbacks to be executed."""

    def on_build_begin(self, model, **model_kwargs)->None:
        """called before ``build`` method of parent and loop"""
        return

    def on_build_end(self, model, **model_kwargs)->None:
        """called at the end ``build`` method of parent and loop"""
        return

    def on_fit_begin(self, x=None, y=None, validation_data=None)->None:
        """called before ``fit`` method of parent loop. This callback does not run
        when cross validation is used. For that consider using ``on_cross_val_begin``."""
        return

    def on_fit_end(self, x=None, y=None, validation_data=None)->None:
        """called at the end ``fit`` method of parent loop.  This callback does not run
        when cross validation is used. For that consider using ``on_cross_val_end``."""

    def on_eval_begin(self, model, iter_num=None, x=None, y=None, validation_data=None)->None:
        """called before ``evaluate`` method of parent loop"""
        return

    def on_eval_end(self, model, iter_num=None, x=None, y=None, validation_data=None)->None:
        """called at the end ``evaluate`` method of parent loop"""
        return

    def on_cross_val_begin(self, model, iter_num=None, x=None, y=None, validation_data=None)->None:
        """called at the start of cross validation."""
        return

    def on_cross_val_end(self, model, iter_num=None, x=None, y=None, validation_data=None)->None:
        """called at the end of cross validation."""
        return


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
    """
    Stops optimization if objective function does not show improvement
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


def data_to_h5(filepath, x, y, val_x, val_y, test_x, test_y):
    import h5py

    f = h5py.File(filepath, mode='w')

    _save_data_to_hdf5('training_data', x, y, f)

    _save_data_to_hdf5('validation_data', val_x, val_y, f)

    _save_data_to_hdf5('test_data', test_x, test_y, f)

    f.close()
    return


def _save_data_to_hdf5(data_type, x, y, f):
    """Saves one data_type in h5py. data_type is string indicating whether
    it is training, validation or test data."""



    assert x is not None
    group_name = f.create_group(data_type)

    for name, val in zip(['x', 'y'], [x, y]):

        param_dset = group_name.create_dataset(name, val.shape, dtype=val.dtype)
        if not val.shape:
            # scalar
            param_dset[()] = val
        else:
            param_dset[:] = val
    return


def data_to_csv(filepath: str,
                all_features: list,
                x, y):
    if x is None:
        pd.DataFrame().to_csv(filepath)
    else:
        pd.DataFrame(np.concatenate([x, y], axis=1), columns=all_features).to_csv(filepath)
    return