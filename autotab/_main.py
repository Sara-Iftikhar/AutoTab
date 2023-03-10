
import os
import gc
import json
import sys
import time
import math
import types
import shutil
import inspect
import warnings
from typing import List
from typing import Union
from typing import Tuple
from typing import Callable
from collections import OrderedDict
from collections import defaultdict
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SeqMetrics import RegressionMetrics
from SeqMetrics import ClassificationMetrics

from easy_mpl import plot
from easy_mpl import bar_chart
from easy_mpl import taylor_plot
from easy_mpl import dumbbell_plot
from easy_mpl import circular_bar_plot
from easy_mpl import parallel_coordinates

import ai4water
from ai4water import Model

from ai4water.backend import plotly, hyperopt, skopt

from ai4water.models import MLP
from ai4water.models import CNN
from ai4water.models import LSTM
from ai4water.models import TFT
from ai4water.models import TCN
from ai4water.models import CNNLSTM
from ai4water.models import LSTMAutoEncoder

from ai4water.utils.utils import jsonize
from ai4water._optimize import make_space
from ai4water.preprocessing import DataSet
from ai4water.utils.utils import make_model
from ai4water.utils.utils import dateandtime_now
from ai4water.utils.utils import find_best_weight

from ai4water.experiments.utils import dl_space
from ai4water.experiments.utils import regression_space
from ai4water.experiments.utils import classification_space

from ai4water.hyperopt import Real
from ai4water.hyperopt import Integer
from ai4water.hyperopt import HyperOpt
from ai4water.hyperopt import Categorical
from ai4water.hyperopt.utils import loss_histogram
from ai4water.hyperopt.utils import to_skopt_space
from ai4water.hyperopt.utils import plot_convergence
from ai4water.hyperopt.utils import plot_convergences
from ai4water.hyperopt.utils import plot_hyperparameters

from .utils import Callbacks, data_to_h5, data_to_csv

try:
    import wandb
except (ModuleNotFoundError, ImportError):
    wandb = None

assert ai4water.__version__ >= "1.06", f"""
    Your current ai4water version is {ai4water.__version__}.
    Please upgrade your ai4water version to at least 1.06 using
    'pip install --upgrade ai4water'
    """


# TODO's
# custom model which is installed/not installed

# in order to unify the use of metrics
Metrics = {
'regression': lambda t, p, multiclass=False, **kwargs: RegressionMetrics(t, p, **kwargs),
'classification': lambda t, p, multiclass=False, **kwargs: ClassificationMetrics(t, p,
        multiclass=multiclass, **kwargs)
}

METRICS_KWARGS = {
    'accuracy': {},
    "cross_entropy": {},
    'f1_score': {"average": "macro"},
    "precision": {"average": "macro"},
    "recall": {"average": "macro"},
    "specificity": {"average": "macro"},
}

DL_MODELS = {
    "MLP": MLP,
    "LSTM":LSTM,
    "CNN":CNN,
    "CNNLSTM":CNNLSTM,
    "TFT":TFT,
    "TCN":TCN,
    "LSTMAutoEncoder":LSTMAutoEncoder
}

SEP = os.sep

DEFAULT_TRANSFORMATIONS = [
    "minmax", "center", "scale", "zscore",
    "box-cox", "yeo-johnson",  "quantile", "quantile_normal",  "robust",
    "log", "log2", "log10", "sqrt",
    "pareto", "vast",
    "none",
              ]

METRIC_TYPES = {
    "r2": "max",
    "nse": "max",
    "r2_score": "max",
    "kge": "max",
    'log_nse': 'max',
    "corr_coeff": "max",
    'accuracy': "max",
    'f1_score': 'max',
    "mse": "min",
    "rmse": "min",
    "rmsle": "min",
    "mape": "min",
    "nrmse": "min",
    "pbias": "min",
    "bias": "min",
    "med_seq_error": "min",
    "mae": "min",
}

METRIC_NAMES = {
    'r2': "$R^2$",
    "r2_score": "$R^2$ Score"
}


class AttributeNotSetYet:
    def __init__(self):
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        raise AttributeError("""
        The pipeline has not been fitted yet. 
        You must first call .fit method to get {}
        """.format(self.name))

    def __set_name__(self, owner, name):
        self.name = name


class PipelineMixin(object):
    # following attributes are set duirng .fit call
    # so they must not be accessed before calling .fit
    # Following makes sure that a proper error is raised for the user
    # if he/she tries to access them before calling .fit first
    parent_prefix_ = AttributeNotSetYet()
    metrics_ = AttributeNotSetYet()
    parent_iter_ = AttributeNotSetYet()
    child_iter_ = AttributeNotSetYet()
    val_scores_ = AttributeNotSetYet()
    metrics_best_ = AttributeNotSetYet()
    parent_seeds_ = AttributeNotSetYet()
    child_seeds_ = AttributeNotSetYet()
    child_val_scores_ = AttributeNotSetYet()
    baseline_results_ = AttributeNotSetYet()
    start_time_ = AttributeNotSetYet()
    parent_suggestions_ = AttributeNotSetYet()
    _parent_suggestions_ = AttributeNotSetYet()
    callbacks_ = AttributeNotSetYet()
    taylor_plot_data_ = AttributeNotSetYet()
    child_callbacks_ = AttributeNotSetYet()
    CHILD_PREFIX_ = AttributeNotSetYet()

    def __init__(
            self,
            input_features,
            output_features,
            mode,
            category,
    ):
        assert mode in ("regression", "classification"), f"""
        {mode} not allowed as mode. It must be either regression or classification.
        """
        self.mode = mode

        assert category in ("DL", "ML")
        self.category = category

        self.input_features = input_features

        if isinstance(output_features, str):
            output_features = [output_features]
        self.output_features = output_features

        self._transformations_methods = {
            "quantile": {},
            "quantile_normal": {},
            "minmax": {},
            "center": {},
            "scale": {},
            "zscore": {},
            "box-cox": {'treat_negatives': True, 'replace_zeros': True},
            "yeo-johnson": {},
            "robust": {},
            "log": {'treat_negatives': True, 'replace_zeros': True},
            "log2": {'treat_negatives': True, 'replace_zeros': True},
            "log10": {'treat_negatives': True, 'replace_zeros': True},
            "sqrt": {'treat_negatives': True},
            "vast": {},
            "pareto": {},
        }

        self.feature_transformations = {}
        for feat in self.all_features:
            default_feat_trans = self._transformations_methods
            if self.input_transformations is not None and feat in self.input_features:
                # It is possible that the
                # user has specified `input_transformtions` argument. In that case
                # use only those from feat_trans (default) which are in
                # `input_transformations`
                default_feat_trans = {
                    k:v for k,v in default_feat_trans.items() if k in self.input_transformations}

            self.feature_transformations[feat] = default_feat_trans

        self._pp_plots = []
        if self.mode == "regression":
            self._pp_plots =  ["regression", "prediction", "murphy",
                               "residual", "edf"]

    @property
    def all_features(self)->list:
        return self.input_features + self.output_features


class OptimizePipeline(PipelineMixin):
    """
    optimizes model/estimator, its hyperparameters and preprocessing
    operation to be performed on input and output features. It consists of two
    hpo loops. The parent or outer loop optimizes preprocessing/feature engineering,
    feature selection and model selection while the child hpo loop optimizes
    hyperparmeters of child hpo loop.

    Attributes
    ----------

    - metrics_
        a pandas DataFrame of shape (parent_iterations, len(monitor)) which contains
        values of metrics being monitored at each parent iteration.

    - val_scores_
        a 1d numpy array of length equal to parent_iterations which contains value
        of evaluation metric at each parent iteration.

    - parent_suggestions_:
        an ordered dictionary of suggestions to the parent objective function
        during parent hpo loop

    - child_val_scores_:
        a numpy array of shape (parent_iterations, child_iterations) containing
        value of eval_metric at all child hpo loops

    - optimizer_
        an instance of ai4water.hyperopt.HyperOpt [1]_ for parent optimization

    - models
        a list of models being considered for optimization

    - model_space
        a dictionary which contains parameter space for each model

    Example
    -------
        >>> from autotab import OptimizePipeline
        >>> from ai4water.datasets import busan_beach
        >>> data = busan_beach()
        >>> input_features = data.columns.tolist()[0:-1]
        >>> output_features = data.columns.tolist()[-1:]
        >>> pl = OptimizePipeline(input_features=input_features,
        >>>                       output_features=output_features,
        >>>                       inputs_to_transform=input_features)
        >>> results = pl.fit(data=data)

    Note
    ----
    This optimization always solves a minimization problem even if the val_metric
    is $R^2$.

    .. [1] https://ai4water.readthedocs.io/en/latest/hpo.html#hyperopt
    """

    def __init__(
            self,
            input_features,
            output_features,
            inputs_to_transform: Union[list, dict] = None,
            input_transformations: Union[list, dict] = None,  # todo: if we exclude vast, still appear in space
            outputs_to_transform=None,
            output_transformations: Union[list, ] = None,
            models: list = None,
            parent_iterations: int = 100,
            child_iterations: int = 25,
            parent_algorithm: str = "bayes",
            child_algorithm: str = "bayes",
            eval_metric: str = None,
            cv_parent_hpo: bool = None,
            cv_child_hpo: bool = None,
            monitor: Union[list, str] = None,
            mode: str = "regression",
            num_classes:int = None,
            category:str = "ML",
            prefix: str = None,
            wandb_config: dict = None,
            **model_kwargs
    ):
        """
        initializes the class

        Parameters
        ----------
            input_features : list
                names of input features
            output_features : str
                names of output features
            inputs_to_transform : list/dict, optional, (default=None)
                Input features on which feature engineering/transformation is to
                be applied. By default all input features are considered. If you
                want to apply a single transformation on a group of input features,
                then pass this as a dictionary. This is helpful if the input data
                consists of hundred or thousands of input features. If None (default)
                transformations will be applied on all input features. If you don't
                want to apply any transformation on any input feature, pass an empty
                list.
            input_transformations : list, dict
                The transformations to be considered for input features. Default
                is None, in which case all input features are considered.

                If list, then it will be the names of transformations to be considered
                for all input features. By default following transformations are
                considered

                    - ``minmax``  rescale from 0 to 1
                    - ``center``    center the data by subtracting mean from it
                    - ``scale``     scale the data by dividing it with its standard deviation
                    - ``zscore``    first performs centering and then scaling
                    - ``box-cox``
                    - ``yeo-johnson``
                    - ``quantile``
                    - ``quantile_normal``
                    - ``robust``
                    - ``log``  natural logarithm
                    - ``log2``  log with base 2
                    - ``log10``  log with base 10
                    - ``sqrt``    square root

                The user can however, specify list of transformations to be considered
                for each input feature. In such a case, this argument must be a
                dictionary whose keys are names of input features and values are
                list of transformations.

            outputs_to_transform : list, optional
                Output features on which feature engineering/transformation is to
                be applied. If None, then transformations on outputs are not applied.
            output_transformations : Optional (default=None)
                The transformations to be considered for outputs/targets. The user
                can consider any transformation as given for ``input_transformations``
            models : list, optional
                The models/algorithms to consider during optimization. If not given, then all
                available models from sklearn, xgboost, catboost and lgbm are
                considered. For neural networks, following 6 model types are
                considered by default

                    - MLP [1]_   multi layer perceptron
                    - CNN [2]_  1D convolution neural network
                    - LSTM [3]_ Long short term memory network
                    - CNNLSTM [4]_  CNN-> LSTM
                    - LSTMAutoEncoder [5]_ LSTM based autoencoder
                    - TCN [6]_ Temporal convolution networks
                    - TFT [7]_ Temporal fusion Transformer

                However, in such cases, the ``category`` must be ``DL``.

            parent_iterations : int, optional (default=100)
                Number of iterations for parent optimization loop
            child_iterations : int, optional
                Number of iterations for child optimization loop. If set to 0,
                the child hpo loop is not run which means the hyperparameters
                of the model are not optimized. You can customize number of hpo
                iterations for each model by making using of :meth: `change_child_iterations`
                method.
            parent_algorithm : str, optional
                Algorithm for optimization of parent optimization
            child_algorithm : str, optional
                Algorithm for optimization of child optimization
            eval_metric : str, optional
                Validation metric to calculate val_score in objective function.
                The parent and child hpo loop optimizes/improves this metric. This metric is
                calculated on validation data. If cross validation is performed then
                this metric is calculated using cross validation.
            cv_parent_hpo : bool, optional (default=False)
                Whether we want to apply cross validation in parent hpo loop or not?.
                If given, the parent hpo loop will optimize the cross validation score.
                The  model is fitted on whole training data (training+validation) after
                cross validation and the metrics printed (other than parent_val_metric)
                are calculated on the based the updated model i.e. the one fitted on
                whole training (training + validation) data.
            cv_child_hpo : bool, optional (default=False)
                Whether we want to apply cross validation in child hpo loop or not?.
                If False, then val_score will be calculated on validation data.
                The type of cross validator used is taken from model.config['cross_validator']
            monitor : Union[str, list], optional, (default=None)
                Names of performance metrics to monitor in parent hpo loop. If None,
                then R2 is monitored for regression and accuracy for classification.
            mode : str, optional (default="regression")
                whether this is a ``regression`` problem or ``classification``
            num_classes : int, optional (default=None)
                number of classes, only relevant if mode=="classification".
            category : str, optional (default="DL")
                either "DL" or "ML". If DL, the pipeline is optimized for neural networks.
            wandb_config : dict
                The keyword arguments to initiate wand.init() as dictionary. It is
                only valid if wandb package is installed.  Default value is None,
                which means, wandb will not be utilized. For simplest case, pass
                a dictionary with `project` as key.
                >>> dict(project="my_project")
                The user must however login wandb
                before!
            **model_kwargs :
                any additional key word arguments for ai4water's Model

        References
        ----------
        .. [1] https://ai4water.readthedocs.io/en/latest/models/models.html#ai4water.models.MLP

        .. [2] https://ai4water.readthedocs.io/en/latest/models/models.html#ai4water.models.CNN

        .. [3] https://ai4water.readthedocs.io/en/latest/models/models.html#ai4water.models.LSTM

        .. [4] https://ai4water.readthedocs.io/en/latest/models/models.html#ai4water.models.CNNLSTM

        .. [5] https://ai4water.readthedocs.io/en/latest/models/models.html#ai4water.models.LSTMAutoEncoder

        .. [6] https://ai4water.readthedocs.io/en/latest/models/models.html#ai4water.models.TCN

        .. [7] https://ai4water.readthedocs.io/en/latest/models/models.html#ai4water.models.TFT

        """

        # None means all inputs are to be considered.
        if inputs_to_transform is None:
            inputs_to_transform = input_features

        if isinstance(inputs_to_transform, dict):
            # apply same transformation on group of inputs
            self._groups = inputs_to_transform
            self.inputs_to_transform = list(inputs_to_transform.keys())
            self.groups_present = True
        else:
            self.groups_present = False
            # apply unique transformation on each input feature
            self._groups = {inp:[inp] for inp in inputs_to_transform}
            self.inputs_to_transform = inputs_to_transform

        self.input_transformations = input_transformations

        self.output_transformations = output_transformations or DEFAULT_TRANSFORMATIONS

        super(OptimizePipeline, self).__init__(input_features,
                                               output_features,
                                               mode,
                                               category)

        if self.groups_present:
            self.feature_transformations = {
                k:self._transformations_methods for k in inputs_to_transform.keys()}

        self.num_classes = num_classes

        self.models = models
        if models is None:
            if mode == "regression":
                if category == "ML":
                    self.models = list(regression_space(2).keys())
                else:
                    self.models = list(dl_space(2).keys())
            else:
                if category == "ML":
                    self.models = list(classification_space(2).keys())
                else:
                    self.models = list(dl_space(2).keys())

        elif isinstance(models, list):
            assert all([isinstance(obj, str) for obj in models])
            if len(set(models)) != len(models):
                raise ValueError(f"models contain repeating values. \n{models}")

            if self.category == "DL":
                assert all([model in self.models for model in models]), f"""
        Only following deep learning models can be considered {DL_MODELS.keys()}
                """

        self.parent_iterations = parent_iterations
        self.child_iterations = child_iterations
        # for internal use, we keep child_iter for each model
        self._child_iters = {model: child_iterations for model in self.models}
        self.parent_algorithm = parent_algorithm
        self.child_algorithm = child_algorithm

        if eval_metric is None:
            if self.mode == "regression":
                eval_metric = "mse"
            else:
                eval_metric = "accuracy"
        self.eval_metric = eval_metric
        self.cv_parent_hpo = cv_parent_hpo
        self.cv_child_hpo = cv_child_hpo

        for arg in ['model', 'x_transformation', 'y_transformation']:
            if arg in model_kwargs:
                raise ValueError(f"argument {arg} not allowed")
        model_kwargs['input_features'] = input_features
        model_kwargs['output_features'] = output_features
        # if the user has supplied the mode, we should put it in model_kwargs
        model_kwargs['mode'] = self.mode
        self.model_kwargs = model_kwargs

        self.outputs_to_transform = outputs_to_transform
        if outputs_to_transform is not None:
            if isinstance(outputs_to_transform, str):
                outputs_to_transform = [outputs_to_transform]
            self._groups.update({outp: [outp] for outp in outputs_to_transform})

        # self.seed = None
        if monitor is None:
            if mode == "regression":
                monitor = ['r2']
            else:
                monitor = ['accuracy']

        if isinstance(monitor, str):
            monitor = [monitor]

        # evaluation_metric is monitored by default
        if eval_metric not in monitor:
            monitor.append(eval_metric)

        assert isinstance(monitor, list)
        self.monitor = monitor

        if self.category == "ML":
            if self.mode == "regression":
                space = regression_space(num_samples=10)
            else:
                space = classification_space(num_samples=10)
        else:
            space = dl_space(num_samples=10)

        # model_space contains just those models which are being considered
        self.model_space = {}
        for mod, mod_sp in space.items():
            if mod in self.models:
                self.model_space[mod] = mod_sp

        self._optimize_model = True
        self._model = None

        if self.outputs_to_transform is None:
            self._features_to_transform = self.inputs_to_transform
        else:
            self._features_to_transform = self.inputs_to_transform + self.outputs_to_transform

        self.batch_space = []
        self.lr_space = []
        if category == "DL":
            self.batch_space = [Categorical([8, 16, 32, 64], name="batch_size")]
            self.lr_space = [Real(1e-5, 0.05, num_samples=10, name="lr")]

        if wandb_config is None:
            self.use_wb = False
        else:
            self.use_wb = True
        self.wandb_config = wandb_config

        # information about transformations which are to be modified
        self._tr_modifications = {}

    def get_np_errstate(self):
        default = {'divide':'ignore','over':'ignore','under':'ignore','invalid':'ignore'}
        return getattr(self, 'np_errstate', default)

    def set_np_errstate(self, value:dict):
        return setattr(self, 'np_errstate', value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)->None:
        """
        Even if an error is encountered during ``fit``, the results, report and config
        must be saved.

        """
        if exc_type:
            print(f"{exc_type} occured, version info is below: \n {self._version_info()}")

        self.exc_type_ = exc_type
        self.exc_val_ = exc_val

        self.save_results()

        self.report()

        self._save_config()
        return

    @property
    def num_ins(self):
        return len(self.input_features)

    @property
    def input_shape(self):
        if self.category == "DL":
            if "ts_args" in self.model_kwargs:
                return self.model_kwargs['ts_args']['lookback'], self.num_ins
            else:
                return self.num_ins,
        return
    @property
    def outputs_to_transform(self):
        return self._out_to_transform

    @outputs_to_transform.setter
    def outputs_to_transform(self, x):
        if x:
            if isinstance(x, str):
                x = [x]
            assert isinstance(x, list)
            for i in x:
                assert i in self.output_features
        self._out_to_transform = x

    def maybe_make_path(self):
        _path = os.path.join(os.getcwd(), "results", self.parent_prefix_)
        if not os.path.exists(_path):
            os.makedirs(_path)
        return _path

    @property
    def use_wb(self):
        return self._use_wb

    @use_wb.setter
    def use_wb(self, x):
        self._use_wb = x

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, x):
        self._mode = x

    @property
    def Metrics(self):
        return Metrics[self.mode]

    @property
    def num_outputs(self):
        if self.mode == "classification":
            return self.num_classes
        else:
            return len(self.output_features)

    def classes_(self, y:np.ndarray):

        if self.mode == "classification":
            if self.category == "ML":
                return self._model.classes_
            return np.unique(y[~np.isnan(y)])

        raise NotImplementedError

    def _save_config(self):
        if not hasattr(self, 'path'):
            return
        cpath = os.path.join(self.path, "config.json")
        config = self.config()
        with open(cpath, 'w') as fp:
            json.dump(jsonize(config), fp, indent=4)
        return

    def update_model_space(self, space: dict) -> None:
        """updates or changes the search space of an already existing model

        Parameters
        ---------
            space : dict
                a dictionary whose keys are names of models and values are parameter
                space for that model.
        Returns
        -------
            None

        Example
        -------
            >>> pl = OptimizePipeline(...)
            >>> rf_space = {'max_depth': [5,10, 15, 20],
            >>>          'n_models': [5,10, 15, 20]}
            >>> pl.update_model_space({"RandomForestRegressor": rf_space})

            Similarly we can also update for a deep learning model as below

            >>> pl = OptimizePipeline(input_features=["tide_cm"], output_features="tetx_coppml",
            ...       category="DL")
            >>> pl.update_model_space({"MLP": {
            ...     "units": Integer(low=8, high=128, prior='uniform', transform='identity', name='units'),
            ...     "activation": Categorical(["relu", "elu", "tanh", "sigmoid"], name="activation"),
            ...     "num_layers": Integer(low=1, high=5, name="num_layers")
            ...         }})
            we can confirm it by printing the model space
            >>> pl.model_space['MLP']
        """
        for model, space in space.items():
            if model not in self.model_space:
                raise ValueError(f"{model} is not valid because it is not being considered.")
            space = to_skopt_space(space)
            self.model_space[model] = {'param_space': [s for s in space]}
        return

    def add_dl_model(
            self,
            model: Callable,
            space:Union[list, Real, Categorical, Integer]
    )->None:
        """adds a deep learning model to be considered.

        Parameters
        ----------
            model : callable
                the model to be added
            space : list
                the search space of the model
        """
        if isinstance(model, types.FunctionType):
            model_config = model()
            assert isinstance(model_config, dict), f"model does not require valid model config {model_config}"
            assert len(model_config) == 1, f"model config has length of 1 {len(model_config)}"
            assert 'layers' in model_config, f"model config must have 'layers' key {model_config.keys()}"

            if not isinstance(space, list):
                space = [space]

            model_name = model.__name__
            space = to_skopt_space(space)
            self.models.append(model_name)
            DL_MODELS[model_name] = model
            self.model_space[model_name] = {'param_space': space}
            self._child_iters[model_name] = self.child_iterations
        else:
            raise NotImplementedError

    def add_model(
            self,
            model: dict
    ) -> None:
        """adds a new model which will be considered during optimization.

        Parameters
        ----------
            model : dict
                a dictionary of length 1 whose value should also be a dictionary
                of parameter space for that model

        Example
        -------
            >>> pl = OptimizePipeline(...)
            >>> pl.add_model({"XGBRegressor": {"n_estimators": [100, 200,300, 400, 500]}})

        """
        msg = """{} is already present. If you want to change its space, please 
              consider using 'change_model_space' function.
              """
        for model_name, model_space in model.items():
            assert model_name not in self.model_space, msg.format(model_name)
            assert model_name not in self.models, msg.format(model_name)
            assert model_name not in self._child_iters, msg.format(model_name)

            model_space = to_skopt_space(model_space)
            self.model_space[model_name] = {'param_space': model_space}
            self.models.append(model_name)
            self._child_iters[model_name] = self.child_iterations

        return

    def remove_transformation(
            self,
            transformation:Union[str, list],
            feature:Union[str, list] = None
    )->None:
        """Removes one or more transformation from being considered. This function
        modifies the ``feature_transformations`` attribute of the class.

        Parameters
        ----------
            transformation : str/list
                the name/names of transformation to be removed.
            feature : str/list, optional (default=None)
                name of feature for which the transformation should not be considered.
                If not given, the transformation will be removed from all the input features.

        Returns
        -------
        None

        Examples
        --------
            >>> pl = OptimizePipeline(...)
            ... # remove box-cox transformation altogether
            >>> pl.remove_transformation('box-cox')
            ... # remove multiple transformations
            >>> pl.remove_transformation(['yeo-johnson', 'log'])
            ... # remove a transformation for a certain feature
            >>> pl.remove_transformation('log2', 'tide_cm')
            ... # remove a transformation for more than one features
            >>> pl.remove_transformation('log10', ['tide_cm', 'wat_temp_c'])
        """
        if isinstance(transformation, str):
            transformation = [transformation]

        if feature is None:
            feature = self.input_features
            # so that space does not have the transformation/s in it
            for trans in transformation:
                DEFAULT_TRANSFORMATIONS.remove(trans)
        elif isinstance(feature, str):
            feature = [feature]

        assert isinstance(transformation, list)
        assert isinstance(feature, list)

        # removing the transformations from feature_transformations
        for trans in transformation:
            for feat in feature:
                feat_trans = self.feature_transformations[feat].copy()
                feat_trans.pop(trans)
                self.feature_transformations[feat] = feat_trans

        # we need to remove these modifications from space as well
        # so that they are not suggested by the algorithm
        for feat in feature:
            tr_for_feat = self.feature_transformations[feat]
            self._tr_modifications[feat] = list(tr_for_feat.keys())

        return

    def remove_model(self, models: Union[str, list]) -> None:
        """
        removes an model/models from being considered. The follwoing
        attributes are updated.

            - models
            - model_space
            - _child_iters

        Parameters
        ----------
            models : list, str
                name or names of model to be removed.

        Example
        -------
            >>> pl = OptimizePipeline(...)
            ... # If we don't want 'ExtraTreeRegressor' to be considered
            >>> pl.remove_model("ExtraTreeRegressor")
        """
        if isinstance(models, str):
            models = [models]

        for model in models:
            self.models.remove(model)
            self.model_space.pop(model)
            self._child_iters.pop(model)

        return

    def change_child_iteration(self, model: dict):
        """
        We may want to change the child hpo iterations for one or more models.
        For example we may want to run only 10 iterations for LinearRegression but 40
        iterations for XGBRegressor. In such a case we can use this function to
        modify child hpo iterations for one or more models. The iterations for all
        the remaining models will remain same as defined by the user at the start.
        This method updated `_child_iters` dictionary

        Parameters
        ----------
            model : dict
                a dictionary whose keys are names of models and values are number
                of iterations for that model during child hpo
        Example
        -------
            >>> pl = OptimizePipeline(...)
            >>> pl.change_child_iteration({"XGBRegressor": 10})
            ... # If we want to change iterations for more than one models
            >>> pl.change_child_iteration(({"XGBRegressor": 30,
            ...                             "RandomForestRegressor": 20}))
        """
        for _model, _iter in model.items():
            if _model not in self._child_iters:
                raise ValueError(f"{_model} is not a valid model name")
            self._child_iters[_model] = _iter
        return

    def space(self) -> list:
        """makes the parameter space for parent hpo"""

        append = {}
        y_categories = []

        if self.input_transformations is None:
            x_categories = DEFAULT_TRANSFORMATIONS
        elif isinstance(self.input_transformations, list):
            x_categories = self.input_transformations
        else:
            x_categories = DEFAULT_TRANSFORMATIONS
            assert isinstance(self.input_transformations, dict)

            for feature, transformation in self.input_transformations.items():
                assert isinstance(transformation, list)
                append[feature] = transformation

        if self.outputs_to_transform:
            # if the user has provided name of any outupt feature
            # on feature transformation is to be applied

            if isinstance(self.output_transformations, list):
                assert all([t in DEFAULT_TRANSFORMATIONS for t in self.output_transformations]), f"""
                transformations must be one of {DEFAULT_TRANSFORMATIONS}"""

                for out in self.output_features:
                    append[out] = self.output_transformations
                y_categories = self.output_transformations

            else:
                assert isinstance(self.output_transformations, dict)
                for out_feature, y_transformations in self.output_transformations.items():

                    assert out_feature in self.output_features
                    assert isinstance(y_transformations, list)
                    assert all(
                        [t in DEFAULT_TRANSFORMATIONS for t in self.output_transformations]), f"""
                        transformations must be one of {DEFAULT_TRANSFORMATIONS}"""
                    append[out_feature] = y_transformations
                y_categories = list(self.output_transformations.values())

        # append will contain modifications that need to be applied for both x_spacea nd y_space
        append.update(self._tr_modifications)

        sp = make_space(self.inputs_to_transform, categories=x_categories,
                          append={k:v for k,v in append.items() if k in self.input_features})

        if self.outputs_to_transform:
            sp += make_space(self.outputs_to_transform, categories=y_categories,
                              append={k:v for k,v in append.items() if k in self.output_features})

        if len(self.models)>1:
            algos = Categorical(self.models, name="model")
            sp = sp + [algos]
        else:
            self._optimize_model = False
            self._model = self.models[0]

        return sp

    def change_batch_size_space(self, space:list, low=None, high=None):
        """changes the value of class attribute ``batch_space``.
        It should be used after pipeline initialization and before calling ``fit`` method.
        """
        assert self.category == "DL"
        if isinstance(space, list):
            self.batch_space = [Categorical(space, name="lr")]
        else:
            self.batch_space = [Integer(low, high, name="lr", num_samples=10)]
        return

    def change_lr_space(self, space:list, low=None, high=None):
        """changes the value of class attribute ``lr_space``.
        It should be used after pipeline initialization and before calling ``fit`` method.
        """
        assert self.category == "DL"
        if isinstance(space, list):
            self.lr_space = [Categorical(space, name="lr")]
        else:
            self.lr_space = [Real(low, high, name="lr", num_samples=10)]
        return

    def change_transformation_behavior(
            self,
            transformation:str,
            new_behavior:dict,
            features:Union[list, str] = None
    )->None:
        """
        change the behvior of a transformation i.e. the way it is applied.
        If ``features`` is not not given, it will modify the behavior of transformation
        for all features. This function modifies the ``feature_transformations``
        attribute of the class.

        Parameters
        ----------
            transformation : str
                The name of transformation whose behavior is to be modified.
            new_behavior : dict
                key, word arguments which determine the new behavior of Transformation.
                These key,word arguments are given to the specifified transformation
                when it is initialized.
            features : str/list, optional (default=None)
                The name or names of features for which the behavior should be modified.
                If not given, the changed behavior of transformation will apply to all
                input features.

        Returns
        -------
        None

        Example
        -------
        >>> from autotab import OptimizePipeline
        >>> from ai4water.datasets import busan_beach

        >>> data = busan_beach()
        >>> input_features=data.columns.tolist()[0:-1]
        >>> output_features=data.columns.tolist()[-1:]
        >>> pl = OptimizePipeline(
        ...                    input_features=input_features,
        ...                    output_features=output_features
        ...                     )
        >>> pl.change_transformation_behavior('yeo-johnson', {'pre_center': True}, 'wind_dir_deg')
        ... # we can change behavior behavior for multiple features as well
        >>> pl.change_transformation_behavior('yeo-johnson', {'pre_center': True},
        ...                                   ['air_p_hpa',  'mslp_hpa'])
        """
        assert transformation in DEFAULT_TRANSFORMATIONS

        assert not self.groups_present  # todo

        if features is None:
            features = self.all_features
        elif isinstance(features, str):
            features = [features]

        assert all([feature in self.all_features for feature in features])

        assert isinstance(new_behavior, dict)

        for feature in features:
            self.feature_transformations[feature][transformation] = new_behavior

        return

    @property
    def max_child_iters(self) -> int:
        # the number of child hpo iterations can be different based upon models
        # this property calculates maximum child iterations
        return max(self._child_iters.values())

    def training_data(self, *args, **kwargs)->Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def validation_data(self, *args, **kwargs)->Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def test_data(self, *args, **kwargs)->Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _save_data(self, *args, **kwargs)->None:
        raise NotImplementedError

    def reset(self):
        # called at the start of fit method

        # a new path is created every time we call .fit
        self.parent_prefix_ = f"pipeline_opt_{dateandtime_now()}"
        self.path = self.maybe_make_path()

        self.metrics_ = pd.DataFrame(
            np.full((self.parent_iterations, len(self.monitor)), np.nan),
            columns=self.monitor_names
        )

        self.parent_iter_ = 0
        self.child_iter_ = 0
        self.val_scores_ = np.full(self.parent_iterations, np.nan)

        metrics_best = np.full((self.parent_iterations, len(self.monitor)), np.nan)
        self.metrics_best_ = pd.DataFrame(metrics_best, columns=self.monitor_names)

        self.parent_seeds_ = np.random.randint(0, 10000, self.parent_iterations)
        self.child_seeds_ = np.random.randint(0, 10000, self.max_child_iters)

        # each row indicates parent iteration, column indicates child iteration
        self.child_val_scores_ = np.full((self.parent_iterations,
                                          self.max_child_iters),
                                         np.nan)
        self.start_time_ = time.asctime()

        self.parent_suggestions_ = OrderedDict()
        self._parent_suggestions_ = OrderedDict()

        # create container to store data for Taylor plot
        # It will be populated during postprocessing
        self.taylor_plot_data_ = {
            'simulations': {"test": {}},
            'observations': {"test": None}
        }

        self.baseline_results_ = None

        self._save_config()  # will also make path if it does not already exists

        self._print_header()

        self.callbacks_ = None

        # TODO, currently there are no callbacks for child iteration
        self.child_callbacks_ = [Callbacks()]

        self._wb_init()
        return

    def _wb_init(self):
        """initializes the wandb"""
        if self.use_wb:

            if self.child_iterations>0:
                text = self.child_algorithm
            else:
                text = "no_hpo"

            def_tags = [self.category, self.mode, self.parent_algorithm,
                        f"{len(self.models)}_models", f"{self.num_ins}_inputs",
                        self.eval_metric_name]

            if self.mode == "classification":
                def_tags += [f"{self.num_classes}_classes"]

            init_config = dict(
                config = {sp.name: sp.categories for sp in self.space()},
                notes = f"{self.mode} with {self.category}",
                entity = "entity",
                tags = def_tags,
                name =  f"{self.parent_algorithm}_{text}_{os.path.basename(self.path)[-15:]}"
            )

            init_config.update(self.wandb_config)

            self.wb_run_ = wandb.init(**init_config)

        return

    def _print_header(self):
        # prints the first line on console
        formatter = "{:<5} {:<18} " + "{:<15} " * (len(self.monitor))
        print(formatter.format(
            "Iter",
            self.eval_metric_name,
            *self.monitor_names)
        )

        return

    @property
    def eval_metric_name(self)->str:
        if isinstance(self.eval_metric, str):
            return self.eval_metric
        elif callable(self.eval_metric):
            return self.eval_metric.__name__
        else:
            return str(self.eval_metric)

    @property
    def monitor_names(self)->List[str]:
        names = []
        for pm in self.monitor:
            if callable(pm):
                names.append(pm.__name__)
            else:
                names.append(str(pm))
        return names

    def fit(
            self,
            x:np.ndarray = None,
            y:np.ndarray = None,
            data: pd.DataFrame = None,
            validation_data:Tuple[np.ndarray, np.ndarray] = None,
            previous_results:dict = None,
            process_results:bool = True,
            callbacks:Union[Callbacks, List[Callbacks]] = None
    ) -> "ai4water.hyperopt.HyperOpt":
        """
        Optimizes the pipeline for the given data.
        Either
            - only x,y should be given (validation data will be taken from x and y based upon `val_fraction` argument
            - or x,y and validation_data should be given
            - or only data should be given (training and validation data will be taken from data based upon `train_fraction` and `val_fraction` arguments`)
        every other combination of x,y, data and validation_data will raise error

        Note
        ----
        If test_data is not to be extracted/seprated from x,y/data then you must set
        `train_fraction` to 1.0. Please check
        `this tutorial <https://ai4water.readthedocs.io/projects/Examples/en/latest/_notebooks/model/data_splitting.html>`_
        for more on data splitting.

        Parameters
        ----------
            x : np.ndarray
                input data for training + validation + test. If your ``x`` does not
                contain test portion, set ``train_fraction`` to 1.0 during
                initializtion of OptimizePipeline class.
            y : np.ndarray
                output/target/label for training data. It must of same length as ``x``.
            data :
                A pandas dataframe which contains input (x) and output (y) features
                Only required if ``x`` and ``y`` are not given. The training and validation
                data will be extracted from this data.
            validation_data : tuple
                validation data on which pipeline is optimized. Only required if ``data``
                is not given.
            previous_results : dict, optional (default=None)
                path of file which contains xy values.
            process_results : bool, optional (default=True)
                Wether to perform postprocessing of optimization of results or not.
            callbacks : list, optional (default=None)
                list of callbacks to run

        Returns
        --------
            an instance of ai4water.hyperopt.HyperOpt class which is used for
            optimization.
        """

        train_x, train_y, val_x, val_y, _, _ = self.verify_data(x, y, data, validation_data)

        self.reset()

        skopt_cbs = self._verify_cbs(callbacks)

        optimizer = HyperOpt(
            self.parent_algorithm,
            param_space=self.space(),
            objective_fn=self.parent_objective,
            num_iterations=self.parent_iterations,
            opt_path=self.path,
            verbosity = 0,
            process_results=False,
        )

        if previous_results is not None:
            optimizer.add_previous_results(previous_results)

        self.save_results()

        self.report()

        self._save_config()

        res = optimizer.fit(x=train_x, y=train_y, validation_data = (val_x, val_y))

        if process_results:
            self._proces_hpo_results(optimizer)

        self._wb_finish()

        setattr(self, 'optimizer_', optimizer)

        return res

    def _wb_finish(self):
        """prepares the logs and puts them on wandb"""
        if self.use_wb and self.parent_iter_ > 0:

            # 🐝 Create a wandb Table to log parent suppestions and metrics
            df = pd.DataFrame(
                [list(val.values()) for val in self._parent_suggestions_.values()],
            columns=list(self._parent_suggestions_[0].keys())
            )

            df['iterations'] = self.parent_suggestions_.keys()
            df['seeds'] = self.parent_seeds_

            df = pd.concat([df, self.metrics_], axis=1)

            df['hyperparas'] = [list(val['model'].values())[0] for val in self.parent_suggestions_.values()]

            table = wandb.Table(data=df, allow_mixed_types=True,
                                columns=df.columns.tolist())

            self.wb_run_.log({"result": table})

            if self.child_iter_>0:
                table = wandb.Table(
                    data=pd.DataFrame(self.child_val_scores_),
                    allow_mixed_types=True)

                self.wb_run_.log({"child_hpo_results": table})

            self.wb_run_.notes = self.report(False)

            cols = self.metrics_best_.columns
            # find last nan value in each column
            indices = self.metrics_best_.apply(pd.Series.last_valid_index)
            vals = [self.metrics_best_[col].iloc[index] for col, index in zip(cols, indices)]
            summary_metrics = {metric:val for metric, val in zip(cols, vals)}
            self.wb_run_.summary = summary_metrics

            self.wb_run_.finish()
        return

    def _verify_cbs(self, callbacks=None):
        if callbacks is None:
            callbacks = [Callbacks()]

        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        assert isinstance(callbacks, list), f"callbacks of type {type(callbacks)} not allowed"

        from skopt.callbacks import EarlyStopper

        skopt_cbs = []
        native_cbs = []
        for cbk in callbacks:
            if isinstance(cbk, EarlyStopper):
                skopt_cbs.append(cbk)
            elif isinstance(cbk, Callbacks):
                native_cbs.append(cbk)
            else:
                raise ValueError(f"""
            Each callback must be an instance of Callback class but you provided a 
            callback of type {type(cbk)}""")

        setattr(self, 'callbacks_', native_cbs)

        return skopt_cbs

    def _proces_hpo_results(self, optimizer):
        """
        postprocessing of hpo results
        """

        from optuna.visualization import plot_contour

        optimizer.save_iterations_as_xy()

        plt.close('all')
        optimizer.plot_parallel_coords(show=False)

        # deep learning related results
        if self.category == "DL":
            plot_convergences(
                optimizer.opt_path,
                what='val_loss',
                ylabel='Validation MSE')
            plot_convergences(
                optimizer.opt_path,
                what='loss',
                ylabel='MSE',
                leg_pos="upper right")

        getattr(optimizer, "_plot_edf")()

        # distributions/historgrams of explored hyperparameters
        getattr(optimizer, "_plot_distributions")(show=False)

        # convergence plot,
        #if sr.x_iters is not None and self.backend != "skopt": # todo
        plt.close('all')
        getattr(optimizer, "_plot_convergence")(show=False)
        if self.use_wb:
            fig = plt.gcf()
            self.wb_run_.log({"convergence": fig})

        plt.close('all')
        # plot of hyperparameter space as explored by the optimizer
        if optimizer.backend != 'skopt' and len(self.space()) < 20 and skopt is not None:
            getattr(optimizer, "_plot_evaluations")()

        hpo_imp = True
        if len(optimizer.best_paras(True))>1:
            plt.close('all')
            try:
                optimizer.plot_importance()
                plt.close('all')
                optimizer.plot_importance(plot_type="bar", show=False)
            except (RuntimeError, AttributeError, ValueError):
                hpo_imp = False
                warnings.warn(f"Error encountered during fanova calculation")

        if hpo_imp and self.use_wb:
            fig = plt.gcf()
            self.wb_run_.log({"importance": fig})

        if optimizer.backend == 'hyperopt':
            loss_histogram([y for y in optimizer.trials.losses()],
                           save=True,
                           fname=os.path.join(optimizer.opt_path, "loss_histogram.png")
                           )
            plot_hyperparameters(
                optimizer.trials,
                fname=os.path.join(optimizer.opt_path, "hyperparameters.png"),
                save=True)

        if plotly is not None:

            if optimizer.backend == 'optuna':

                fig = plot_contour(optimizer.study)
                plotly.offline.plot(fig, filename=os.path.join(optimizer.opt_path, 'contours.html'),
                                    auto_open=False)
        return

    def parent_objective(
            self,
            x=None,
            y=None,
            validation_data=None,
            **suggestions
    ) -> float:
        """
        objective function for parent hpo loop.

        This objective function is to optimize transformations for each input
        feature and the model.

        Parameters
        ----------
            x :
            y :
            validation_data :
            **suggestions :
                key word arguments consisting of suggested transformation for each
                input feature and the model to use
        """

        self.CHILD_PREFIX_ = f"{self.parent_iter_}_{dateandtime_now()}"
        # self.seed = np.random.randint(0, 10000, 1).item()

        if self._optimize_model:
            model = suggestions['model']
        else:
            model = self._model

        x_trnas, y_trans = self._cook_transformations(suggestions)

        if self._child_iters[model]>0:
            # optimize the hyperparas of model using child objective
            opt_paras = self.optimize_model_paras(
                x,
                y,
                validation_data,
                model,
                x_transformations=x_trnas,
                y_transformations=y_trans or None
            )
        else:
            opt_paras = {}

        kwargs = {}
        if self.category == "DL":
            for arg in ['lr', 'batch_size']:
                if arg in opt_paras:
                    kwargs[arg] = opt_paras.pop(arg)
            model_config = DL_MODELS[model](mode=self.mode,
                                            input_shape=self.input_shape,
                                            num_outputs=self.num_outputs,
                                            **opt_paras)
        else:
            model_config = {model: opt_paras}

        # fit the model with optimized hyperparameters and suggested transformations
        _model = self.build_model(
            model=model_config,
            x_transformation=x_trnas,
            y_transformation=y_trans,
            prefix=f"{self.parent_prefix_}{SEP}{self.CHILD_PREFIX_}",
            **kwargs
        )

        # set the global seed. This is only for internal use so that results
        # become more reproducible
        # when the model is built again
        _model.seed_everything(int(self.parent_seeds_[self.parent_iter_]))

        self.parent_suggestions_[self.parent_iter_] = {
            # 'seed': self.seed,
            'x_transformation': x_trnas,
            'y_transformation': y_trans,
            'model': {model: opt_paras},
            'path': _model.path
        }

        self._parent_suggestions_[self.parent_iter_] = suggestions

        val_score = self._fit_and_eval(
            x,
            y,
            validation_data,
            model=_model,
            cross_validate=self.cv_parent_hpo,
            eval_metrics=True,
            callbacks=self.callbacks_
        )

        self.val_scores_[self.parent_iter_] = val_score

        _val_score = val_score
        if self.parent_iter_>0:
            if np.less(val_score, np.nanmin(self.val_scores_[:self.parent_iter_])):
                _val_score = val_score
            else:
                _val_score = ''

        # print the metrics being monitored
        # we fill the nan in metrics_best_ with '' so that it does not gen printed
        formatter = "{:<5} {:<18.3} " + "{:<15.7} " * (len(self.monitor))
        print(formatter.format(
            self.parent_iter_,
            _val_score,
            *self.metrics_best_.loc[self.parent_iter_].fillna('').values.tolist())
        )

        self._wb_log()

        self.parent_iter_ += 1

        return val_score

    def _wb_log(self):
        """logs performance metrics being monitored"""
        if self.use_wb:
            self.wb_run_.log(self.metrics_.loc[self.parent_iter_].to_dict())
        return

    def optimize_model_paras(
            self,
            x,
            y,
            validation_data,
            model: str,
            x_transformations: list,
            y_transformations: list
    ) -> dict:
        """optimizes hyperparameters of a model"""

        def child_objective(lr=0.001, batch_size=32, **suggestions):
            """objective function for optimization of model parameters"""

            if self.category == "DL":
                model_config = DL_MODELS[model](mode=self.mode,
                                                input_shape=self.input_shape,
                                                num_outputs=self.num_outputs,
                                                **suggestions)
            else:
                model_config = {model: suggestions}

            # build child model
            _model = self.build_model(
                model=model_config,
                x_transformation=x_transformations,
                y_transformation=y_transformations,
                prefix=f"{self.parent_prefix_}{SEP}{self.CHILD_PREFIX_}",
                lr=float(lr),
                batch_size=int(batch_size)
            )

            _model.seed_everything(int(self.child_seeds_[self.child_iter_]))

            val_score = self._fit_and_eval(
                x,
                y,
                validation_data,
                model=_model,
                cross_validate=self.cv_child_hpo,
                callbacks=self.child_callbacks_
            )

            # populate all child val scores
            self.child_val_scores_[self.parent_iter_-1, self.child_iter_] = val_score

            self.child_iter_ += 1

            return val_score

        # make space
        child_space = self.model_space[model]['param_space'] + self.batch_space + self.lr_space

        # before starting child hpo, reset iteration counter
        setattr(self, "child_iter_", 0)

        optimizer = HyperOpt(
            self.child_algorithm,
            objective_fn=child_objective,
            num_iterations=self._child_iters[model],
            param_space=child_space,
            verbosity=0,
            process_results=False,
            opt_path=os.path.join(self.path, self.CHILD_PREFIX_),
        )

        optimizer.fit()

        # free memory if possible
        gc.collect()

        # return the optimized parameters
        return optimizer.best_paras()

    def _cook_transformations(self, suggestions):
        """prepares the transformation keyword argument based upon
        suggestions"""

        # container for transformations for all features
        x_transformations = []
        y_transformations = []

        for feature, method in suggestions.items():

            if feature in self._features_to_transform:
                if method != "none":  # don't do anything with this feature
                    # get the relevant transformation for this feature
                    t_config = {"method": method, "features": self._groups[feature]}

                    # some preprocessing is required for log based transformations
                    t_config.update(self.feature_transformations[feature][method])

                    if feature in self.inputs_to_transform:
                        x_transformations.append(t_config)
                    else:
                        y_transformations.append(t_config)

        return x_transformations, y_transformations

    def build_model(
            self,
            model,
            x_transformation,
            y_transformation,
            prefix: Union[str, None] = None,
            verbosity:int = 0,
            batch_size:int = 32,
            lr:float = 0.001,
            path = None,
    ) -> Model:
        """
        build the ai4water Model. When overwriting this method, the user
        must return an instance of ai4water's Model_ class.

        Parameters
        ----------
            model :
                anything which can be fed to AI4Water's Model class.
            x_transformation :
                transformation on input data
            y_transformation :
                transformation on output data
            prefix :
            verbosity : int
                level of output
            batch_size : int
                only used when category is "DL".
            lr :
                only used when category is "DL"
            path : str
                path where to save the model

        .. Model:
            https://ai4water.readthedocs.io/en/master/model.html#ai4water._main.BaseModel
        """

        for cbk in self.callbacks_:
            getattr(cbk, 'on_build_begin')(model, **self.model_kwargs)

        model = Model(
            model=model,
            verbosity=verbosity,
            val_metric=self.eval_metric,
            x_transformation=x_transformation,
            y_transformation=y_transformation,
            # seed=self.seed,
            prefix=prefix,
            batch_size=int(batch_size),
            lr=float(lr),
            path = path,
            **self.model_kwargs
        )

        for cbk in self.callbacks_:
            getattr(cbk, 'on_build_end')(model, **self.model_kwargs)

        return model

    def build_model_from_config(
            self,
            cpath:str
    )->Model:
        """builds ai4water model from config.
        If the user overwrites `py:meth:build_model`, then the user must also
        overwrite this function. Otherwise post-processing will not work

        Parameters
        ----------
            cpath : str
                complete path of config file

        Returns
        -------
        Model
            an instance of `:py:class:ai4water.Model` class
        """

        return Model.from_config_file(cpath)

    def _cv_and_eval(
            self,
            x,
            y,
            validation_data,
            model:ai4water.Model,
            callbacks:list,
    )->float:
        """performs cross validation and evaluates the model"""
        for cbk in callbacks:
            getattr(cbk, 'on_cross_val_begin')(
                model, self.parent_iter_, x=x, y=y, validation_data=validation_data)

        val_scores = model.cross_val_score(
            *combine_train_val(x, y, validation_data=validation_data),
            scoring=[self.eval_metric] + self.monitor,
            refit=False
        )

        for cbk in callbacks:
            getattr(cbk, 'on_cross_val_end')(
                model=model,
                iter_num=self.parent_iter_,
                x=x,
                y=y,
                validation_data=validation_data)

        val_score = val_scores.pop(0)

        for k, pm_val in zip(self.monitor, val_scores):

            self.metrics_.at[self.parent_iter_, k] = pm_val

            func = compare_func1(METRIC_TYPES[k])

            pm_until_this_iter = self.metrics_best_.loc[:self.parent_iter_, k]

            if pm_until_this_iter.isna().sum() == pm_until_this_iter.size:
                best_so_far = fill_val(METRIC_TYPES[k], np.nan)
            else:
                best_so_far = func(self.metrics_best_.loc[:self.parent_iter_, k])

                best_so_far = fill_val(METRIC_TYPES[k], best_so_far)

            func = compare_func(METRIC_TYPES[k])
            if func(pm_val, best_so_far):
                self.metrics_best_.at[self.parent_iter_, k] = pm_val

        return val_score

    def __fit_and_eval(
            self,
            train_x,
            train_y,
            validation_data,
            model:ai4water.Model,
            eval_metrics:bool,
            callbacks:list,
    )->float:
        """fits the model and evaluates"""
        for cbk in callbacks:
            getattr(cbk, 'on_fit_begin')(
                x=train_x, y=train_y, validation_data=validation_data)

        # train the model and evaluate it to calculate val_score

        if self.category == "DL":
            # DL models employ early stopping based upon performance on validation data
            # without monitoring validation loss, training is useless because
            # we can't tell whether the fitted model is overfitted or not.
            model.fit(x=train_x, y=train_y, validation_data=validation_data)
        else:
            model.fit(x=train_x, y=train_y)

        for cbk in callbacks:
            getattr(cbk, 'on_fit_end')(x=train_x, y=train_y, validation_data=validation_data)

        #  evaluate the model to calculate val_score
        return self._eval_model_manually(
            model,
            data=validation_data,
            metric=self.eval_metric,
            metric_name = self.eval_metric_name,
            callbacks=self.callbacks_,
            eval_metrics=eval_metrics
        )

    def _fit_and_eval(
            self,
            train_x,
            train_y,
            validation_data,
            model: ai4water.Model,
            callbacks: list,
            cross_validate:bool = False,
            eval_metrics:bool = False,
    ) -> float:
        """fits the model and evaluates it and returns the score.
        This method also populates on entry/row in `:py:attribute:metrics_` dataframe.

        callbacks : list
            list of callbacks, which can be for parent or child
        """
        if cross_validate:
            return self._cv_and_eval(train_x, train_y, validation_data, model, callbacks)
        else:
            return self.__fit_and_eval(train_x, train_y, validation_data, model, eval_metrics, callbacks)

    def get_best_metric(
            self,
            metric_name: str
    ) -> float:
        """
        returns the best value of a particular performance metric.
        The metric must be recorded i.e. must be given as `monitor` argument.

        Parameters
        ----------
        metric_name : str
            Name of performance metric

        Returns
        -------
        float
            the best value of performance metric achieved
        """
        if metric_name not in self.monitor_names:
            raise MetricNotMonitored(metric_name, self.monitor)

        if METRIC_TYPES[metric_name] == "min":
            return np.nanmin(self.metrics_[metric_name]).item()
        else:
            return np.nanmax(self.metrics_[metric_name]).item()

    def get_best_metric_iteration(
            self,
            metric_name: str = None
    ) -> int:
        """returns iteration of the best value of a particular performance metric.

        Parameters
        ----------
            metric_name : str, optional
                The metric must be recorded i.e. must be given as `monitor` argument.
                If not given, then evaluation metric is used.

        Returns
        -------
        int
            the parent iteration on which metric was obtained.
        """

        metric_name = metric_name or self.eval_metric

        if metric_name not in self.monitor_names:
            raise MetricNotMonitored(metric_name, self.monitor_names)

        if METRIC_TYPES[metric_name] == "min":
            idx = np.nanargmin(self.metrics_[metric_name].values)
        else:
            idx = np.nanargmax(self.metrics_[metric_name].values)

        return int(idx)

    def get_best_pipeline_by_metric(
            self,
            metric_name: str = None
    ) -> dict:
        """returns the best pipeline with respect to a particular performance
        metric.

        Parameters
        ---------
            metric_name : str, optional
                The name of metric whose best value is to be retrieved. The metric
                must be recorded i.e. must be given as `monitor`.
        Returns
        -------
        dict
            a dictionary with following keys

                - ``path`` path where the model is saved on disk
                - ``model`` name of model
                - ``x_transformations``  transformations for the input data
                - ``y_transformations`` transformations for the target data
                - ``iter_num`` iteration number on which this pipeline was achieved
        """

        metric_name = metric_name or self.eval_metric

        iter_num = self.get_best_metric_iteration(metric_name)

        pipeline = self.parent_suggestions_[iter_num]

        pipeline['iter_num'] = iter_num

        return pipeline

    def get_best_pipeline_by_model(
            self,
            model_name: str,
            metric_name: str = None
    ) -> tuple:
        """returns the best pipeline with respect to a particular model and
        performance metric. The metric must be recorded i.e. must be given as
        `monitor` argument.

        Parameters
        ----------
            model_name : str
                The name of model for which best pipeline is to be found. The `best`
                is defined by ``metric_name``.
            metric_name : str, optional
                The name of metric with respect to which the best model is to
                be retrieved. If not given, the best model is defined by the
                evaluation metric.

        Returns
        -------
        tuple
            a tuple of length two

            - first value is a float which represents the value of
                metric
            - second value is a dictionary of pipeline with four keys

                ``x_transformation``
                ``y_transformation``
                ``model``
                ``path``
                ``iter_num``
        """

        metric_name = metric_name or self.eval_metric_name

        # checks if the given metric is a valid metric or not
        if metric_name not in self.monitor_names:
            raise MetricNotMonitored(metric_name, self.monitor_names)

        # initialize an empty dictionary to store model parameters
        model_container = {}

        for iter_num, iter_suggestions in self.parent_suggestions_.items():
            # iter_suggestion is a dictionary and it contains four keys
            model = iter_suggestions['model']

            # model is dictionary, whose key is the model_name and values
            # are model configuration

            if model_name in model:
                # find out the metric value at iter_num
                metric_val = self.metrics_.loc[int(iter_num), metric_name]
                metric_val = round(metric_val, 4)

                iter_suggestions['iter_num'] = iter_num
                model_container[metric_val] = iter_suggestions

        if len(model_container) == 0:
            raise ModelNotUsedError(model_name)

        # sorting the container w.r.t given metric_name
        sorted_container = sorted(model_container.items())

        return sorted_container[-1]

    def baseline_results(
            self,
            x = None,
            y = None,
            data = None,
            test_data = None,
            fit_on_all_train_data:bool = True
    ) -> tuple:
        """
        Returns default performance of all models.

        It runs all the models with their default parameters and without
        any x and y transformation. These results can be considered as
        baseline results and can be compared with optimized model's results.
        The model is trained on 'training'+'validation' data.

        Parameters
        ----------
            x :
                the input data for training
            y :
                the target data for training
            data :
                raw unprepared and unprocessed data from which x,y pairs for both training
                and test will be prepared. It is only required if x, y are not provided.
            test_data :
                a tuple/list of length 2 whose first element is x and second value is y.
                The is the data on which the performance of optimized pipeline will be
                calculated. This should only be given if ``data`` argument is not given.
            fit_on_all_train_data : bool, optional (default=True)
                If true, the model is trained on (training+validation) data.
                This is based on supposition that the data is split into
                training, validation and test sets. The optimization of
                pipeline was performed on validation data. But now, we
                are training the model on all available training data
                which is (training + validation) data. If False, then
                model is trained only on training data.

        Returns
        -------
        tuple
            a tuple of two dictionaries.
            - a dictionary of val_scores on test data for each model
            - a dictionary of metrics being monitored for  each model on test data.
        """

        train_x, train_y, val_x, val_y, test_x, test_y = self.verify_data(
            x,
            y,
            data,
            validation_data=None,
            test_data=test_data
        )

        if self.baseline_results_ is None:

            if self.callbacks_ is None:
                setattr(self, "callbacks_",[Callbacks()])

            val_scores = {}
            metrics = {}

            for model_name in self.models:

                model_config = model_name
                if self.category == "DL":
                    model_config = DL_MODELS[model_name](
                        mode=self.mode,
                        input_shape=self.input_shape,
                        num_outputs=self.num_outputs)

                # build model
                model = self.build_model(
                    model=model_config,
                    path = os.path.join(self.path, "baselines", f"{model_name}_{dateandtime_now()}"),
                    x_transformation=None,
                    y_transformation=None
                )

                if fit_on_all_train_data:
                    model.fit(*combine_train_val(train_x, train_y, (val_x, val_y)))
                else:
                    model.fit(x=train_x, y=train_y)

                t, p = model.predict(test_x, test_y, return_true=True)

                errors = self.Metrics(t, p, multiclass=model.is_multiclass_)
                val_scores[model_name] = getattr(errors, self.eval_metric)(
                    **METRICS_KWARGS.get(self.eval_metric, {}))

                _metrics = {}
                for m, mn in zip(self.monitor, self.monitor_names):
                    if callable(m):
                        _metrics[mn] = m(t,p)
                    else:
                        _metrics[mn] = getattr(errors, m)(
                            **METRICS_KWARGS.get(m, {}))
                metrics[model_name] = _metrics

            results = {
                'val_scores': val_scores,
                'metrics': metrics
            }

            setattr(self, 'baseline_results_', results)

            fpath = os.path.join(self.path, "baselines", "results.json")
            with open(fpath, 'w') as fp:
                json.dump(results, fp, sort_keys=True, indent=4)
        else:
            val_scores, metrics = self.baseline_results_.values()

        return val_scores, metrics

    def dumbbell_plot(
            self,
            x = None,
            y = None,
            data = None,
            test_data = None,
            metric_name: str = None,
            fit_on_all_train_data:bool = True,
            lower_limit: Union[int, float] = None,
            upper_limit: Union[int, float] = None,
            figsize: tuple = None,
            show: bool = True,
            save: bool = True
    ) -> plt.Axes:
        """
        Generate Dumbbell_ plot as comparison of baseline models with
        optimized models. Note that this command will train all the considered models,
        so this can be expensive.

        Parameters
        ----------
            x :
                the input data for training
            y :
                the target data for training
            data :
                raw unprepared and unprocessed data from which x,y pairs for both
                training and test will be prepared. It is only required if x, y
                are not provided.
            test_data :
                a tuple/list of length 2 whose first element is x and second value
                is y. The is the data on which the performance of optimized pipeline
                 will be calculated. This should only be given if ``data`` argument
                 is not given.
            metric_name: str
                The name of metric with respect to which the models have
                to be compared. If not given, the evaluation metric is used.
            fit_on_all_train_data : bool, optional (default=True)
                If true, the model is trained on (training+validation) data.
                This is based on supposition that the data is split into
                training, validation and test sets. The optimization of
                pipeline was performed on validation data. But now, we
                are training the model on all available training data
                which is (training + validation) data. If False, then
                model is trained only on training data.
            lower_limit : float/int, optional (default=None)
                clip the values below this value. Set this value to None to avoid
                clipping.
            upper_limit : float/int, optional (default=None)
                clip the values above this value
            figsize: tuple
                If given, plot will be generated of this size.
            show : bool
                whether to show the plot or not
            save
                By default True. If False, function will not save the
                resultant plot in current working directory.

        Returns
        -------
        plt.Axes
            matplotlib axes object which can be used for further processing

        Examples
        --------
        >>> from autotab import OptimizePipeline
        >>> from ai4water.datasets import busan_beach
        >>> data = busan_beach()
        >>> input_features = data.columns.tolist()[0:-1]
        >>> output_features = data.columns.tolist()[-1:]
        >>> pl = OptimizePipeline(input_features=input_features,
        >>>                       output_features=output_features)
        >>> results = pl.fit(data=data)
        ... # compare models with respect to evaluation metric
        >>> pl.dumbbell_plot(data=data)
        ... # compare the models by also plotting bias value
        >>> pl.dumbbell_plot(data=data, metric_name="r2_score")
        ... # get the matplotlb axes for further processing
        >>> ax = pl.dumbbell_plot(data=data, metric_name="r2_score",
        ...       lower_limit=0.0, show=False)

        .. _Dumbbell:
            https://easy-mpl.readthedocs.io/en/latest/plots.html#easy_mpl.dumbbell_plot
        """
        # todo:
        #  baseline_results returns performance on test data while
        #  get_best_pipelien_by_model returns performance on validation data
        #  todo: they are not comparable

        metric_name = metric_name or self.eval_metric

        _, bl_results = self.baseline_results(
            x=x,
            y=y,
            data=data,
            fit_on_all_train_data=fit_on_all_train_data,
            test_data=test_data
        )
        plt.close('all')

        bl_models = {}
        for k, v in bl_results.items():
            bl_models[k] = v[metric_name]

        optimized_models = {}

        for model_name in self.models:
            try:
                metric_val, _ = self.get_best_pipeline_by_model(
                    model_name, metric_name)
            # the model was not used so consider the baseline result as optimizied
            # result
            except ModelNotUsedError:
                metric_val = bl_models[model_name]

            optimized_models[model_name] = metric_val

        combined = defaultdict(list)
        for d in (bl_models, optimized_models):
            for key, value in d.items():
                combined[key].append(value)

        df = pd.DataFrame.from_dict(combined).transpose()
        df = df.reset_index()
        df.columns = ['models', 'baseline', 'optimized']

        labels = _shred_suffix(df['models'].tolist())

        df.to_csv(os.path.join(self.path, f"dumbbell_{metric_name}_data.csv"))

        if lower_limit:
            idx = df['baseline'] < lower_limit
            df.loc[idx, 'baseline'] = lower_limit

        if upper_limit:
            idx = df['optimized'] > upper_limit
            df.loc[idx, 'optimized'] = upper_limit

        fig, ax = plt.subplots(figsize=figsize)
        ax, _, _ = dumbbell_plot(df['baseline'],
                           df['optimized'],
                           labels=labels,
                           show=False,
                           ax_kws=dict(xlabel=metric_name,
                           ylabel="Models"),
                           ax=ax
                           )

        fpath = os.path.join(self.path, f"dumbbell_{metric_name}")
        if save:
            plt.savefig(fpath, dpi=300, bbox_inches='tight')
        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def taylor_plot(
            self,
            x = None,
            y = None,
            data = None,
            test_data = None,
            fit_on_all_train_data: bool = True,
            plot_bias: bool = True,
            figsize: tuple = None,
            show: bool = True,
            save: bool = True,
            verbosity:int = 0,
            **kwargs
    ) -> plt.Figure:
        """
        makes Taylor_'s plot using the best version of each model.
        The number of models in taylor plot will be equal to the number
        of models which have been considered by the model.

        Parameters
        ----------
            x :
                the input data for training
            y :
                the target data for training
            data :
                raw unprepared and unprocessed data from which x,y pairs for both
                training and test will be prepared. It is only required if x, y
                are not provided.
            test_data :
                a tuple/list of length 2 whose first element is x and second value
                is y. The is the data on which the performance of optimized pipeline
                will be calculated. This should only be given if ``data`` argument
                is not given.
            fit_on_all_train_data : bool, optional (default=True)
                If true, the model is trained on (training+validation) data.
                This is based on supposition that the data is split into
                training, validation and test sets. The optimization of
                pipeline was performed on validation data. But now, we
                are training the model on all available training data
                which is (training + validation) data. If False, then
                model is trained only on training data.
            plot_bias : bool, optional
                whether to plot the bias or not
            figsize : tuple, optional
                a tuple determining figure size
            show : bool, optional
                whether to show the plot or not
            save : bool, optional
                whether to save the plot or not
            verbosity : int, optional (default=0)
                determines the amount of print information
            **kwargs :
                any additional keyword arguments for taylor_plot function of
                easy_mpl_.

        Returns
        -------
        matplotlib.pyplot.Figure
            matplotlib Figure object which can be used for further processing

        Examples
        --------
        >>> from autotab import OptimizePipeline
        >>> from ai4water.datasets import busan_beach
        >>> data = busan_beach()
        >>> input_features = data.columns.tolist()[0:-1]
        >>> output_features = data.columns.tolist()[-1:]
        >>> pl = OptimizePipeline(input_features=input_features,
        >>>                       output_features=output_features)
        >>> results = pl.fit(data=data)
        ... # compare models with respect to evaluation metric
        >>> pl.taylor_plot(data=data)
        ... # compare the models by also plotting bias value
        >>> pl.taylor_plot(data=data, plot_bias=True)
        ... # get the matplotlb Figure object for further processing
        >>> fig = pl.taylor_plot(data=data, show=False)

        .. _easy_mpl:
            https://github.com/Sara-Iftikhar/easy_mpl#taylor_plot

        .. _Taylor:
            https://doi.org/10.1029/2000JD900719
        """

        if self.taylor_plot_data_['observations']['test'] is None:
            self.bfe_all_best_models(x=x,
                                     y=y,
                                     data=data,
                                     test_data=test_data,
                                     fit_on_all_train_data=fit_on_all_train_data,
                                     verbosity=verbosity)

        ax = taylor_plot(
            show=False,
            save=False,
            plot_bias=plot_bias,
            cont_kws={},
            grid_kws={},
            figsize=figsize,
            **self.taylor_plot_data_,  # simulations and trues as keyword arguments
            **kwargs
        )

        ax.legend(loc=(1.01, 0.01))

        fname = os.path.join(self.path, "taylor_plot")

        if save:
            plt.savefig(fname, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        # save taylor plot data as csv file, first make a dataframe
        sim = self.taylor_plot_data_['simulations']['test']
        data = np.column_stack([v.reshape(-1, ) for v in sim.values()])
        df = pd.DataFrame(data, columns=list(sim.keys()))
        df['observations'] = self.taylor_plot_data_['observations']['test']

        df.to_csv(os.path.join(self.path, "taylor_data.csv"), index=False)

        return ax

    def save_results(self)->None:
        """
        saves the results. It is called automatically at the end of optimization.
        It saves tried models and transformations at each step as json file
        with the name ``parent_suggestions.json``.

        An ``errors.csv`` file is saved which contains validation performance of
        the models at each optimization iteration with respect to all metrics
        being monitored.

        The performance of each model during child optimization iteration is saved
        as a csv file with the name ``child_val_scores.csv``.

        The global seeds for parent and child iterations are also saved in csv
        files with name ``parent_seeds.csv`` and ``child_seeds.csv``.
        All of these results are saved in pl.path folder.

        Returns
        -------
        None
        """
        setattr(self, "end_time_", time.asctime())

        # results are only available if fit has been run.
        if hasattr(self, 'parent_iter_'):

        # save parent_suggestions
            parent_suggestions = jsonize(self.parent_suggestions_)
            with open(os.path.join(self.path, "parent_suggestions.json"), "w") as fp:
                json.dump(parent_suggestions, fp, sort_keys=True, indent=True)

            # make a 2d array of all errors being monitored.
            errors = pd.concat([self.metrics_,
                                pd.DataFrame(self.val_scores_, columns=['val_scores'])],
                               axis=1)
            # save the errors being monitored
            fpath = os.path.join(self.path, "errors.csv")
            errors.to_csv(fpath, index_label="iterations")

            # save results of child iterations as csv file
            fpath = os.path.join(self.path, "child_val_scores.csv")
            pd.DataFrame(
                self.child_val_scores_,
                columns=[f'child_iter_{i}' for i in range(self.max_child_iters)]).to_csv(fpath)

            fpath = os.path.join(self.path, 'child_seeds.csv')
            pd.DataFrame(self.child_seeds_, columns=['child_seeds']).to_csv(fpath, index=False)

            fpath = os.path.join(self.path, 'parent_seeds.csv')
            pd.DataFrame(self.parent_seeds_, columns=['parent_seeds']).to_csv(fpath, index=False)
        return

    def metric_report(self, metric_name: str) -> str:
        """report with respect to one performance metric"""
        
        if self.parent_iter_ == 0:
            rep = 'Stopped at first iteration'
        else:
            metric_val_ = self.get_best_metric(metric_name)
            best_model_name = list(self.get_best_pipeline_by_metric(metric_name)['model'].keys())[0]

            rep = f"""
        With respect to {metric_name},
    the best model was {best_model_name} which had 
    '{metric_name}' value of {round(metric_val_, 4)}. This model was obtained at 
    {self.get_best_metric_iteration(metric_name)} iteration and is saved at 
    {self.get_best_pipeline_by_metric(metric_name)['path']}
            """
        return rep

    def report(
            self,
            write: bool = True
    ) -> str:
        """makes the report and writes it in text form"""

        if not hasattr(self, 'start_time_'):
            return "no iteration was run"

        st_time = self.start_time_
        en_time = getattr(self, "end_time_", time.asctime())

        num_models = len(self.models)
        text = f"""
    The optimization started at {st_time} and ended at {en_time} after 
    completing {self.parent_iter_} iterations. The optimization considered 
    {num_models} models. 
        """

        if self.parent_iter_ < self.parent_iterations:
            text += f"""
    The given parent iterations were {self.parent_iterations} but optimization 
    stopped early"""

        if hasattr(self, 'exc_type_'):
            text += f"""
    Execution was stopped due to {str(self.exc_type_)} with {str(self.exc_val_)}
    """

        for metric in self.monitor_names:
            text += self.metric_report(metric)

        if write:
            rep_fpath = os.path.join(self.path, "report.txt")
            with open(rep_fpath, "w") as fp:
                fp.write(text)

        return text

    def _runtime_attrs(self) -> dict:
        """These attributes are only set during call to fit"""
        config = {}
        for attr in ['start_time_', 'end_time_', 'child_iter_', 'parent_iter_']:
            config[attr] = getattr(self, attr, None)

        data_config = {}
        if hasattr(self, 'data_'):
            data_config['type'] = self.data_.__class__.__name__
            if isinstance(self.data_, pd.DataFrame):
                data_config['shape'] = self.data_.shape
                data_config['columns'] = self.data_.columns

        config['data'] = data_config
        return config

    def _init_paras(self) -> dict:
        """Returns the initializing parameters of this class"""
        signature = inspect.signature(self.__init__)

        init_paras = {}
        for para in signature.parameters.values():
            if para.name not in ["prefix"]:
                init_paras[para.name] = getattr(self, para.name)

        return init_paras

    @staticmethod
    def _sys_info()->dict:
        """returns system information as a dictionary"""
        import platform

        info = {}
        environ = {}
        for k,v in os.environ.items():
            if k in ['CONDA_DEFAULT_ENV', 'NUMBER_OF_PROCESSORS', 'USERNAME',
                     'CONDA_PREFIX', 'OS']:
                environ[k] = v

        info['environ'] = environ
        info['platform'] = [str(val) for val in platform.uname()]

        return info

    def _version_info(self) -> dict:
        """returns version of the third party libraries used"""

        import SeqMetrics
        import matplotlib
        import sklearn
        import easy_mpl
        from . import __version__
        versions = dict()
        versions['ai4water'] = ai4water.__version__
        versions['SeqMetrics'] = SeqMetrics.__version__
        versions['easy_mpl'] = easy_mpl.__version__
        versions['numpy'] = np.__version__
        versions['pandas'] = pd.__version__
        versions['matplotlib'] = matplotlib.__version__
        versions['sklearn'] = sklearn.__version__
        versions['python'] = sys.version
        versions['autotab'] = __version__

        try:
            import xgboost
            versions['xgboost'] = xgboost.__version__
        except (ModuleNotFoundError, ImportError):
            versions['xgboost'] = None

        try:
            import catboost
            versions['catboost'] = catboost.__version__
        except (ModuleNotFoundError, ImportError):
            versions['catboost'] = None

        try:
            import lightgbm
            versions['lightgbm'] = lightgbm.__version__
        except (ModuleNotFoundError, ImportError):
            versions['lightgbm'] = None

        try:
            import tensorflow
            versions['tensorflow'] = tensorflow.__version__
        except (ModuleNotFoundError, ImportError):
            versions['tensorflow'] = None

        versions['sys_info'] = self._sys_info()
        return versions

    def config(self) -> dict:
        """
        Returns a dictionary which contains all the information about the class
        and from which the class can be created.

        Returns
        -------
        dict
            a dictionary with two keys ``init_paras`` and ``runtime_paras`` and
            ``version_info``.

        """
        _config = {
            'init_paras': self._init_paras(),
            'version_info': self._version_info(),
            'runtime_attrs': self._runtime_attrs()
        }
        return _config

    @classmethod
    def from_config_file(cls, config_file: str) -> "OptimizePipeline":
        """Builds the class from config file.

        Parameters
        ----------
            config_file : str
                complete path of config file which has .json extension

        Returns
        -------
            an instance of OptimizePipeline class
        """

        if not os.path.isfile(config_file):
            raise ValueError(f"""
            config_file must be complete path of config file but it is 
            {config_file} of type {type(config_file)}
            """)

        with open(config_file, 'r') as fp:
            config = json.load(fp)

        model_kwargs = config['init_paras'].pop('model_kwargs')

        for arg in ['input_features', 'output_features']:
            if arg in model_kwargs:
                model_kwargs.pop(arg)

        if 'mode' in config['init_paras'] and 'mode' in model_kwargs:
            model_kwargs.pop('mode')

        pl = cls(**config['init_paras'], **model_kwargs)

        pl.start_time_ = config['runtime_attrs']

        path = os.path.dirname(config_file)
        fpath = os.path.join(path, "parent_suggestions.json")
        if os.path.exists(fpath):
            with open(fpath, "r") as fp:
                parent_suggestions = json.load(fp)

            pl.parent_suggestions_ = {int(k):v for k,v in parent_suggestions.items()}
            pl.parent_iter_ = len(parent_suggestions)

        fpath = os.path.join(path, "errors.csv")
        if os.path.exists(fpath):
            errors = pd.read_csv(fpath, index_col="iterations")

            # don't put val_scores in metrics_
            pl.val_scores_ = errors.pop('val_scores').values

            pl.metrics_ = errors

        pl.taylor_plot_data_ = {
                'simulations': {"test": {}},
                'observations': {"test": None}
            }

        fpath = os.path.join(path, "taylor_data.csv")
        if os.path.exists(fpath):
            taylor_data = pd.read_csv(fpath)
            pl.taylor_plot_data_['observations']['test'] = taylor_data.pop(
                'observations')

        pl.parent_prefix_ = os.path.basename(path)
        pl.path = path

        fpath = os.path.join(path, 'parent_seeds.csv')
        if os.path.exists(fpath):
            pl.parent_seeds_ = pd.read_csv(fpath).values

        fpath = os.path.join(path, "baselines", "results.json")
        pl.baseline_results_ = None
        if os.path.exists(fpath):
            with open(fpath, 'r') as fp:
                pl.baseline_results_ = json.load(fp)

        # TODO, must check whether callbacks were used or not,
        # if true, must raise error here.
        pl.callbacks_ = [Callbacks()]

        return pl

    @classmethod
    def from_config(cls, config: dict) -> "OptimizePipeline":
        """Builds the class from config dictionary

        Parameters
        ----------
            config : dict
                a dictionary which contains `init_paras` key.

        Returns
        -------
        OptimizePipeline
            an instance of OptimizePipeline class
        """
        return cls(**config['init_paras'])

    def refit_pipeline(
            self,
            x=None,
            y=None,
            data=None,
            test_data: Union[tuple, list] = None,
            metric_name: str = None,
            model_name: str = None,
    )->Model:

        if test_data is None:
            test_data = (None, None)

        train_x, train_y, val_x, val_y, test_x, test_y = self.verify_data(
            x=x,
            y=y,
            data=data,
            validation_data=None,
            test_data=test_data,
            save=True,
            save_name="from_scratch"
        )

        pipeline = self.get_best_pipeline(metric_name, model_name)

        model = self.build_model(
            model=pipeline['model'],
            x_transformation=pipeline['x_transformation'],
            y_transformation=pipeline['y_transformation']
        )

        x, y = combine_train_val(train_x, train_y, (val_x, val_y))
        model.fit(x, y)

        return model

    def evaluate_pipeline(
            self,
            x = None,
            y = None,
            metric_name: str = None,
            model_name: str = None,

    )->Model:
        """
        Evaluates the pipeline

        parameters
        ----------
        x :
        y :
        metric_name :
        model_name :


        Returns
        --------
        Model
        """
        pipeline = self.get_best_pipeline(metric_name, model_name)

        cpath = os.path.join(pipeline['path'], "config.json")
        model = self.build_model_from_config(cpath)
        wpath = os.path.join(pipeline['path'], "weights")
        model.verbosity = 1
        model.update_weights(os.path.join(wpath, find_best_weight(wpath)))

        print(model.evaluate(x=x, y=y, metrics="nse"))

        return model

    def get_best_pipeline(
            self,
            metric_name:str=None,
            model_name:str=None
    )->dict:
        """finds best pipeline"""
        metric_name = metric_name or self.eval_metric

        if model_name:
            _, pipeline = self.get_best_pipeline_by_model(model_name, metric_name)
        else:
            pipeline = self.get_best_pipeline_by_metric(metric_name=metric_name)

        return pipeline

    def be_best_model_from_config(
            self,
            x=None,
            y=None,
            data=None,
            test_data: Union[tuple, list] = None,
            metric_name: str = None,
            model_name: str = None,
            verbosity = 1
    )->Model:
        """Build and Evaluate the best model with respect to metric *from config*.

        Parameters
        ----------
            x :
                the input data for training
            y :
                the target data for training
            data :
                raw unprepared and unprocessed data from which x,y pairs for both
                training and test will be prepared. It is only required if x, y
                are not provided.
            test_data :
                a tuple/list of length 2 whose first element is x and second value
                is y. The is the data on which the performance of optimized
                pipeline will be calculated. This should only be given if ``data``
                argument is not given.
            metric_name : str
                the metric with respect to which the best model is fetched
                and then built/evaluated. If not given, the best model is
                built/evaluated with respect to evaluation metric.
            model_name : str, optional
                If given, the best version of this model will be fetched and built.
                The 'best' will be decided based upon `metric_name`
            verbosity : int, optional (default=1)
                determines the amount of print information

        Returns
        -------
            an instance of trained ai4water Model
        """
        if test_data is None:
            test_data = (None, None)

        train_x, train_y, val_x, val_y, *test_data = self.verify_data(
            x=x,
            y=y,
            data=data,
            validation_data=None,
            test_data=test_data)

        pipeline = self.get_best_pipeline(metric_name, model_name)

        cpath = os.path.join(pipeline['path'], "config.json")
        if verbosity:
            print(f"building using config file from {cpath}")
        model = self.build_model_from_config(cpath)
        model.config['verbosity'] = verbosity
        model.verbosity = verbosity

        if self.category == "ML":
            wpath = os.path.join(pipeline['path'], "weights",
                                 list(pipeline['model'].keys())[0])
            model.update_weights(wpath)

        else:
            wpath = os.path.join(pipeline['path'], "weights")
            model.update_weights(os.path.join(wpath, find_best_weight(wpath)))

        self._populate_results(model, train_x, train_y, val_x, val_y, *test_data)

        return model

    def bfe_model_from_scratch(
            self,
            iter_num: int,
            x = None,
            y = None,
            data = None,
            test_data: Union[tuple, list]=None,
            fit_on_all_train_data: bool = True,
    )->Model:
        """
        Builds, trains and evalutes the model from a specific iteration.
        The model is trained on 'training'+'validation' data.

        Parameters
        ----------
            iter_num : int
                iteration number from which to choose the model
            x :
                the input data for training
            y :
                the target data for training
            data :
                raw unprepared and unprocessed data from which x,y pairs for both
                training and test will be prepared. It is only required if x, y
                are not provided.
            test_data :
                a tuple/list of length 2 whose first element is x and second
                value is y. The is the data on which the performance of optimized
                pipeline will be calculated. This should only be given if ``data``
                argument is not given.
            fit_on_all_train_data : bool, optional (default=True)
                If true, the model is trained on (training+validation) data.
                This is based on supposition that the data is split into
                training, validation and test sets. The optimization of
                pipeline was performed on validation data. But now, we
                are training the model on all available training data
                which is (training + validation) data. If False, then
                model is trained only on training data.
        Returns
        -------
            an instance of trained ai4water Model
        """
        if test_data is None:
            test_data = (None, None)

        train_x, train_y, val_x, val_y, test_x, test_y = self.verify_data(
            x=x,
            y=y,
            data=data,
            validation_data=None,
            test_data=test_data,
            save=True,
            save_name="from_scratch_all"
        )

        pipeline = self.parent_suggestions_[iter_num]
        prefix = f"{self.path}{SEP}results_from_scratch{SEP}iteration_{iter_num}"

        model = self._build_and_eval_from_scratch(
            model=pipeline['model'],
            train_x=train_x,
            train_y=train_y,
            validation_data=(val_x, val_y),
            test_x=test_x,
            test_y=test_y,
            x_transformation=pipeline['x_transformation'],
            y_transformation=pipeline['y_transformation'],
            prefix=prefix,
            fit_on_all_train_data=fit_on_all_train_data,
            seed=self.parent_seeds_[int(pipeline['iter_num'])-1]
        )
        return model

    def bfe_best_model_from_scratch(
            self,
            x = None,
            y = None,
            data = None,
            test_data:tuple = None,
            metric_name: str = None,
            model_name: str = None,
            fit_on_all_train_data: bool = True,
            verbosity:int = 1,
    )->Model:
        """
        Builds, Trains and Evaluates the **best model** with respect to metric from
        scratch. The model is trained on 'training'+'validation' data. Running
        this mothod will also populate ``taylor_plot_data_`` dictionary.

        Parameters
        ----------
            x :
                the input data for training
            y :
                the target data for training
            data :
                raw unprepared and unprocessed data from which x,y pairs for both
                training and test will be prepared. It is only required if x, y
                are not provided.
            test_data :
                a tuple/list of length 2 whose first element is x and second value
                is y. The is the data on which the peformance of optimized
                pipeline will be calculated. This should only be given if ``data``
                argument is not given.
            metric_name : str
                the metric with respect to which the best model is searched
                and then built/trained/evaluated. If None, the best model is
                chosen based on the evaluation metric.
            model_name : str, optional
                If given, the best version of this model will be found and built.
                The 'best' will be decided based upon `metric_name`
            fit_on_all_train_data : bool, optional (default=True)
                If true, the model is trained on (training+validation) data.
                This is based on supposition that the data is split into
                training, validation and test sets. The optimization of
                pipeline was performed on validation data. But now, we
                are training the model on all available training data
                which is (training + validation) data. If False, then
                model is trained only on training data.
            verbosity : int, optional (default=1)
                determines amount of information to be printed.

        Returns
        -------
            an instance of trained ai4water Model
        """
        if test_data is None:
            test_data = (None, None)

        train_x, train_y, val_x, val_y, test_x, test_y = self.verify_data(
            x=x, y=y,
            data=data,
            validation_data=None,
            test_data=test_data,
            save=True,
            save_name="from_scracth"
        )

        metric_name = metric_name or self.eval_metric

        if model_name:
            met_val, pipeline = self.get_best_pipeline_by_model(
                model_name,
                metric_name)
        else:
            met_val = self.get_best_metric(metric_name)
            pipeline = self.get_best_pipeline_by_metric(metric_name=metric_name)

        met_val = round(met_val, 3)

        model_name = model_name or ''
        suffix = f"{SEP}{metric_name}_{met_val}_{model_name}"
        prefix = f"{self.path}{SEP}results_from_scratch{suffix}"

        model_config = pipeline['model']

        if self.category == "DL":
            model_name = list(model_config.keys())[0]
            kwargs = list(model_config.values())[0]

            model_config = DL_MODELS[model_name](mode=self.mode,
                                                 input_shape=self.input_shape,
                                                 num_outputs=self.num_outputs,
                                                 **kwargs)

        model = self._build_and_eval_from_scratch(
            model=model_config,
            train_x=train_x,
            train_y = train_y,
            validation_data=(val_x, val_y),
            test_x=test_x,
            test_y=test_y,
            x_transformation=pipeline['x_transformation'],
            y_transformation=pipeline['y_transformation'],
            prefix=prefix,
            fit_on_all_train_data=fit_on_all_train_data,
            verbosity=verbosity,
            seed=self.parent_seeds_[int(pipeline['iter_num'])-1]
        )

        return model

    def _build_and_eval_from_scratch(
            self,
            model: Union[str, dict],
            train_x,
            train_y,
            validation_data,
            test_x,
            test_y,
            x_transformation: Union[str, dict],
            y_transformation: Union[str, dict],
            prefix:str,
            model_name=None,
            verbosity:int = 1,
            fit_on_all_train_data:bool = True,
            seed:int = None,
    ) -> "Model":
        """builds and evaluates the model from scratch. If model_name is given,
        model's predictions are saved in 'taylor_plot_data_' dictionary
        """
        model = self.build_model(
            model=model,
            x_transformation=x_transformation,
            y_transformation=y_transformation,
            prefix=prefix,
            verbosity=verbosity
        )

        if seed:
            model.seed_everything(int(seed))

        if fit_on_all_train_data:
            x, y = combine_train_val(train_x, train_y, validation_data)
            model.fit(x, y)
            self._populate_results(
                model, x, y, *validation_data,
                test_x=test_x, test_y=test_y,
                model_name=model_name)
        else:
            if self.category == "ML":
                model.fit(train_x, train_y)
            else:
                model.fit(train_x, train_y, validation_data=validation_data)
            self._populate_results(
                model, train_x, train_y, *validation_data, test_x, test_y,
                model_name=model_name)

        return model

    def _populate_results(
            self,
            model: Model,
            train_x,
            train_y,
            val_x,
            val_y,
            test_x=None,
            test_y=None,
            model_name=None
    ) -> None:
        """evaluates/makes predictions from model on traiing/validation/test data.
        if model_name is given, model's predictions are saved in 'taylor_plot_data_'
        dictionary
        """

        model.predict(train_x, train_y, metrics="all", plots=self._pp_plots)

        t, p = model.predict(
            val_x, val_y, metrics="all", plots=self._pp_plots, return_true=True)

        if test_x is not None:
            t, p = model.predict(
                test_x,
                test_y,
                metrics="all",
                plots=self._pp_plots,
                return_true=True)


        if model_name:
            self.taylor_plot_data_['observations']['test'] = t
            self.taylor_plot_data_['simulations']['test'][model_name] = p

        return

    def evaluate_model(
            self,
            model: Model,
            x = None,
            y = None,
            data=None,
            metric_name: str = None,
    )->float:
        """Evaluates the ai4water's Model on the data for the metric.

        Parameters
        ----------
            model :
                an instance of ai4water's Model class
            data :
                raw, unprocessed data form which x,y pairs are made
            metric_name : str, optional
                name of performance metric. If not given, evaluation metric
                is used.
            x :
                alternative to ``data``. Only required if ``data`` is not given.
            y :
                only required if x is given

        Returns
        -------
            float, the evaluation score of model with respect to ``metric_name``
        """
        metric_name = metric_name or self.eval_metric

        assert hasattr(model, 'predict')

        if x is not None:
            assert y is not None
            t, p = model.predict(
                x=x, y=y, process_results=False, return_true=True)
        else:
            assert x is None
            t, p = model.predict_on_test_data(
                data=data, process_results=False, return_true=True)

        errors = self.Metrics(t, p, multiclass=model.is_multiclass_)

        return getattr(errors, metric_name)()

    def bfe_all_best_models(
            self,
            x = None,
            y = None,
            data = None,
            test_data:tuple = None,
            metric_name: str = None,
            fit_on_all_train_data: bool = True,
            verbosity:int = 0,
    ) -> None:
        """
        builds, trains and evaluates best versions of all the models.
        The model is trained on 'training'+'validation' data.

        Parameters
        ----------
            x :
                the input data for training
            y :
                the target data for training
            data :
                raw unprepared and unprocessed data from which x,y pairs for both
                training and test will be prepared. It is only required if x, y
                are not provided.
            test_data :
                a tuple/list of length 2 whose first element is x and second value
                is y. The is the data on which the performance of optimized pipeline
                will be calculated. This should only be given if ``data`` argument
                is not given.
            metric_name : str
                the name of metric to determine best version of a model. If not
                given, parent_val_metric will be used.
            fit_on_all_train_data : bool, optional (default=True)
                If true, the model is trained on (training+validation) data.
                This is based on supposition that the data is split into
                training, validation and test sets. The optimization of
                pipeline was performed on validation data. But now, we
                are training the model on all available training data
                which is (training + validation) data. If False, then
                model is trained only on training data.
            verbosity : int, optional (default=0)
                determines the amount of print information
        Returns
        -------
        None

        """

        train_x, train_y, val_x, val_y, test_x, test_y = self.verify_data(
            x=x, y=y,  data=data,
            validation_data=None,
            test_data=test_data)

        met_name = metric_name or self.eval_metric_name

        for model in self.models:

            try:
                metric_val, pipeline = self.get_best_pipeline_by_model(
                    model, met_name)
            except ModelNotUsedError:
                continue

            prefix = f"{self.path}{SEP}results_from_scratch{SEP}{met_name}_{metric_val}_{model}"

            model_config = pipeline['model']

            if self.category == "DL":
                model_name = list(model_config.keys())[0]
                kwargs = list(model_config.values())[0]

                model_config = DL_MODELS[model_name](mode=self.mode,
                                                     input_shape=self.input_shape,
                                                     num_outputs=self.num_outputs,
                                                     **kwargs)
            _ = self._build_and_eval_from_scratch(
                model=model_config,
                train_x=train_x,
                train_y=train_y,
                validation_data=(val_x, val_y),
                test_x=test_x,
                test_y = test_y,
                x_transformation=pipeline['x_transformation'],
                y_transformation=pipeline['y_transformation'],
                prefix=prefix,
                model_name=model,
                fit_on_all_train_data=fit_on_all_train_data,
                verbosity=verbosity,
                seed=self.parent_seeds_[int(pipeline['iter_num'])-1],
            )

        return

    def post_fit(
            self,
            x = None,
            y = None,
            data = None,
            test_data:Union[list, tuple] = None,
            fit_on_all_train_data:bool = True,
            show:bool = True
    ) -> None:
        """post processing of results to draw dumbbell plot and taylor plot.

        Parameters
        ----------
        x :
            the input data for training
        y :
            the target data for training
        data :
            raw unprepared and unprocessed data from which x,y pairs for both training
            and test will be prepared. It is only required if x, y are not provided.
        test_data :
            a tuple/list of length 2 whose first element is x and second value is y.
            The is the data on which the performance of optimized pipeline will be
            calculated. This should only be given if ``data`` argument is not given.
        fit_on_all_train_data : bool, optional (default=True)
            If true, the model is trained on (training+validation) data.
            This is based on supposition that the data is split into
            training, validation and test sets. The optimization of
            pipeline was performed on validation data. But now, we
            are training the model on all available training data
            which is (training + validation) data. If False, then
            model is trained only on training data.
        show : bool, optional (default=True)
            whether to show the plots or not

        Returns
        -------
        None

        """

        self.bfe_all_best_models(x=x,
                                 y=y,
                                 data=data,
                                 fit_on_all_train_data=fit_on_all_train_data,
                                 test_data=test_data)
        self.dumbbell_plot(x=x,
                           y=y,
                           data=data,
                           test_data=test_data,
                           fit_on_all_train_data=fit_on_all_train_data,
                           metric_name=self.eval_metric,
                           show=show)

        # following plots only make sense if more than one models are tried
        if self._optimize_model:
            self.taylor_plot(x=y,
                             y=y,
                             data=data,
                             test_data=test_data,
                             fit_on_all_train_data=fit_on_all_train_data,
                             show=show)
            self.compare_models(show=show)
            self.compare_models(plot_type="bar_chart", show=show)

        return

    def cleanup(
            self,
            dirs_to_exclude: Union[str, list] = None
    ) -> None:
        """removes the folders from path except the 'results_from_scratch' and
        the folders defined by user.

        Parameters
        ----------
            dirs_to_exclude : str, list, optional
                The names of folders inside path which should not be deleted.

        Returns
        -------
            None
        """
        if isinstance(dirs_to_exclude, str):
            dirs_to_exclude = [dirs_to_exclude]

        if dirs_to_exclude is None:
            dirs_to_exclude = []

        for _item in os.listdir(self.path):
            _path = os.path.join(self.path, _item)
            if os.path.isdir(_path):
                if _item not in ['results_from_scratch'] + dirs_to_exclude:
                    shutil.rmtree(_path)
        return

    def compare_models(
            self,
            metric_name: str = None,
            plot_type: str = "circular",
            show : bool = False,
            **kwargs
    )->plt.Axes:
        """
        Compares all the models with respect to a metric and plots a bar plot.

        Parameters
        ----------
            metric_name : str, optional
                The metric with respect to which to compare the models.
            plot_type : str, optional
                if "circular" then `easy_mpl.circular_bar_plot <https://easy-mpl.readthedocs.io/en/latest/#module-12>`_
                is drawn otherwise a simple bar_plot is drawn.
            show : bool, optional
                whether to show the plot or not
            **kwargs :
                keyword arguments for `easy_mpl.circular_bar_plot <https://easy-mpl.readthedocs.io/en/latest/#module-12>`_
                or `easy_mpl.bar_chart <https://easy-mpl.readthedocs.io/en/latest/#module-1>`_

        Returns
        -------
            matplotlib.pyplot.Axes

        Exmaples
        --------
        >>> from autotab import OptimizePipeline
        >>> from ai4water.datasets import busan_beach
        >>> data = busan_beach()
        >>> input_features = data.columns.tolist()[0:-1]
        >>> output_features = data.columns.tolist()[-1:]
        >>> pl = OptimizePipeline(input_features=input_features,
        >>>                       output_features=output_features)
        >>> results = pl.fit(data=data)
        ... # compare models with respect to evaluation metric
        >>> pl.compare_models()
        ... # compare models with respect to bar_chart and plot comparison using bar_chart
        >>> pl.compare_models('r2', "bar_chart")
        ... # compare models with respect to r2 and get the matplotlb axes for further processing
        >>> axes = pl.compare_models('r2', show=False)
        """

        metric_name = metric_name or self.eval_metric

        models = {}

        for model in self.models:

            try:
                metric_val, _ = self.get_best_pipeline_by_model(model, metric_name)
                models[model] = metric_val
            except ModelNotUsedError:
                continue

        labels = _shred_suffix(list(models.keys()))

        plt.close('all')
        if plot_type == "circular":
            ax = circular_bar_plot(np.array(list(models.values())),
                                   labels,
                                   sort=True,
                                   show=False,
                                   **kwargs)
        else:
            ax = bar_chart(
                list(models.values()),
                labels,
                ax_kws={'xlabel': METRIC_NAMES.get(metric_name, metric_name)},
                sort=True,
                show=False,
                **kwargs)

        fpath = os.path.join(self.path, f"{plot_type}_plot_wrt_{metric_name}")
        plt.savefig(fpath, dpi=300, bbox_inches='tight')

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def _eval_model_manually(
            self,
            model: Model,
            data:tuple,
            metric: Union[str, Callable],
            metric_name:str,
            callbacks:list,
            eval_metrics=False
    ) -> float:
        """evaluates the model
        callbacks : list
            list of callbacks, which can be parent or child callbacks
        """

        t, p = model.predict(*data, return_true=True, process_results=False)

        for cbk in callbacks:
            getattr(cbk, 'on_eval_begin')(
                model, self.parent_iter_, x=None, y=None, validation_data=data)

        if len(p) == p.size:
            p = p.reshape(-1, 1)  # TODO, for cls, Metrics do not accept (n,) array

        if self.mode=="classification":
            # if array has shape (n,1)/(n,) then we should not do
            # np.argmax
            if len(t) != t.size:
               t = np.argmax(t, axis=1)
               p = np.argmax(p, axis=1)
        else:

            # 32 bit float can cause overflow when calculating some metrics
            p = p.astype(np.float64)

        errors = self.Metrics(
            t,
            p,
            remove_zero=True,
            remove_neg=True,
            multiclass=model.is_multiclass_)
        
        if callable(metric):
            val_score = metric(t, p)
        else:    
            val_score = getattr(errors, metric)()

        metric_type = METRIC_TYPES.get(metric_name, 'min')

        # the optimization will always solve minimization problem so if
        # the metric is to be maximized change the val_score accordingly
        if metric_type != "min":
            val_score = 1.0 - val_score

        # val_score can be None/nan/inf
        if not math.isfinite(val_score):
            _metric_type = METRIC_TYPES[self.eval_metric]
            func = compare_func1(_metric_type)
            best_so_far = func(self.val_scores_)
            val_score = fill_val(_metric_type, best_so_far)

        if eval_metrics:
            # calculate all additional performance metrics which are being monitored

            for _metric, metric_name in zip(self.monitor, self.monitor_names):
                if callable(_metric):
                    pm = _metric(t,p)
                else:
                    pm = getattr(errors, _metric)(**METRICS_KWARGS.get(_metric, {}))

                self.metrics_.at[self.parent_iter_, metric_name] = pm

                func = compare_func1(METRIC_TYPES[metric_name])

                pm_until_this_iter = self.metrics_best_.loc[:self.parent_iter_, metric_name]

                if pm_until_this_iter.isna().sum() == pm_until_this_iter.size:
                    best_so_far = fill_val(METRIC_TYPES[metric_name], np.nan)
                else:
                    best_so_far = func(
                        self.metrics_best_.loc[:self.parent_iter_, metric_name])

                    best_so_far = fill_val(METRIC_TYPES[metric_name], best_so_far)

                func = compare_func(METRIC_TYPES[metric_name])
                if func(pm, best_so_far):

                    self.metrics_best_.at[self.parent_iter_, metric_name] = pm

        for cbk in callbacks:
            getattr(cbk, 'on_eval_end')(
                model, self.parent_iter_, x=None, y=None, validation_data=data)

        return val_score

    def verify_data(
            self,
            x=None,
            y=None,
            data=None,
            validation_data=None,
            test_data = None,
            save:bool= False,
            save_name:str = ''
    )->tuple:
        """
        only x,y should be given
        or x,y and validation_data should be given
        or only data should be given
        test_data, if given should only be given as tuple
        every other combination of x,y, data and validation_data will raise error
        """
        model_maker = make_model(**self.model_kwargs)
        data_config = model_maker.data_config

        if test_data is None:
            test_data = (None, None)

        test_x, test_y = test_data

        def num_examples(samples):
            if isinstance(samples, list):
                assert len(set(len(sample) for sample in samples)) == 1
                return len(samples[0])
            return len(samples)

        category = self.category
        if 'category' in data_config:
            data_config.pop('category')

        if x is None:
            # case 3: only data should be given

            assert y is None, f"y must only be given if x is given. x is {type(x)}"
            assert data is not None, f"if x is given, data must not be given"
            assert validation_data is None, f"validation data must only be given if x is given"
            assert test_x is None, f"test data must only be given if x is given"
            assert isinstance(data, pd.DataFrame), f"data must be dataframe, but it is {type(data)}"
            dataset = DataSet(data=data,
                              save=data_config.pop('save') or True,
                              category = category,
                              **data_config)
            train_x, train_y = dataset.training_data()
            val_x, val_y = dataset.validation_data()
            test_x, test_y = dataset.test_data()

        else:
            assert y is not None, f"if x is given, corresponding y must also be given"
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.values

            assert isinstance(y, np.ndarray)
            assert num_examples(x) == num_examples(y)

            if validation_data is None:
                # case 1: only x,y should be given
                # get train_x, train_y, val_x, val_y from DataSet

                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                data = pd.DataFrame(np.concatenate([x, y], axis=1), columns=self.all_features)
                dataset = DataSet(data=data,
                                  save=data_config.pop('save') or True,
                                  category = category,
                                  **data_config)
                train_x, train_y = dataset.training_data()
                val_x, val_y = dataset.validation_data()
                test_x, test_y = dataset.test_data()

            else:
                # case 2: x,y and validation_data should be given
                msg = f"Validation data must be of type tuple but it is {type(validation_data)}"
                assert isinstance(validation_data, (tuple, list)), msg
                msg = f"Validation_data tuple must have length 2 but it has {len(validation_data)}"
                assert len(validation_data) == 2, msg
                msg1 = f"second value in Validation data must be ndarray"
                assert isinstance(validation_data[1], (np.ndarray, pd.Series, pd.DataFrame)), msg1
                assert num_examples(validation_data[0]) == num_examples(validation_data[1])
                    
                train_x, train_y = x, y
                if isinstance(train_y, (pd.DataFrame, pd.Series)):
                    train_y = train_y.values

                val_x, val_y = validation_data
                if isinstance(val_y, (pd.DataFrame, pd.Series)):
                    val_y = val_y.values

        if save:
            try:
                import h5py
                filepath = os.path.join(self.path, f"data_{save_name}.h5")
                data_to_h5(filepath, train_x, train_y, val_x, val_y, test_x, test_y)
            except (ModuleNotFoundError, ImportError):
                data_to_csv(os.path.join(self.path, f"training_data_{save_name}.csv"), self.all_features, train_x, train_y)
                data_to_csv(os.path.join(self.path, f"validation_data_{save_name}.csv"), self.all_features, val_x, val_y)
                data_to_csv(os.path.join(self.path, f"test_data_{save_name}.csv"), self.all_features, test_x, test_y)

        if train_x.ndim > 2 and 'murphy' in self._pp_plots:
            self._pp_plots.remove('murphy')

        train_y = self._verify_ouput(train_y)
        val_y = self._verify_ouput(val_y)
        test_y = self._verify_ouput(test_y)

        return train_x, train_y, val_x, val_y, test_x, test_y

    def _verify_ouput(self, outputs):
        if outputs is not None:
            if self.mode == 'classification':
                if isinstance(outputs, np.ndarray):
                    outputs = outputs.astype(int)
                if self.category == "DL" and self.num_classes == 2:
                    outputs = np.argmax(outputs, 1).reshape(-1, 1)
        return outputs

    def plot_convergence(
            self,
            metric_name:str = None,
            original:bool = False,
            ax:plt.Axes = None,
            save:bool = True,
            show:bool = False,
            **kwargs
    ):
        """
        plots convergence of optimization.

        parameters
        -----------
        metric_name : str
            name of performance metric w.r.t which the convergence is to be shown
        original : bool
            whether to show the original convergence or only show the improvement
        ax : plt.Axes
            matplotlib Axes on which to draw the plot
        save : bool
        show : bool


        returns
        --------
        plt.Axes
        """

        metric_name = metric_name or self.eval_metric

        errors = os.path.join(self.path, "errors.csv")
        serialized = os.path.join(self.path, "serialized.json")
        if os.path.exists(errors):
            df = pd.read_csv(errors)
            y = df[metric_name]
        elif os.path.exists(serialized):
            serialized = os.path.join(self.path, "serialized.json")
            with open(serialized, 'r') as fp:
                results= json.load(fp)
                y = results['func_vals']
        else:
            raise FileNotFoundError

        _kwargs = {
        }

        if kwargs is None:
            kwargs = dict()

        _kwargs.update(kwargs)

        plt.close('all')
        if original:
            ax = plot(y, '--.',
                 ax_kws=dict(xlabel="Number of calls $n$",
                 ylabel=r"$\min f(x)$ after $n$ calls"),
                               show=False,
                               **_kwargs)
        else:
            ax = plot_convergence(y, ax=ax, show=False, **_kwargs)

        if save:
            fname = os.path.join(self.path, "convergence.png")
            plt.savefig(fname, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return ax

    def parallel_coordinates(self):
        x = []
        y = []
        for iter_sugges, iter_y in zip(self.parent_suggestions_.values(), self.val_scores_):
            trans_x = [tr['method'] for tr in iter_sugges['x_transformation']]
            trans_y = [tr['method'] for tr in iter_sugges['y_transformation']]
            model = [model for model in iter_sugges['model']]

            y.append(iter_y)
            if len(trans_y) == 0:
                trans_y = ['none']

            x.append(trans_x + trans_y)

            names = [tr['features'] for tr in iter_sugges['x_transformation']]

        names = [item for sublist in names for item in sublist]

        df = pd.DataFrame(x, columns=names + self.output_features)

        return parallel_coordinates(df, categories=y, figsize=(20, 6))


def combine_train_val(train_x, train_y, validation_data):
    if validation_data is None:
        return train_x, train_y

    x_val, y_val = validation_data
    if isinstance(train_x, list):
        x = []
        for val in range(len(train_x)):
            if x_val is not None:
                _val = np.concatenate([train_x[val], x_val[val]])
                x.append(_val)
            else:
                _val = train_x[val]

        y = train_y
        if hasattr(y_val, '__len__') and len(y_val) > 0:
            y = np.concatenate([train_y, y_val])

    elif isinstance(train_x, np.ndarray):
        x, y = train_x, train_y
        # if not validation data is available then use only training data
        if x_val is not None:
            if hasattr(x_val, '__len__') and len(x_val)>0:
                x = np.concatenate([train_x, x_val])
                y = np.concatenate([train_y, y_val])
    else:
        raise NotImplementedError

    return x, y


def _shred_suffix(labels:list)->list:

    new_labels = []

    for label in labels:
        if label.endswith('Regressor'):
            label = label.replace('Regressor', '')
        elif label.endswith('Classifier'):
            label = label.replace('Classifier', '')
        new_labels.append(label)

    return new_labels


class MetricNotMonitored(Exception):

    def __init__(self, metric_name, available_metrics):
        self.metric = metric_name
        self.avail_metrics = available_metrics

    def __str__(self):
        return f"""
        metric {self.metric} was not monitored. Please choose from
        {self.avail_metrics}
        """


class ModelNotUsedError(Exception):

    def __init__(self, model_name):
        self.model = model_name

    def __str__(self):
        return f"""model {self.model} is not used during optimization"""


def compare_func(metric_type:str):
    # use np.less and np.greater than np.less_equal and np.greater_equal
    # because in classification we can get exactly same output again
    # and again
    if metric_type == "min":
        return np.less
    return np.greater


def compare_func1(metric_type:str):
    if metric_type == "min":
        return np.nanmin
    return np.nanmax


def fill_val(metric_type:str, best_so_far):
    if math.isfinite(best_so_far):
        return best_so_far
    if metric_type == "min":
        return 99999999999999.0
    return -9999999999.0
