
import os
import gc
import json
import sys
import time
import math
import shutil
import inspect
from typing import Union, Callable
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SeqMetrics import RegressionMetrics, ClassificationMetrics
from easy_mpl import dumbbell_plot, taylor_plot, circular_bar_plot, bar_chart

import ai4water
from ai4water import Model
from ai4water._optimize import make_space
from ai4water.hyperopt.utils import to_skopt_space
from ai4water.utils.utils import dateandtime_now, jsonize
from ai4water.hyperopt import Categorical, HyperOpt, Integer, Real
from ai4water.models import MLP, CNN, LSTM, CNNLSTM, LSTMAutoEncoder, TFT, TCN
from ai4water.experiments.utils import regression_space, classification_space, dl_space


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
    "minmax", "center", "scale", "zscore", "box-cox", "yeo-johnson",
    "quantile", "robust", "log", "log2", "log10", "sqrt", "none",
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
}

def compare_func(metric_type:str):
    if metric_type == "min":
        return np.less_equal
    return np.greater_equal


def compare_func1(metric_type:str):
    if metric_type == "min":
        return np.nanmin
    return np.nanmax


def fill_val(metric_type:str, best_so_far):
    if math.isfinite(best_so_far):
        return best_so_far
    if metric_type == "min":
        return 99999999999999
    return -9999999999


class OptimizePipeline(object):
    """
    optimizes model/estimator, its hyperparameters and preprocessing
    operation to be performed on input and output features. It consists of two
    hpo loops. The parent or outer loop optimizes preprocessing/feature engineering,
    feature selection and model selection while the child hpo loop optimizes
    hyperparmeters of child hpo loop.

    Attributes
    -----------

    - metrics_

    - parent_suggestions_:
        an ordered dictionary of suggestions to the parent objective function
        during parent hpo loop

    - child_val_metrics:
        a numpy array containing val_metrics of all child hpo loops

    - optimizer
        an instance of ai4water.hyperopt.HyperOpt for parent optimization

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
    -----
    This optimizationa always solves a minimization problem even if the val_metric
    is r2.
    """

    def __init__(
            self,
            inputs_to_transform,
            input_transformations: Union[list, dict] = None,
            outputs_to_transform=None,
            output_transformations: Union[list, ] = None,
            models: list = None,
            parent_iterations: int = 100,
            child_iterations: int = 25,
            parent_algorithm: str = "bayes",
            child_algorithm: str = "bayes",
            eval_metric: str = "mse",
            cv_parent_hpo: bool = None,
            cv_child_hpo: bool = None,
            monitor: Union[list, str] = None,
            mode: str = "regression",
            category:str = "ML",
            prefix: str = None,
            **model_kwargs
    ):
        """
        initializes the class

        Parameters
        ----------
            inputs_to_transform : list
                Input features on which feature engineering/transformation is to
                be applied. By default all input features are considered. If you
                want to apply a single transformation on a group of input features,
                then pass this as a dictionary. This is helpful if the input data
                consists of hundred or thousands of input features.
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
                    - ``robust``
                    - ``log``
                    - ``log2``
                    - ``log10``
                    - ``sqrt``    square root

                The user can however, specify list of transformations to be considered
                for each input feature. In such a case, this argument must be a
                dictionary whose keys are names of input features and values are
                list of transformations.

            outputs_to_transform : list, optional
                Output features on which feature engineering/transformation is to
                be applied. If None, then transformations on outputs are not applied.
            output_transformations :
                The transformations to be considered for outputs/targets. The user
                can consider any transformation as given for ``input_transformations``
            models : list, optional
                The models/algorithms to consider during optimzation. If not given, then all
                available models from sklearn, xgboost, catboost and lgbm are
                considered. For neural neworks, following 6 model types are
                considered by default

                    - MLP   multi layer perceptron
                    - CNN   1D convolution neural network
                    - LSTM  Long short term memory network
                    - CNNLSTM   CNN-> LSTM
                    - LSTMAutoEncoder  LSTM based autoencoder
                    - TCN  Temporal convolution networks
                    - TFT  Temporal fusion Transformer

                However, in such cases, the ``category`` must be ``DL``.

            parent_iterations : int, optional
                Number of iterations for parent optimization loop
            child_iterations : int, optional
                Number of iterations for child optimization loop. It set to 0,
                the child hpo loop is not run which means the hyperparameters
                of the model are not optimized. You can customize iterations for
                each model by making using of :meth: `change_child_iterations` method.
            parent_algorithm : str, optional
                Algorithm for optimization of parent optimzation
            child_algorithm : str, optional
                Algorithm for optimization of child optimization
            eval_metric : str, optional
                Validation metric to calculate val_score in objective function.
                The parent and child hpo loop optimizes/improves this metric. This metric is
                calculated on valdation data. If cross validation is performed then
                this metric is calculated using cross validation.
            cv_parent_hpo : bool, optional
                Whether we want to apply cross validation in parent hpo loop or not?.
                If given, the parent hpo loop will optimize the cross validation score.
                The  model is fitted on whole training data (training+validation) after
                cross validation and the metrics printed (other than parent_val_metric)
                are calculated on the based the updated model i.e. the one fitted on
                whole training (trainning+validation) data.
            cv_child_hpo :
                Whether we want to apply cross validation in child hpo loop or not?.
                If False, then val_score will be caclulated on validation data.
                The type of cross validator used is taken from model.config['cross_validator']
            monitor :
                Nmaes of performance metrics to monitor in parent hpo loop. If None,
                then R2 is monitored for regression and accuracy for classification.
            mode : str, optional (default="regression")
                whether this is a ``regression`` problem or ``classification``
            category : str, optional (detault="DL")
                either "DL" or "ML". If DL, the pipeline is optimized for neural networks.
            **model_kwargs :
                any additional key word arguments for ai4water's Model

        """
        if isinstance(inputs_to_transform, dict):
            self._groups = inputs_to_transform
            self.inputs_to_transform = list(inputs_to_transform.keys())
        else:
            self._groups = {inp:[inp] for inp in inputs_to_transform}
            self.inputs_to_transform = inputs_to_transform

        self.input_transformations = input_transformations

        self.output_transformations = output_transformations or DEFAULT_TRANSFORMATIONS

        assert mode in ("regression", "classification")
        self.mode = mode

        assert category in ("DL", "ML")
        self.category = category

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
                assert all([model in DL_MODELS.keys() for model in models]), f"""
                Only following deel learning models can be considered {DL_MODELS.keys()}
                """


        self.parent_iterations = parent_iterations
        self.child_iterations = child_iterations
        # for internal use, we keep child_iter for each model
        self._child_iters = {model: child_iterations for model in self.models}
        self.parent_algorithm = parent_algorithm
        self.child_algorithm = child_algorithm
        self.eval_metric = eval_metric
        self.cv_parent_hpo = cv_parent_hpo
        self.cv_child_hpo = cv_child_hpo

        for arg in ['model', 'x_transformation', 'y_transformation']:
            if arg in model_kwargs:
                raise ValueError(f"argument {arg} not allowed")
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

    @property
    def path(self):
        _path = os.path.join(os.getcwd(), "results", self.parent_prefix_)
        if not os.path.exists(_path):
            os.makedirs(_path)
        return _path

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, x):
        self._mode = x

    @property
    def Metrics(self):
        if self.mode == "regression":
            return RegressionMetrics
        return ClassificationMetrics

    @property
    def input_features(self):
        if 'input_features' in self.model_kwargs:
            return self.model_kwargs['input_features']
        else:
            raise ValueError

    @property
    def output_features(self):
        if 'output_features' in self.model_kwargs:
            _output_features = self.model_kwargs['output_features']
            if isinstance(_output_features, str):
                _output_features = [_output_features]
            return _output_features
        else:
            raise ValueError

    def _save_config(self):
        cpath = os.path.join(self.path, "config.json")
        config = self.config()
        with open(cpath, 'w') as fp:
            json.dump(jsonize(config), fp, indent=4)
        return

    def update_model_space(self, space: dict) -> None:
        """updates or changes the search space of an already existing model

        Parameters
        ---------
            space
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
        """
        for model, space in space.items():
            if model not in self.model_space:
                raise ValueError(f"{model} is not valid because it is not being considered.")
            space = to_skopt_space(space)
            self.model_space[model] = {'param_space': [s for s in space]}
        return

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
        iterations for XGBRegressor. In such a canse we can use this function to
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
            If we want to change iterations for more than one models
            >>> pl.change_child_iteration(({"XGBRegressor": 30,
            >>>                             "RandomForestRegressor": 20}))
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

        sp = make_space(self.inputs_to_transform + (self.outputs_to_transform or []),
                        categories=list(set(x_categories + y_categories)),
                        append=append)

        if len(self.models)>1:
            algos = Categorical(self.models, name="model")
            sp = sp + [algos]
        else:
            self._optimize_model = False
            self._model = self.models[0]

        return sp

    def add_batch_size_space(self):
        return Categorical([4, 8, 16, 32, 64], name="batch_size")

    def add_lr_space(self):
        space = Real(1e-5, 0.05, name="lr", num_samples=10)
        return space

    @property
    def max_child_iters(self) -> int:
        # the number of child hpo iterations can be different based upon models
        # this property calculates maximum child iterations
        return max(self._child_iters.values())

    def reset(self):
        # called at the start of fit method

        self.metrics_ = {metric: OrderedDict() for metric in self.monitor}

        self.parent_iter_ = 0
        self.child_iter_ = 0
        self.val_scores_ = np.full(self.parent_iterations, np.nan)

        metrics_best = np.full((self.parent_iterations, len(self.metrics_)), np.nan)
        self.metrics_best_ = pd.DataFrame(metrics_best, columns=list(self.metrics_.keys()))

        # each row indicates parent iteration, column indicates child iteration
        self.child_val_scores_ = np.full((self.parent_iterations,
                                          self.max_child_iters),
                                         np.nan)
        self.start_time_ = time.asctime()

        self.parent_suggestions_ = OrderedDict()

        # create container to store data for Taylor plot
        # It will be populated during postprocessing
        self.taylor_plot_inputs_ = {
            'simulations': {"test": {}},
            'observations': {"test": None}
        }

        self.parent_prefix_ = f"pipeline_opt_{dateandtime_now()}"

        self._save_config()  # will also make path if it does not already exists

        self._print_header()
        return

    def _print_header(self):
        # prints the first line on console
        formatter = "{:<5} {:<18} " + "{:<15} " * (len(self.metrics_))
        print(formatter.format(
            "Iter",
            self.eval_metric,
            *[k for k in self.metrics_.keys()])
        )

        return

    def fit(
            self,
            data: pd.DataFrame,
            previous_results=None
    ) -> "ai4water.hyperopt.HyperOpt":
        """
        Optimizes the pipeline for the given data.

        Parameters
        ----------
            data :
                A pandas dataframe which contains input and output features
            previous_results : dict, optional
                path of file which contains xy values.

        Returns
        --------
            an instance of ai4water.hyperopt.HyperOpt class which is used for
            optimization.
        """

        self.data_ = data

        self.reset()

        parent_opt = HyperOpt(
            self.parent_algorithm,
            param_space=self.space(),
            objective_fn=self.parent_objective,
            num_iterations=self.parent_iterations,
            opt_path=self.path,
            verbosity = 0
        )

        if previous_results is not None:
            parent_opt.add_previous_results(previous_results)

        res = parent_opt.fit()

        setattr(self, 'optimizer', parent_opt)

        self.save_results()

        self.report()

        self._save_config()

        return res

    def parent_objective(
            self,
            **suggestions
    ) -> float:
        """
        objective function for parent hpo loop.

        This objective fuction is to optimize transformations for each input
        feature and the model.

        Parameters
        ----------
            **suggestions :
                key word arguments consisting of suggested transformation for each
                input feature and the model to use
        """

        self.parent_iter_ += 1

        self.CHILD_PREFIX = f"{self.parent_iter_}_{dateandtime_now()}"
        # self.seed = np.random.randint(0, 10000, 1).item()

        if self._optimize_model:
            model = suggestions['model']
        else:
            model = self._model

        x_trnas, y_trans = self._cook_transformations(suggestions)

        if self._child_iters[model]>0:
            # optimize the hyperparas of model using child objective
            opt_paras = self.optimize_model_paras(
                model,
                x_transformations=x_trnas,
                y_transformations=y_trans or None
            )
        else:
            opt_paras = {}

        if self.category == "DL":
            model_config = DL_MODELS[model](**opt_paras)
        else:
            model_config = {model: opt_paras}

        # fit the model with optimized hyperparameters and suggested transformations
        _model = self._build_model(
            model=model_config,
            val_metric=self.eval_metric,
            x_transformation=x_trnas,
            y_transformation=y_trans,
            prefix=f"{self.parent_prefix_}{SEP}{self.CHILD_PREFIX}",
        )

        self.parent_suggestions_[self.parent_iter_] = {
            # 'seed': self.seed,
            'x_transformation': x_trnas,
            'y_transformation': y_trans,
            'model': {model: opt_paras},
            'path': _model.path
        }

        val_score = self._fit_and_eval(_model, self.cv_parent_hpo,
                                       self.eval_metric)

        # calculate all additional performance metrics which are being monitored
        t, p = _model.predict(data='validation', return_true=True,
                             process_results=False)
        errors = self.Metrics(t, p, remove_zero=True, remove_neg=True)

        for k, v in self.metrics_.items():
            pm = getattr(errors, k)()
            v[self.parent_iter_] = pm

            func = compare_func1(METRIC_TYPES[k])
            best_so_far = func(self.metrics_best_.loc[:self.parent_iter_, k])

            best_so_far = fill_val(METRIC_TYPES[k], best_so_far)

            func = compare_func(METRIC_TYPES[k])
            if func(pm, best_so_far):

                self.metrics_best_.loc[self.parent_iter_-1, k] = pm

        self.val_scores_[self.parent_iter_ - 1] = val_score  # -1 because array indexing starts from 0

        _val_score = val_score if np.less_equal(val_score, np.nanmin(self.val_scores_[:self.parent_iter_ ])) else ''

        # print the merics being monitored
        # we fill the nan in metrics_best_ with '' so that it does not gen printed
        formatter = "{:<5} {:<18.3} " + "{:<15.7} " * (len(self.metrics_))
        print(formatter.format(
            self.parent_iter_,
            _val_score,
            *self.metrics_best_.loc[self.parent_iter_-1].fillna('').values.tolist())
        )

        return val_score

    def optimize_model_paras(
            self,
            model: str,
            x_transformations: list,
            y_transformations: list
    ) -> dict:
        """optimizes hyperparameters of a model"""

        def child_objective(**suggestions):
            """objective function for optimization of model parameters"""

            self.child_iter_ += 1

            if self.category == "DL":
                model_config = DL_MODELS[model](**suggestions)
            else:
                model_config = {model: suggestions}

            # build child model
            _model = self._build_model(
                model=model_config,
                val_metric=self.eval_metric,
                x_transformation=x_transformations,
                y_transformation=y_transformations,
                prefix=f"{self.parent_prefix_}{SEP}{self.CHILD_PREFIX}",
            )

            val_score = self._fit_and_eval(_model, self.cv_child_hpo,
                                           self.eval_metric)

            # populate all child val scores
            self.child_val_scores_[self.parent_iter_-1, self.child_iter_-1] = val_score

            return val_score

        # make space
        child_space = self.model_space[model]['param_space']
        self.child_iter_ = 0  # before starting child hpo, reset iteration counter

        optimizer = HyperOpt(
            self.child_algorithm,
            objective_fn=child_objective,
            num_iterations=self._child_iters[model],
            param_space=child_space,
            verbosity=0,
            process_results=False,
            opt_path=os.path.join(self.path, self.CHILD_PREFIX),
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
                    t = {"method": method, "features": self._groups[feature]}

                    # some preprocessing is required for log based transformations
                    if method.startswith("log"):
                        t["treat_negatives"] = True
                        t["replace_zeros"] = True
                    elif method == "box-cox":
                        t["treat_negatives"] = True
                        t["replace_zeros"] = True
                    elif method == "sqrt":
                        t['treat_negatives'] = True

                    if feature in self.inputs_to_transform:
                        x_transformations.append(t)
                    else:
                        y_transformations.append(t)

        return x_transformations, y_transformations

    def _build_model(
            self,
            model: dict,
            val_metric: str,
            x_transformation,
            y_transformation,
            prefix: Union[str, None]
    ) -> Model:
        """
        build the ai4water Model. When overwriting this method, the user
        must return an instance of ai4water's Model_ class.

        .. Model:
            https://ai4water.readthedocs.io/en/master/model.html#ai4water._main.BaseModel
        """
        model = Model(
            model=model,
            verbosity=0,
            val_metric=val_metric,
            x_transformation=x_transformation,
            y_transformation=y_transformation,
            # seed=self.seed,
            prefix=prefix,
            **self.model_kwargs
        )
        return model

    def _fit_and_eval(self, model, cross_validate, metric_to_compute) -> float:
        """fits the model and evaluates it and returns the score"""
        if cross_validate:
            # val_score will be obtained by performing cross validation
            val_score = model.cross_val_score(data=self.data_, refit=True)
        else:
            # train the model and evaluate it to calculate val_score
            model.fit(data=self.data_)
            val_score = eval_model_manually(model, metric_to_compute, self.Metrics)

        return val_score

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
            the best value of performance metric acheived
        """
        if metric_name not in self.metrics_:
            raise MetricNotMonitored(metric_name, list(self.metrics_.keys()))

        if METRIC_TYPES[metric_name] == "min":
            return np.nanmin(list(self.metrics_[metric_name].values())).item()
        else:
            return np.nanmax(list(self.metrics_[metric_name].values())).item()

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
        """

        metric_name = metric_name or self.eval_metric

        if metric_name not in self.metrics_:
            raise MetricNotMonitored(metric_name, list(self.metrics_.keys()))

        if METRIC_TYPES[metric_name] == "min":
            idx = np.nanargmin(list(self.metrics_[metric_name].values()))
        else:
            idx = np.nanargmax(list(self.metrics_[metric_name].values()))

        return int(idx + 1)

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
            a dictionary with follwoing keys

                - ``path`` path where the model is saved on disk
                - ``model`` name of model
                - ``x_transfromations``  transformations for the input data
                - ``y_transformations`` transformations for the target data
        """

        metric_name = metric_name or self.eval_metric

        idx = self.get_best_metric_iteration(metric_name)

        return self.parent_suggestions_[idx]

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

                x_transformation
                y_transformation
                model
                path
        """

        metric_name = metric_name or self.eval_metric

        # checks if the given metric is a valid metric or not
        if metric_name not in self.metrics_:
            raise MetricNotMonitored(metric_name, self.metrics_.keys())

        # initialize an empty dictionary to store model parameters
        model_container = {}

        for iter_num, iter_suggestions in self.parent_suggestions_.items():
            # iter_suggestion is a dictionary and it contains four keys
            model = iter_suggestions['model']
            # model is dictionary, whose key is the model_name and values
            # are model configuration

            if model_name in model:
                # find out the metric value at iter_num
                metric_val = self.metrics_[metric_name][iter_num]
                metric_val = round(metric_val, 4)

                model_container[metric_val] = iter_suggestions

        if len(model_container) == 0:
            raise ModelNotUsedError(model_name)

        # sorting the container w.r.t given metric_name
        sorted_container = sorted(model_container.items())

        return sorted_container[-1]

    def baseline_results(self, data=None) -> tuple:
        """
        Returns default performance of all models.

        It runs all the models with their default parameters and without
        any x and y transformation. These results can be considered as
        baseline results and can be compared with optimized model's results.
        The model is trained on 'training'+'validation' data.

        Parameters
        ----------
            data :
                If given, will override data given during .fit call.

        Returns
        -------
        tuple
            a tuple of two dictionaries.
            - a dictionary of val_scores on test data for each model
            - a dictionary of metrics being monitored for  each model on test data.
        """
        val_scores = {}
        metrics = {}

        for _model in self.models:

            if self.category == "DL":
                _model = DL_MODELS[_model]

            # build model
            model = self._build_model(
                model=_model,
                val_metric=self.eval_metric,
                prefix=f"{self.parent_prefix_}{SEP}baselines",
                x_transformation=None,
                y_transformation=None
            )

            if data is None:
                data = self.data_
            model.fit_on_all_training_data(data=data)

            t, p = model.predict(return_true=True)
            errors = self.Metrics(t, p)
            val_scores[_model] = getattr(errors, self.eval_metric)()

            _metrics = {}
            for m in self.metrics_.keys():
                _metrics[m] = getattr(errors, m)()
            metrics[_model] = _metrics

        results = {
            'val_scores': val_scores,
            'metrics': metrics
        }

        with open(os.path.join(self.path, "baselines", "results.json"), 'w') as fp:
            json.dump(results, fp, sort_keys=True, indent=4)

        return val_scores, metrics

    def dumbbell_plot(
            self,
            metric_name: str = None,
            figsize: tuple = None,
            show: bool = True,
            save: bool = True
    ) -> plt.Axes:
        """
        Generate Dumbbell_ plot as comparison of baseline models with
        optimized models.

        Parameters
        ----------
            metric_name: str
                The name of metric with respect to which the models have
                to be compared. If not given, the evaluation metric is used.
            figsize: tuple
                If given, plot will be generated of this size.
            show : bool
                whether to show the plot or not
            save
                By default True. If False, function will not save the
                resultant plot in current working directory.

        Returns
        -------
            matplotlib Axes

        .. _Dumbbell:
            https://easy-mpl.readthedocs.io/en/latest/plots.html#easy_mpl.dumbbell_plot
        """
        metric_name = metric_name or self.eval_metric

        _, bl_results = self.baseline_results()

        bl_models = {}
        for k, v in bl_results.items():
            bl_models[k] = v[metric_name]

        optimized_models = {}

        for model_name in self.models:
            try:
                metric_val, _ = self.get_best_pipeline_by_model(model_name, metric_name)
            # the model was not used so consider the baseline result as optimzied
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

        _labels = df['models'].tolist()
        labels = []

        for label in _labels:
            if label.endswith('Regressor'):
                label = label.replace('Regressor', '')
            elif label.endswith('Classifier'):
                label = label.replace('Classifier', '')
            labels.append(label)

        df.to_csv(os.path.join(self.path, f"dumbell_{metric_name}_data.csv"))

        baseline = np.where(df['baseline']<-1.0, -1.0, df['baseline'])
        fig, ax = plt.subplots(figsize=figsize)
        ax = dumbbell_plot(baseline,
                           df['optimized'],
                           labels=labels,
                           show=False,
                           xlabel=metric_name,
                           ylabel="Models",
                           ax=ax
                           )

        fpath = os.path.join(self.path, f"dumbell_{metric_name}")
        if save:
            plt.savefig(fpath, dpi=300, bbox_inches='tight')
        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def taylor_plot(
            self,
            plot_bias: bool = True,
            figsize: tuple = None,
            show: bool = True,
            save: bool = True,
            **kwargs
    ) -> plt.Figure:
        """
        makes `Taylor`_'s plot using the best version of each model.
        The number of models in taylor plot will be equal to the number
        of models which have been considered by the model.

        Parameters
        ----------
            plot_bias : bool, optional
                whether to plot the bias or not
            figsize : tuple, optional
                a tuple determining figure size
            show : bool, optional
                whether to show the plot or not
            save : bool, optional
                whether to save the plot or not
            **kwargs :
                any additional keyword arguments for taylor_plot function of `easy_mpl`_.

        Returns
        -------
            matplotlib.pyplot.Figure

        .. _easy_mpl:
            https://github.com/Sara-Iftikhar/easy_mpl#taylor_plot

        .. _Taylor:
            https://doi.org/10.1029/2000JD900719
        """

        if self.taylor_plot_inputs_['observations']['test'] is None:
            self.bfe_all_best_models()

        ax = taylor_plot(
            show=False,
            save=False,
            plot_bias=plot_bias,
            cont_kws={},
            grid_kws={},
            figsize=figsize,
            **self.taylor_plot_inputs_,  # simulations and trues as keyword arguments
            **kwargs
        )
        fname = os.path.join(self.path, "taylor_plot")

        if save:
            plt.savefig(fname, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        # save taylor plot data as csv file, first make a dataframe
        sim = self.taylor_plot_inputs_['simulations']['test']
        data = np.column_stack([v.reshape(-1, ) for v in sim.values()])
        df = pd.DataFrame(data, columns=list(sim.keys()))
        df['observations'] = self.taylor_plot_inputs_['observations']['test']

        df.to_csv(os.path.join(self.path, "taylor_data.csv"))

        return ax

    def save_results(self)->None:
        """
        saves the results. It is called automatically at the end of optimization.
        It saves tried models and transformations at each step as json file
        with the name ``parent_suggestions.json``.

        An ``errors.csv`` file is saved which contains validation peformance of the models
        at each optimization iteration with respect to all metrics being monitored.

        The performance of each model during child optimization iteration is saved as a csv
        file with the name ``child_val_scores.csv``.

        All of these results are saved in pl.path folder.

        Returns
        -------
        None
        """
        self.end_time_ = time.asctime()

        # save parent_suggestions
        parent_suggestions = jsonize(self.parent_suggestions_)
        with open(os.path.join(self.path, "parent_suggestions.json"), "w") as fp:
            json.dump(parent_suggestions, fp, sort_keys=True)

        # make a 2d array of all erros being monitored.
        errors = np.column_stack([list(v.values()) for v in self.metrics_.values()])
        # add val_scores as new columns
        errors = np.column_stack([errors, self.val_scores_])
        # save the errors being monitored
        fpath = os.path.join(self.path, "errors.csv")
        pd.DataFrame(errors,
                     columns=list(self.metrics_.keys()) + ['val_scores']
                     ).to_csv(fpath)

        # save results of child iterations as csv file
        fpath = os.path.join(self.path, "child_val_scores.csv")
        pd.DataFrame(
            self.child_val_scores_,
            columns=[f'child_iter_{i}' for i in range(self.max_child_iters)]).to_csv(fpath)
        return

    def metric_report(self, metric_name: str) -> str:
        """report with respect to one performance metric"""
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
        """makes the reprot and writes it in text form"""
        st_time = self.start_time_
        en_time = getattr(self, "end_time_", time.asctime())

        num_models = len(self.models)
        text = f"""
    The optization started at {st_time} and ended at {en_time} after 
completing {self.parent_iter_} iterations. The optimization considered {num_models} models. 
        """

        if self.parent_iter_ < self.parent_iterations:
            text += f"""
The given parent iterations were {self.parent_iterations} but optimization stopped early"""

        for metric in self.metrics_.keys():
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
    def _version_info() -> dict:
        """returns version of the third party libraries used"""
        import ai4water
        import SeqMetrics
        import matplotlib
        import sklearn
        import easy_mpl
        versions = dict()
        versions['ai4water'] = ai4water.__version__
        versions['SeqMetrics'] = SeqMetrics.__version__
        versions['easy_mpl'] = easy_mpl.__version__
        versions['numpy'] = np.__version__
        versions['pandas'] = pd.__version__
        versions['matplotlib'] = matplotlib.__version__
        versions['sklearn'] = sklearn.__version__
        versions['python'] = sys.version

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

        cls.start_time_ = config['runtime_attrs']

        path = os.path.dirname(config_file)
        fpath = os.path.join(path, "parent_suggestions.json")
        if os.path.exists(fpath):
            with open(fpath, "r") as fp:
                parent_suggestions = json.load(fp)

            cls.parent_suggestions_ = parent_suggestions
            cls.parent_iter_ = len(parent_suggestions)

        fpath = os.path.join(path, "errors.csv")
        if os.path.exists(fpath):
            errors = pd.read_csv(fpath)
            cls.metrics_ = errors.to_dict()

        return cls(**config['init_paras'], **model_kwargs)

    @classmethod
    def from_config(cls, config: dict) -> "OptimizePipeline":
        """Builds the class from config dictionary

        Parameters
        ----------
            config : dict
                a dictionary which contains `init_paras` key.

        Returns
        -------
            an instance of OptimizePipeline class
        """
        return cls(**config['init_paras'])

    def be_best_model_from_config(
            self,
            metric_name: str = None,
            model_name: str = None
    )->Model:
        """Build and Evaluate the best model with respect to metric *from config*.

        Parameters
        ----------
            metric_name : str
                the metric with respect to which the best model is fetched
                and then built/evaluated. If not given, the best model is
                built/evaluated with respect to evaluation metric.
            model_name : str, optional
                If given, the best version of this model will be fetched and built.
                The 'best' will be decided based upon `metric_name`

        Returns
        -------
            an instance of trained ai4water Model
        """

        metric_name = metric_name or self.eval_metric

        if model_name:
            _, pipeline = self.get_best_pipeline_by_model(model_name, metric_name)
        else:
            pipeline = self.get_best_pipeline_by_metric(metric_name=metric_name)

        cpath = os.path.join(pipeline['path'], "config.json")
        model = Model.from_config_file(cpath)

        wpath = os.path.join(pipeline['path'], "weights",
                             list(pipeline['model'].keys())[0])
        model.update_weights(wpath)

        self._populate_results(model)

        return model

    def bfe_model_from_scratch(
            self,
            iter_num: int,
    )->Model:
        """
        Builds, trains and evalutes the model from a specific iteration.
        The model is trained on 'training'+'validation' data.

        Parameters
        ----------
            iter_num : int
                iteration number from which to choose the model

        Returns
        -------
            an instance of trained ai4water Model
        """
        pipeline = self.parent_suggestions_[iter_num]
        prefix = f"{self.path}{SEP}results_from_scratch{SEP}iteration_{iter_num}"

        model = self._build_and_eval_from_scratch(
            model=pipeline['model'],
            x_transformation=pipeline['x_transformation'],
            y_transformation=pipeline['y_transformation'],
            prefix=prefix
        )
        return model

    def bfe_best_model_from_scratch(
            self,
            metric_name: str = None,
            model_name: str = None
    )->Model:
        """
        Builds, Trains and Evaluates the **best model** with respect to metric from
        scratch. The model is trained on 'training'+'validation' data. Running
        this mothod will also populate ``taylor_plot_inputs_`` dictionary.

        Parameters
        ----------
            metric_name : str
                the metric with respect to which the best model is searched
                and then built/trained/evaluated. If None, the best model is
                chosen based on the evaluation metric.
            model_name : str, optional
                If given, the best version of this model will be found and built.
                The 'best' will be decided based upon `metric_name`

        Returns
        -------
            an instance of trained ai4water Model
        """

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

        model = self._build_and_eval_from_scratch(
            model=pipeline['model'],
            x_transformation=pipeline['x_transformation'],
            y_transformation=pipeline['y_transformation'],
            prefix=prefix)

        return model

    def _build_and_eval_from_scratch(
            self,
            model,
            x_transformation,
            y_transformation,
            prefix,
            model_name=None,
    ) -> "Model":
        """builds and evaluates the model from scratch. If model_name is given,
        model's predictions are saved in 'taylor_plot_inputs_' dictionary
        """
        model = self._build_model(model=model,
                                  x_transformation=x_transformation,
                                  y_transformation=y_transformation,
                                  prefix=prefix,
                                  val_metric=self.eval_metric)

        model.fit_on_all_training_data(data=self.data_)

        model.dh_.to_disk(model.path)

        self._populate_results(model, model_name=model_name)

        return model

    def _populate_results(
            self,
            model,
            model_name=None
    ) -> None:
        """evaluates/makes predictions from model on traiing/validation/test data.
        if model_name is given, model's predictions are saved in 'taylor_plot_inputs_'
        dictionary
        """
        model.predict(data='training', metrics="all")
        model.predict(data='validation', metrics="all")
        t, p = model.predict(data='test', metrics="all", return_true=True)

        if model_name:
            self.taylor_plot_inputs_['observations']['test'] = t
            self.taylor_plot_inputs_['simulations']['test'][model_name] = p

        return

    def _evaluate_model(
            self,
            model: Callable,
            metric_name: str = None,
            data = 'test',
            x = None,
            y = None
    )->float:
        """Evaluates the ai4water's Model on the data for the metric.

        Parameters
        ----------
            model :
                an instance of ai4water's Model class
            metric_name : str
                name of performance metric. If not given, evaluation metric
                is used.
            data :
                either a dataframe or a string. If string, it must be one of
                ``training``, ``validation`` and ``test``.
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
            t, p = model.predict(x=x, y=y, process_results=False, return_true=True)
        else:
            assert x is None
            t, p = model.predict(data=data, process_results=False, return_true=True)

        errors = self.Metrics(t, p)

        return getattr(errors, metric_name)()

    def bfe_all_best_models(
            self,
            metric_name: str = None
    ) -> None:
        """
        builds, trains and evaluates best versions of all the models.
        The model is trained on 'training'+'validation' data.

        Parameters
        ----------
            metric_name : str
                the name of metric to determine best version of a model. If not given,
                parent_val_metric will be used.

        Returns
        -------
        None

        """

        met_name = metric_name or self.eval_metric

        for model in self.models:

            try:
                metric_val, pipeline = self.get_best_pipeline_by_model(model, met_name)
            except ModelNotUsedError:
                continue

            prefix = f"{self.path}{SEP}results_from_scratch{SEP}{met_name}_{metric_val}_{model}"

            _ = self._build_and_eval_from_scratch(
                model=pipeline['model'],
                x_transformation=pipeline['x_transformation'],
                y_transformation=pipeline['y_transformation'],
                prefix=prefix,
                model_name=model
            )

        return

    def post_fit(self, show:bool = True) -> None:
        """post processing of results to draw dumbell plot and taylor plot.

        Parameters
        ----------
        show : bool, optional
            whether to show the plots or not

        Returns
        -------
        None

        """

        self.bfe_all_best_models()
        self.dumbbell_plot(metric_name=self.eval_metric, show=show)

        # following plots only make sense if more than one models are tried
        if self._optimize_model:
            self.taylor_plot(show=show)
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
        """

        metric_name = metric_name or self.eval_metric

        models = {}

        for model in self.models:

            try:
                metric_val, _ = self.get_best_pipeline_by_model(model, metric_name)
                models[model] = metric_val
            except ModelNotUsedError:
                continue

        labels = []

        for label in models.keys():
            if label.endswith('Regressor'):
                label = label.replace('Regressor', '')
            elif label.endswith('Classifier'):
                label = label.replace('Classifier', '')
            labels.append(label)

        plt.close('all')
        if plot_type == "circular":
            ax = circular_bar_plot(np.array(list(models.values())),
                                   labels,
                                   sort=True,
                                   show=False,
                                   **kwargs)
        else:
            ax = bar_chart(list(models.values()),
                           labels,
                           sort=True,
                           show=False,
                           **kwargs)

        fpath = os.path.join(self.path, f"{plot_type}_plot_wrt_{metric_name}")
        plt.savefig(fpath, dpi=300, bbox_inches='tight')

        if show:
            plt.tight_layout()
            plt.show()

        return ax


def eval_model_manually(model, metric: str, Metrics) -> float:
    """evaluates the model"""
    # make prediction on validation data
    t, p = model.predict(data='validation', return_true=True,
                         process_results=False)
    
    p = p.reshape(-1, 1)  # TODO, for cls, Metrics do not accept (n,) array

    errors = Metrics(t, p, remove_zero=True, remove_neg=True)
    val_score = getattr(errors, metric)()

    metric_type = METRIC_TYPES.get(metric, 'min')

    # the optimization will always solve minimization problem so if
    # the metric is to be maximized change the val_score accordingly
    if metric_type != "min":
        val_score = 1.0 - val_score

    # val_score can be None/nan/inf
    if not math.isfinite(val_score):
        val_score = 1.0

    return val_score


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