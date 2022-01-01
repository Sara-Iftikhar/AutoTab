
import site
site.addsitedir('E:\\AA\\AI4Water')

import os
import gc
import math
from typing import Union, Dict
from collections import OrderedDict

import numpy as np
import pandas as pd

from ai4water import Model
from ai4water._optimize import make_space
from ai4water.hyperopt import Categorical, HyperOpt, Integer
from ai4water.experiments.utils import regression_space, classification_space
from ai4water.utils.utils import dateandtime_now
from ai4water.postprocessing.SeqMetrics import RegressionMetrics, ClassificationMetrics
from ai4water.utils.utils import MATRIC_TYPES


SEP = os.sep

DEFAULT_TRANSFORMATIONS = [
    "minmax", "center", "scale", "zscore", "box-cox", "yeo-johnson",
    "quantile", "robust", "log", "log2", "log10", "sqrt", "none",
              ]


class OptimizePipeline(object):
    """
    optimizes model/estimator to use, its hyperparameters and preprocessing
    operation to be performed on features. It consists of two hpo loops. The
    parent or outer loop optimizes preprocessing/feature engineering, feature
    selection and model selection while the child hpo loop optimizes hyperparmeters
    of child hpo loop.

    Attributes
    -----------

    - metrics

    - parent_suggestions:
        an ordered dictionary of suggestions to the parent objective function
        during parent hpo loop

    - child_val_metrics:
        a numpy array containing val_metrics of all child hpo loops

    - optimizer
        an instance of ai4water.hyperopt.HyperOpt for parent optimization

    Note
    -----
    This optimizationa always sovlves a minimization problem even if the val_metric
    is r2.
    """

    def __init__(
            self,
            inputs_to_transform,
            input_transformations: Union[list, dict] = None,
            outputs_to_transform=None,
            output_transformations: Union[list] = None,
            models: list = None,
            parent_iterations: int = 100,
            child_iterations: int = 25,
            parent_algorithm: str = "bayes",
            child_algorithm: str = "bayes",
            parent_val_metric: str = "mse",
            child_val_metric: str = "mse",
            parent_cross_validator: str = None,
            child_cross_validator: str = None,
            monitor: Union[list, str] = "r2",
            mode: str = "regression",
            **model_kws
    ):
        """
        initializes

        Arguments:
            inputs_to_transform:
                Input features on which feature engineering/transformation is to
                be applied. By default all input features are considered.
            input_transformations:
                The transformations to be considered for input features.

                If list, then it will be the names of transformations to be considered for
                all input features. By default following transformations are considered

                    - `minmax`  rescale from 0 to 1
                    - `center`    center the data by subtracting mean from it
                    - `scale`     scale the data by dividing it with its standard deviation
                    - `zscore`    first performs centering and then scaling
                    - `box-cox`
                    - `yeo-johnson`
                    - `quantile`
                    - `robust`
                    - `log`
                    - `log2`
                    - `log10`
                    - `sqrt`    square root

                The user can however, specify list of transformations to be considered for
                each input feature. In such a case, this argument must be a dictionary
                whose keys are names of input features and values are list of transformations.

            outputs_to_transform:
                Output features on which feature engineering/transformation is to
                be applied.
            output_transformations:
                The transformations to be considered for outputs/targets. By default
                following transformations are considered for outputs

                    - `log`
                    - `log10`
                    - `sqrt`
                    - `log2`
            models:
                The models to consider during optimzatino.
            parent_iterations:
                Number of iterations for parent optimization loop
            child_iterations:
                Number of iterations for child optimization loop
            parent_algorithm:
                Algorithm for optimization of parent optimzation
            child_algorithm:
                Algorithm for optimization of child optimization
            parent_val_metric:
                Validation metric to calculate val_score in parent objective function
            child_val_metric:
                Validation metric to calculate val_score in child objective function
            parent_cross_validator:
                cross validator to be used in parent objective function. If None,
                then val_score will be calculated on validation data
            child_cross_validator:
                cross validator to be used in child objective function. If None,
                then val_score will be caclulated on validation data
            monitor:
                Nmaes of performance metrics to monitor in parent hpo loop
            mode:
                whether this is a `regression` problem or `classification`
            model_kws:
                any additional key word arguments for ai4water's Model

        """
        self.inp_to_transform = inputs_to_transform
        self.inp_transformations = input_transformations
        self.out_to_transform = outputs_to_transform
        self.y_transformations = output_transformations
        self.models = models
        self.parent_iters = parent_iterations
        self.child_iters = child_iterations
        self.parent_algo = parent_algorithm
        self.child_algo = child_algorithm
        self.parent_val_metric = parent_val_metric
        self.child_val_metric = child_val_metric
        self.parent_cv = parent_cross_validator
        self.child_cv = child_cross_validator
        self.mode = mode
        self.model_kwargs = model_kws

        # self.seed = None

        if isinstance(monitor, str):
            monitor = [monitor]
        assert isinstance(monitor, list)

        self.metrics = {metric: OrderedDict() for metric in monitor}

        self.parent_suggestions = OrderedDict()

        self.parent_prefix = f"pipeline_opt_{dateandtime_now()}"

        self.child_val_metrics = np.full((self.parent_iters, self.child_iters),
                                         np.nan)

        if self.mode == "regression":
            self.estimator_space = regression_space(num_samples=10)
        else:
            self.estimator_space = classification_space(num_samples=10)

    @property
    def path(self):
        return os.path.join(os.getcwd(), "results", self.parent_prefix)

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, x):
        if x is None:
            if self.mode == "regression":
                x = list(regression_space(2).keys())
            else:
                x = list(classification_space(2).keys())
        self._models = x

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
            return self.model_kwargs['output_features']
        else:
            raise ValueError

    def space(self) -> list:
        """makes the parameter space for parent hpo"""

        append = None
        if self.inp_transformations is None:
            inp_transformations = DEFAULT_TRANSFORMATIONS
        elif isinstance(self.inp_transformations, list):
            inp_transformations = self.inp_transformations
        else:
            inp_transformations = DEFAULT_TRANSFORMATIONS
            assert isinstance(self.inp_transformations, dict)
            append = {}
            for feature, transformation in self.inp_transformations.items():
                assert isinstance(transformation, list)
                append[feature] = transformation

        if self.y_transformations:

            if isinstance(self.y_transformations, list):
                assert all([t in DEFAULT_TRANSFORMATIONS for t in self.y_transformations]), f"""
                transformations must be one of {DEFAULT_TRANSFORMATIONS}"""

                for out in self.output_features:
                    append[out] = self.y_transformations

            else:
                assert isinstance(self.y_transformations, dict)
                for out_feature, out_transformations in self.y_transformations.items():

                    assert out_feature in self.output_features
                    assert isinstance(out_transformations, list)
                    assert all(
                        [t in DEFAULT_TRANSFORMATIONS for t in self.y_transformations]), f"""
                        transformations must be one of {DEFAULT_TRANSFORMATIONS}"""
                    append[out_feature] = out_transformations

        sp = make_space(self.inp_to_transform,
                        categories=inp_transformations,
                        append=append)

        algos = Categorical(self.models, name="estimator")
        sp = sp + [algos]

        return sp

    def reset(self):

        self.parent_iter_ = 0
        self.child_iter_ = 0
        self.val_scores_ = OrderedDict()

        return

    def fit(
            self,
            data: pd.DataFrame,
            previous_results=None
    ) -> "ai4water.hyperopt.HyperOpt":
        """

        Arguments:
            data:
                A pandas dataframe
            previous_results:
                path of file which contains xy values.
        Returns:
            an instance of ai4water.hyperopt.HyperOpt class which is used for optimization.
        """

        self.data = data

        self.reset()

        parent_opt = HyperOpt(
            self.parent_algo,
            param_space=self.space(),
            objective_fn=self.parent_objective,
            num_iterations=self.parent_iters,
            opt_path=self.path
        )

        if previous_results is not None:
            parent_opt.add_previous_results(previous_results)

        formatter = "{:<5} {:<18} " + "{:<15} " * (len(self.metrics))
        print(formatter.format(
            "Iter",
            self.parent_val_metric,
            *[k for k in self.metrics.keys()])
        )

        res = parent_opt.fit()

        setattr(self, 'optimizer', parent_opt)

        # make a 2d array of all erros being monitored.
        errors = np.column_stack([list(v.values()) for v in self.metrics.values()])
        # add val_scores as new columns
        errors = np.column_stack([errors, list(self.val_scores_.values())])
        # save the errors being monitored
        fpath = os.path.join(self.path, "errors.csv")
        pd.DataFrame(errors,
                     columns=list(self.metrics.keys()) + ['val_scores']
                     ).to_csv(fpath)

        # save results of child iterations as csv file
        fpath = os.path.join(self.path, "child_iters.csv")
        pd.DataFrame(self.child_val_metrics,
                     columns=[f'iter_{i}' for i in range(self.child_iters)]).to_csv(fpath)
        return res

    def parent_objective(
            self,
            **suggestions
    ) -> float:
        """objective function for parent hpo loop.
        This objective fuction is to optimize transformations for each input
        feature and the model.

        Arguments:
            suggestions:
                key word arguments consisting of suggested transformation for each
                input feature and the model to use
        """

        self.parent_iter_ += 1

        # self.seed = np.random.randint(0, 10000, 1).item()

        # container for transformations for all features
        x_transformations = []
        y_transformations = []

        for feature, method in suggestions.items():

            if feature in self.data:
                if method == "none":  # don't do anything with this feature
                    pass
                else:
                    # get the relevant transformation for this feature
                    t = {"method": method, "features": [feature]}

                    # some preprocessing is required for log based transformations
                    if method.startswith("log"):
                        t["treat_negatives"] = True
                        t["replace_zeros"] = True
                    elif method == "box-cox":
                        t["treat_negatives"] = True
                        t["replace_zeros"] = True
                    elif method == "sqrt":
                        t['treat_negatives'] = True

                    if feature in self.input_features:
                        x_transformations.append(t)
                    else:
                        y_transformations.append(t)

        # optimize the hyperparas of estimator using child objective
        opt_paras = self.optimize_estimator_paras(
            suggestions['estimator'],
            x_transformations=x_transformations,
            y_transformations=y_transformations or None
        )

        self.parent_suggestions[self.parent_iter_] = {
            # 'seed': self.seed,
            'x_transformation': x_transformations,
            'y_transformation': y_transformations,
            'estimator_paras': opt_paras
        }

        # fit the model with optimized hyperparameters and suggested transformations
        model = Model(
            model={suggestions["estimator"]: opt_paras},
            val_metric=self.parent_val_metric,
            verbosity=0,
            # seed=self.seed,
            x_transformation=x_transformations,
            y_transformation=y_transformations,
            prefix=self.parent_prefix,
            **self.model_kwargs
        )

        if self.parent_cv is None:  # train the model and evaluate it to calculate val_score
            # train the model
            model.fit(data=self.data)

            val_score = eval_model_manually(model, self.parent_val_metric, self.Metrics)
        else:  # val_score will be obtained by performing cross validation
            val_score = model.cross_val_score(data=self.data)

        # calculate all additional performance metrics which are being monitored
        t, p = model.predict(data='validation', return_true=True, process_results=False)
        errors = RegressionMetrics(t, p, remove_zero=True, remove_neg=True)

        for k, v in self.metrics.items():
            v[self.parent_iter_] = getattr(errors, k)()

        self.val_scores_[self.parent_iter_] = val_score

        # print the merics being monitored
        formatter = "{:<5} {:<18.3f} " + "{:<15.7f} " * (len(self.metrics))
        print(formatter.format(
            self.parent_iter_,
            val_score,
            *[v[self.parent_iter_] for v in self.metrics.values()])
        )

        return val_score

    def optimize_estimator_paras(
            self, estimator: str,
            x_transformations: list,
            y_transformations: list
    ) -> dict:
        """optimizes hyperparameters of an estimator"""

        CHILD_PREFIX = f"{self.child_iter_}_{dateandtime_now()}"

        def child_objective(**suggestions):
            """objective function for optimization of estimator parameters"""

            self.child_iter_ += 1

            # build child model
            model = Model(
                model={estimator: suggestions},
                verbosity=0,
                val_metric=self.child_val_metric,
                x_transformation=x_transformations,
                y_transformation=y_transformations,
                # seed=self.seed,
                prefix=f"{self.parent_prefix}{SEP}{CHILD_PREFIX}",
                **self.model_kwargs
            )

            if self.child_cv is None:
                # fit child model
                model.fit(data=self.data)
                val_score = eval_model_manually(model, self.child_val_metric, self.Metrics)
            else:
                val_score = model.cross_val_score(data=self.data)

            # populate all child val scores
            self.child_val_metrics[self.parent_iter_-1, self.child_iter_-1] = val_score

            return val_score

        # make space
        child_space = self.estimator_space[estimator]['param_space']
        self.child_iter_ = 0  # before starting child hpo, reset iteration counter

        optimizer = HyperOpt(
            self.child_algo,
            objective_fn=child_objective,
            num_iterations=self.child_iters,
            param_space=child_space,
            verbosity=0,
            process_results=False,
            opt_path=os.path.join(self.path, CHILD_PREFIX),
        )

        optimizer.fit()

        # free memory if possible
        gc.collect()

        # return the optimized parameters
        return optimizer.best_paras()


def eval_model_manually(model, metric: str, Metrics) -> float:
    """evaluates the model"""
    # make prediction on validation data
    t, p = model.predict(data='validation', return_true=True, process_results=False)
    errors = Metrics(t, p, remove_zero=True, remove_neg=True)
    val_score = getattr(errors, metric)()

    metric_type = MATRIC_TYPES.get(metric, 'min')

    # the optimization will always solve minimization problem so if
    # the metric is to be maximized change the val_score accordingly
    if metric_type != "min":
        val_score = 1.0 - val_score

    # val_score can be None/nan/inf
    if not math.isfinite(val_score):
        val_score = 1.0

    return val_score
