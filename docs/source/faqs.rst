Frequently Asked Questions
**************************

I don't want to optimize preprocessing step
===========================================
If you dont want any preprocessing steps, keep `inputs_to_transform` and `outputs_to_transform` arguments
equal to None or an empty list. In this way transformations will not be optimized for both inputs and targets.
As shown in below example,

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(
    ...     inputs_to_transform=[],
    ...     outputs_to_transform=[],
    ...     )
    >>> results = pl.fit(data=data)

I don't want to optimize hyperprameters of the models
=====================================================
If you dont want to optimize hyperparameters of the models,
the child iterations needs to be set to zero. As shown in below example,

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(child_iterations=0)
    >>> results = pl.fit(data=data)

I don't want to optimize model selection
========================================
If you dont want to optimize model selection,
keep `models` argument equals to None or an empty list. As shown in below example,

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(models=[])
    >>> results = pl.fit(data=data)

I want to optimize pipeline for only one model
==============================================
You can set `models` parameter to the desired model.
In this way, pipeline will be optimized by using only one model.
For example, in the following code, only `AdaBoostRegressor` will
be used in pipeline optimization.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(
    >>> models=["AdaBoostRegressor"])
    >>> results = pl.fit(data=data)

I want to optimize pipeline for only selected models
====================================================
List the desired models in `models` as a
list. In this way, pipeline will be optimized
for the selected models.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(
    >>> models=[
    ...     "GradientBoostingRegressor",
    ...    "HistGradientBoostingRegressor",
    ...    "DecisionTreeRegressor",
    ...    "CatBoostRegressor",
    ...    "ExtraTreeRegressor",
    ...    "ExtraTreesRegressor",
    ...    ])
    >>> results = pl.fit(data=data)

Can I use different optimization algorithms for parent and child iterations
===========================================================================
Different optimization algorithms can be set by `parent_algorithm` and
`child_algorithm`.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(
    ...        parent_algorithm="bayes",
    ...        child_algorithm="bayes"
    ...    )
    >>> results = pl.fit(data=data)

How to monitor more than one metrics
====================================
The metrics you want to monitor can be given to `monitor` as a list.
In this example, two metrics NSE and R2 are being monitored.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(monitor=['r2', 'nse'])
    >>> results = pl.fit(data=data)

How to find best/optimized pipeline
===================================
There are two functions to get best pipeline after optimization.
They are `get_best_pipeline_by_metric` which returns optimized pipeline
according to given metric. On the other hand, `get_best_pipeline_by_model`
gives us best pipeline according to given model.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline()
    >>> results = pl.fit(data=data)
    >>> pl.get_best_pipeline_by_metric(metric_name='nse')
    >>> pl.get_best_pipeline_by_model(model_name='RandomForest_regressor')

Find best pipeline with respect to a specific (performance) metric
==================================================================
`get_best_pipeline_by_metric` function can be used to get best pipeline with
respect to a specific (performance) metric.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline()
    >>> results = pl.fit(data=data)
    >>> pl.get_best_pipeline_by_metric(metric_name='nse')

Find best pipeline with respect to a particular model
=====================================================
`get_best_pipeline_by_model` returns the best pipeline with respect to a particular model and
performance metric. The metric must be recorded i.e. must be given as
`monitor` argument.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline()
    >>> results = pl.fit(data=data)
    >>> pl.get_best_pipeline_by_model(model_name='RandomForest_regressor')

Change search space of a particular model
=========================================
`update_model_space` updates or changes the search space
of an already existing model.

.. code-block:: python

    >>> pl = OptimizePipeline(...)
    >>> rf_space = {'max_depth': [5,10, 15, 20],
    >>>          'n_models': [5,10, 15, 20]}
    >>> pl.update_model_space({"RandomForestRegressor": rf_space})

consider only selected transformations
======================================
Selected transformations can be given to `input_transformations`
and `output_transformations`. In this way, the given transformations
will be used for preprocessing steps.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(
    ...                    input_transformations=['minmax', 'log', 'zscore'],
    ...                    output_transformations=['quantile', 'box-cox', 'yeo-johnson']
    ...                       )
    >>> results = pl.fit(data=data)

do not optimize transformations for input data
==============================================
If you dont want to optimize transformations for input data,
keep `inputs_to_transform` argument equal to None or an empty list.
In this way transformations will not be optimized for input data.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(inputs_to_transform=[])
    >>> results = pl.fit(data=data)

change number of optimization iterations of a specific model
============================================================
Number of optimization iterations for a particular model
can be changed by using `change_child_iteration` function after initializing the
OptimizePipeline class. For example we may want to change the
child hpo iterations for one or more models. We may want to
run only 10 iterations for LinearRegression but 40
iterations for XGBRegressor. In such a case we can use this function to
modify child hpo iterations for one or more models. The iterations for all
the remaining models will remain same as defined by the user at the start.

.. code-block:: python

    >>> pl = OptimizePipeline(...)
    >>> pl.change_child_iteration({"XGBRegressor": 10})
    #If we want to change iterations for more than one models
    >>> pl.change_child_iteration(({"XGBRegressor": 30,
    >>>                             "RandomForestRegressor": 20}))

where are all the results stored
================================
The results are stored in folder named results in the
current working directory. The exact path of stored results can
be checked by printing `model.path`.

what if optimization stops in the middle
========================================
If optimization stops in the middle due to an error,
remaining results can be saved and analyzed by using these commands.

.. code-block:: python
    >>> pl.save_results()
    >>> pl.post_fit()

what is ``config.json`` file
============================
`config.json` is a simply plain text file that stores information
about pipeline such as parameters, pipeline configuration. The pipeline
can be built again by using `from_config` function.

How to include results from previous runs
=========================================
The path to `iterations.json` from previous pipeline results
has to be given to fit function in order to include results
from previous runs.

.. code-block:: python

    >>> from _main import OptimizePipeline
    >>> pl = OptimizePipeline(inputs_to_transform=[])
    >>> results = pl.fit(data=data, previous_results=fpath)

What versions of underlying libraries do this package depends
=============================================================
Currently `AutoTab` is strongly coupled with a ML python framework
`AI4Water`, whose version should be 1.1 or greater. Another dependency
is `h5py` which does not have any specific version requirement.
