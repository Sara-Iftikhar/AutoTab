Frequently Asked Questions
**************************

What is difference between parent and child iterations/algorithm?
=================================================================
AutoTab operates based upon parent and child optimization iterations
The parent iteration is responsible for preprocessing step optimization
and model optimization. During each parent iteration, when the preprocessing
and model is selected/suggested by the algorithm for this iteration, the
child optimization loops starts. The job of child optimization loop is
to optimize hyperparameters of the selected/suggested model. The user can
specify any algorithm from following algorithms for parent and child optimization
algorithms.

    - bayes
    - random
    - grid
    - bayes_rf
    - tpe
    - atpe
    - cmaes

what splitting scheme is used
=============================
By default it is supposed that the data is split into 3 sets i.e. training, validation
and test sets. validation data is only used during pipeline optimization inside
``.fit`` method while the test data is only used after optimization. If you have
only two sets i.e. training and validation, set ``fit_on_all_train_data`` to False
during ``post_fit``

Is the pipeline optimized for test data or validation data?
===========================================================
The pipeline is optimized for validation data.

What transformations are considered by default?
===============================================
To find out the transformations being considered, you can
print the ``DEFAULT_TRANSFORMATIONS`` variable as below

.. code-block:: python

    >>> from autotab._main import DEFAULT_TRANSFORMATIONS
    >>> print(DEFAULT_TRANSFORMATIONS)

if you want to know the transformations being considered for a specific
feature in the pipeline then use following code

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(...)
    >>> pl.feature_transformations[feature_name]

I don't want to optimize preprocessing step
===========================================
If you dont want any preprocessing steps, keep `inputs_to_transform`
and `outputs_to_transform` arguments equal to None or an empty list.
In this way transformations will not be optimized for both inputs and targets.
As shown in below example,

.. code-block:: python

    >>> from autotab import OptimizePipeline
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

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(child_iterations=0)
    >>> results = pl.fit(data=data)

I don't want to optimize model selection
========================================
If you dont want to optimize model selection,
keep `models` argument equals to None or an empty list. As shown in below example,

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(models=[])
    >>> results = pl.fit(data=data)

I want to optimize pipeline for only one model
==============================================
You can set `models` parameter to the desired model.
In this way, pipeline will be optimized by using only one model.
For example, in the following code, only `AdaBoostRegressor` will
be used in pipeline optimization.

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(
    >>> models=["AdaBoostRegressor"])
    >>> results = pl.fit(data=data)

I want to optimize pipeline for only selected models
====================================================
List the desired models in `models` as a
list. In this way, pipeline will be optimized
for the selected models.

.. code-block:: python

    >>> from autotab import OptimizePipeline
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

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(
    ...        parent_algorithm="bayes",
    ...        child_algorithm="bayes"
    ...    )
    >>> results = pl.fit(data=data)

How to monitor more than one metrics
====================================
The metrics you want to monitor can be given to `monitor` as a list.
In this example, two metrics NSE and $R^2$ are being monitored.

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(monitor=['r2', 'nse'])
    >>> results = pl.fit(data=data)

How to find best/optimized pipeline
===================================
There are two functions to get best pipeline after optimization.
They are `get_best_pipeline_by_metric` which returns optimized pipeline
according to given metric. On the other hand, `get_best_pipeline_by_model`
gives us best pipeline according to given model.

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline()
    >>> results = pl.fit(data=data)
    >>> pl.get_best_pipeline_by_metric(metric_name='nse')
    >>> pl.get_best_pipeline_by_model(model_name='RandomForest_regressor')

Find best pipeline with respect to a specific (performance) metric
==================================================================
`get_best_pipeline_by_metric` function can be used to get best pipeline with
respect to a specific (performance) metric.

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline()
    >>> results = pl.fit(data=data)
    >>> pl.get_best_pipeline_by_metric(metric_name='nse')

Find best pipeline with respect to a particular model
=====================================================
`get_best_pipeline_by_model` returns the best pipeline with respect to a particular model and
performance metric. The metric must be recorded i.e. must be given as
`monitor` argument.

.. code-block:: python

    >>> from autotab import OptimizePipeline
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

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(
    ...                    input_transformations=['minmax', 'log', 'zscore'],
    ...                    output_transformations=['quantile', 'box-cox', 'yeo-johnson']
    ...                       )
    >>> results = pl.fit(data=data)

do not optimize transformations for input data
==============================================
If you dont want to optimize transformations for input data,
keep `inputs_to_transform` argument equal to empty list (**not None**).
In this way transformations will not be optimized for input data.

.. code-block:: python

    >>> from autotab import OptimizePipeline
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

    >>> from autotab import OptimizePipeline
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

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(...)
    >>> print(pl.path)

what if optimization stops in the middle
========================================
If optimization stops in the middle due to an error,
remaining results can be saved and analyzed by using these commands.

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(...)
    >>> pl.fit(data=data)
    .. # if above command stops in the middle due to an error
    >>> pl.save_results()
    >>> pl.post_fit(data=data)

what is ``config.json`` file
============================
`config.json` is a simply plain text file that stores information
about pipeline such as parameters, pipeline configuration. The pipeline
can be built again by using `from_config_file` method as shown below.

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> config_path = "path/to/config.json"
    >>> new_pipeline = OptimizePipeline.from_config_file(config_path)

How to include results from previous runs
=========================================
The path to `iterations.json` from previous pipeline results
has to be given to fit function in order to include results
from previous runs.

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(...)
    >>> fpath = "path/to/previous/iterations.json"
    >>> results = pl.fit(data=data, previous_results=fpath)

What versions of underlying libraries does this package depends on
==================================================================
Currently ``AutoTab`` is strongly coupled with the machine learning framework
`AI4Water`, whose version should be 1.6 or greater. Another optional dependency
is `h5py` which does not have any specific version requirement. It
is used for data storage. If it is not available, then data is stored in csv file
format.

how to use cross validation during pipeline optimization
========================================================
By default the pipeline is evaluated on the validation data according to ``eval_metric``.
However, you can choose to perform cross validation on child or parent or on both
iterations. To perform cross validation at parent iterations, set ``cv_parent_hpo``
to ``True``. Similarly to perform cross validation at child iteration, set ``cv_child_hpo``
to True. You must pass the ``cross_validator`` argument as well to determine
what kind of cross validation to be performed. Consider the following example
where cross validation is performed using KFold during parent iterations.

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(
    ...           ...    # add other arguments
    ...           cv_parent_hpo=True,
    ...           cross_validator={"KFold": {"n_splits": 5}},
    ...    )

Instead of ``KFold``, we can also choose ``LeaveOneOut``, or ``ShuffleSplit`` or ``TimeSeriesSplit``.


how to change search space for batch_size and learning rate
===========================================================
The learning_rate and batch_size search space is only active for
deep learning models i.e. when the ``category`` is "DL". The default
search space for learning rate is ``Real(low=1e-5, high=0.05, num_samples=10, name="lr")``
while for batch_size, the default search space is ``[8, 16, 32, 64]``.
We can change the default search space by making use of ``change_batch_size_space``
and ``change_lr_space`` methods after class initialization. For example we can
achieve a different batch_size search space as below

.. code-block:: python

    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(
    ...         ...  # add other arguments
    ...         category="DL
    ...           )
    ... pl.change_batch_size_space([32, 64, 128, 256, 512])

