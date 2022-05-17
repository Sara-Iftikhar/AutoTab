quick start
***********

This page describes optimization of pipeline for different problems and using different
models.

Optimize pipeline for machine learning models (regression)
==========================================================

This covers all scikit-learng models, catboost, lightgbm and xgboost

.. code-block:: python

    >>> from ai4water.datasets import busan_beach
    >>> from autotab import OptimizePipeline

    >>> data = busan_beach()
    >>> input_features = data.columns.tolist()[0:-1]
    >>> output_features = data.columns.tolist()[-1:]

    >>> pl = OptimizePipeline(
    ...         inputs_to_transform=input_features,
    ...         outputs_to_transform=output_features,
    ...             models=["LinearRegression",
    ...        "LassoLars",
    ...        "Lasso",
    ...        "RandomForestRegressor",
    ...        "HistGradientBoostingRegressor",
    ...         "CatBoostRegressor",
    ...         "XGBRegressor",
    ...         "LGBMRegressor",
    ...         "GradientBoostingRegressor",
    ...         "ExtraTreeRegressor",
    ...         "ExtraTreesRegressor"
    ...                         ],
    ...         parent_iterations=30,
    ...         child_iterations=12,
    ...         parent_algorithm='bayes',
    ...         child_algorithm='bayes',
    ...         eval_metric='mse',
    ...         monitor=['r2', 'nse'],
    ...         input_features=input_features,
    ...         output_features=output_features,
    ...         split_random=True,
    ...     )

    >>> pl.fit(data=data)

    >>> pl.post_fit(data=data)

machine learning models (classification)
==============================================================

This covers all scikit-learng models, catboost, lightgbm and xgboost

.. code-block:: python

    >>> from ai4water.datasets import MtropicsLaos
    >>> from autotab import OptimizePipeline

    >>> data = MtropicsLaos().make_classification(lookback_steps=1)
    >>> input_features = data.columns.tolist()[0:-1]
    >>> output_features = data.columns.tolist()[-1:]

    >>> pl = OptimizePipeline(
    ...          mode="classification",
    ...          eval_metric="accuracy",
    ...         inputs_to_transform=input_features,
    ...         outputs_to_transform=output_features,
    ...             models=["ExtraTreeClassifier",
    ...                         "RandomForestClassifier",
    ...                         "XGBClassifier",
    ...                         "CatBoostClassifier",
    ...                         "LGBMClassifier",
    ...                         "GradientBoostingClassifier",
    ...                         "HistGradientBoostingClassifier",
    ...                         "ExtraTreesClassifier",
    ...                         "RidgeClassifier",
    ...                         "SVC",
    ...                         "KNeighborsClassifier",
    ...                         ],
    ...         parent_iterations=30,
    ...         child_iterations=12,
    ...         parent_algorithm='bayes',
    ...         child_algorithm='bayes',
    ...         monitor=['accuracy'],
    ...         input_features=input_features,
    ...         output_features=output_features,
    ...         split_random=True,
    ...     )

    >>> pl.fit(data=data)

    >>> pl.post_fit(data=data)

deep learning models (regression)
=================================

This covers MLP, LSTM, CNN, CNNLSTM, TFT, TCN, LSTMAutoEncoder for regression .
Each model can consist of stacks of layers. For example MLP can consist of
stacks of Dense layers. The number of layers are also optimized. When using
deep learning models, also set the value fo ``epochs`` because the default
value is 14 which is too small for a deep learning model. Also consider
setting values for ``batch_size`` and ``lr``.

.. code-block:: python

    >>> from ai4water.datasets import busan_beach
    >>> from autotab import OptimizePipeline

    >>> data = busan_beach()
    >>> input_features = data.columns.tolist()[0:-1]
    >>> output_features = data.columns.tolist()[-1:]

    >>> pl = OptimizePipeline(
    ...         inputs_to_transform=input_features,
    ...         outputs_to_transform=output_features,
    ...         models=["MLP", "LSTM", "CNN", "CNNLSTM", "TFT", "TCN", "LSTMAutoEncoder"],
    ...         parent_iterations=30,
    ...         child_iterations=12,
    ...         parent_algorithm='bayes',
    ...         child_algorithm='bayes',
    ...         eval_metric='mse',
    ...         monitor=['r2', 'nse'],
    ...         input_features=input_features,
    ...         output_features=output_features,
    ...         split_random=True,
    ...         epochs=100,
    ...         category="DL",
    ...         ts_args={"lookback": 14},
    ...     )

    >>> pl.fit(data=data)

    >>> pl.post_fit(data=data)

deep learning models (classification)
=====================================

This covers MLP, LSTM, CNN, CNNLSTM, TFT, TCN, LSTMAutoEncoder for classification problem.
Each model can consist of stacks of layers. For example MLP can consist of
stacks of Dense layers. The number of layers are also optimized.

.. code-block:: python

    >>> from ai4water.datasets import MtropicsLaos
    >>> from autotab import OptimizePipeline

    >>> data = MtropicsLaos().make_classification(lookback_steps=1,)
    >>> input_features = data.columns.tolist()[0:-1]
    >>> output_features = data.columns.tolist()[-1:]

    >>> pl = OptimizePipeline(
    ...         category="DL",
    ...         mode="classification",
    ...         eval_metric="accuracy",
    ...         inputs_to_transform=input_features,
    ...         outputs_to_transform=output_features,
    ...         models=["MLP", "CNN"],
    ...         parent_iterations=30,
    ...         child_iterations=12,
    ...         parent_algorithm='bayes',
    ...         child_algorithm='bayes',
    ...         monitor=['f1_score'],
    ...         input_features=input_features,
    ...         output_features=output_features,
    ...         split_random=True,
    ...         epochs=100,
    ...     )

    >>> pl.fit(data=data)

    >>> pl.post_fit(data=data)

deep learning models (multi-class classification)
===========================================================

For multi-class classification with neural networks, we must set
``num_classes`` argument to some value greater than 2.

.. code-block:: python
    >>> from autotab import OptimizePipeline
    >>> pl = OptimizePipeline(models=[
    >>>         "MLP",
    >>>     ],
    >>>         input_features=multi_cls_input_features,
    >>>         output_features=multi_cls_output_features,
    >>>         parent_algorithm="bayes",
    >>>         loss="categorical_crossentropy",
    >>>         parent_iterations=10,
    >>>         child_iterations=0,
    >>>         epochs=20,
    >>>         category="DL",
    >>>         mode="classification",
    >>>         num_classes = 4,
    >>>         eval_metric="accuracy",
    >>>         monitor="f1_score",
    >>>     )
    >>> pl.fit(data=multi_cls_data,)

Check ClassificationMetrics class of SeqMetrics library for the name of metrics
which can be used for monitoring