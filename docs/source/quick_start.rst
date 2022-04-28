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
    ...         eval_metric='mse',
    ...         monitor=['r2', 'nse'],
    ...         input_features=input_features,
    ...         output_features=output_features,
    ...         split_random=True,
    ...         train_fraction=1.0,
    ...         epochs=100,
    ...     )

    >>>     pl.fit(data=data)

    >>>     pl.post_fit()

machine learning models (classification)
==============================================================

This covers all scikit-learng models, catboost, lightgbm and xgboost

.. code-block:: python

    >>> from ai4water.datasets import busan_beach
    >>> from autotab import OptimizePipeline

    >>> data = busan_beach()
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
    ...         train_fraction=1.0,
    ...         epochs=100,
    ...     )

    >>>     pl.fit(data=data)

    >>>     pl.post_fit()

deep learning models (regression)
=======================================================

This covers MLP, LSTM, CNN, CNNLSTM, TFT, TCN, LSTMAutoEncoder for regression .
Each model can consist of stacks of layers. For example MLP can consist of
stacks of Dense layers. The number of layers are also optimized.

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
    ...         train_fraction=1.0,
    ...         epochs=100,
    ...     )

    >>>     pl.fit(data=data)

    >>>     pl.post_fit()

deep learning models (classification)
===========================================================

This covers MLP, LSTM, CNN, CNNLSTM, TFT, TCN, LSTMAutoEncoder for classification problem.
Each model can consist of stacks of layers. For example MLP can consist of
stacks of Dense layers. The number of layers are also optimized.

.. code-block:: python

    >>> from ai4water.datasets import busan_beach
    >>> from autotab import OptimizePipeline

    >>> data = busan_beach()
    >>> input_features = data.columns.tolist()[0:-1]
    >>> output_features = data.columns.tolist()[-1:]

    >>> pl = OptimizePipeline(
    ...          mode="classification",
    ...          eval_metric="accuracy",
    ...         inputs_to_transform=input_features,
    ...         outputs_to_transform=output_features,
    ...         models=["MLP", "LSTM", "CNN", "CNNLSTM", "TFT", "TCN", "LSTMAutoEncoder"],
    ...         parent_iterations=30,
    ...         child_iterations=12,
    ...         parent_algorithm='bayes',
    ...         child_algorithm='bayes',
    ...         monitor=['accuracy'],
    ...         input_features=input_features,
    ...         output_features=output_features,
    ...         split_random=True,
    ...         train_fraction=1.0,
    ...         epochs=100,
    ...     )

    >>>     pl.fit(data=data)

    >>>     pl.post_fit()

deep learning models (multi-class classification)
===========================================================
