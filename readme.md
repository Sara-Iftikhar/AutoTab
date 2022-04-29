[![Documentation Status](https://readthedocs.org/projects/autotab/badge/?version=latest)](https://autotab.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/autotab.svg)](https://badge.fury.io/py/autotab)

# autotab

optimize pipeline for any machine learning mdoel using hierarchical optimization 
method for tabular datasets.

# Installation

This package can be installed using pip from pypi using following command

    pip install autotab
    
or using github link for the latest code

	python -m pip install git+https://github.com/Sara-Iftikhar/autotab.git

or using setup file, go to folder where this repoitory is downloaded

    python setup.py install

## optimizing pipeline for machine learning models

This covers all scikit-learng models, catboost, lightgbm and xgboost

```python
from ai4water.datasets import busan_beach
from autotab import OptimizePipeline

data = busan_beach()

pl = OptimizePipeline(
    inputs_to_transform=data.columns.tolist()[0:-1],
    parent_iterations=30,
    child_iterations=12,
    parent_algorithm='bayes',
    child_algorithm='bayes',
    eval_metric='mse',
    monitor=['r2', 'nse'],

    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True,
    train_fraction=1.0,
)

pl.fit(data=data)

pl.post_fit()
```

## optimizing pipeline for deep learning models
This covers MLP, LSTM, CNN, CNNLSTM, TFT, TCN, LSTMAutoEncoder.
Each model can consist of stacks of layers. For example MLP can consist of 
stacks of Dense layers. The number of layers are also optimized.

```python
from ai4water.datasets import busan_beach
from autotab import OptimizePipeline

data = busan_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]

pl = OptimizePipeline(
    inputs_to_transform=input_features,
    outputs_to_transform=output_features,    
    models=["MLP"],
    parent_iterations=30,
    child_iterations=12,
    parent_algorithm='bayes',
    child_algorithm='bayes',
    eval_metric='mse',
    monitor=['r2', 'nse'],

    input_features=input_features,
    output_features=output_features,
    split_random=True,
    train_fraction=1.0,
    epochs=100, 
)

pl.fit(data=data)

pl.post_fit(data=data)
```


For classification make following adjustments while initializing the pipeline
```python
pl = OptimizePipeline(
    mode="classification",
    eval_metric="accuracy",

    models=["ExtraTreeClassifier",
            "RandomForestClassifier",
            # "XGBClassifier",
            # "CatBoostClassifier",
            # "LGBMClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
            "ExtraTreesClassifier",
            "RidgeClassifier",
            "SVC",
            "KNeighborsClassifier",
            ],

)
```