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


# Example
Click here to [open this example in binder](https://nbviewer.jupyter.org/github/AtrCheema/AI4Water/blob/dev/examples/paper/compare_ml.ipynb)
or cick here to
<a href="https://colab.research.google.com/github/Sara-Iftikhar/AutoTab/blob/master/notebooks/regression_cv.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="open in colab"/></a>

```python
from ai4water.datasets import busan_beach
from skopt.plots import plot_objective
from autotab import OptimizePipeline

data = busan_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]

pl = OptimizePipeline(
    inputs_to_transform=data.columns.tolist()[0:-1],
    parent_iterations=400,
    child_iterations=20,
    parent_algorithm='bayes',
    child_algorithm="random",
    cv_parent_hpo=True,
    eval_metric='mse',
    monitor=['r2', 'nse'],
    input_transformations = ['minmax', 'zscore', 'log', 'log10', 'sqrt', 'robust', 'quantile'],
    output_transformations = ['minmax', 'zscore', 'log', 'log10', 'sqrt', 'robust', 'quantile'],
    models=[ "LinearRegression",
            "LassoLars",
            "Lasso",
            "RandomForestRegressor",
            "HistGradientBoostingRegressor",
             "CatBoostRegressor",
             "XGBRegressor",
             "LGBMRegressor",
             "GradientBoostingRegressor",
             "ExtraTreeRegressor",
             "ExtraTreesRegressor"
             ],

    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    cross_validator={"KFold": {"n_splits": 5}},
    split_random=True,
)
```

```python
results = pl.fit(data=data, process_results=False)
```

```python
pl.optimizer._plot_convergence(save=False)
```

```python
pl.optimizer._plot_parallel_coords(figsize=(16, 8), save=False)
```

```python
_ = pl.optimizer._plot_distributions(save=False)
```

```python
pl.optimizer.plot_importance(save=False)
```

```python
pl.optimizer.plot_importance(save=False, plot_type="bar")
```

```python
_ = plot_objective(results)
```

```python
pl.optimizer._plot_evaluations(save=False)
```

```python
pl.optimizer._plot_edf(save=False)
```

```python
pl.dumbbell_plot(data=data)
```

```python
pl.dumbbell_plot(data, 'r2')
```

```python
pl.taylor_plot(data=data, save=False, figsize=(6,6))
```

```python
pl.compare_models()
```

```python
pl.compare_models(plot_type="bar_chart")
```

```python
pl.compare_models("r2", plot_type="bar_chart")
```

```python
model = pl.bfe_best_model_from_scratch(data=data)
```

```python
pl.evaluate_model(model, data=data)
```

```python
pl.evaluate_model(model, data, 'nse')
```

```python
pl.evaluate_model(model, data, 'r2')
```


```python
model = pl.bfe_best_model_from_scratch(data, 'r2')
```

```python
pl.evaluate_model(model, data, 'r2')
```

```python
print(f"all results are save in {pl.path} folder")
```