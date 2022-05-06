[![Documentation Status](https://readthedocs.org/projects/autotab/badge/?version=latest)](https://autotab.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/autotab.svg)](https://badge.fury.io/py/autotab)
[![DOI](https://zenodo.org/badge/433432707.svg)](https://zenodo.org/badge/latestdoi/433432707)
[![Downloads](https://pepy.tech/badge/autotab)](https://pepy.tech/project/autotab)

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
Click here to 
[![badge](https://img.shields.io/badge/open%20in-binder-579ACA.svg?logo=data:image/png;base64,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)](https://nbviewer.jupyter.org/github/Sara-Iftikhar/AutoTab/blob/master/notebooks/regression_cv.ipynb)
or cick here to
<a href="https://colab.research.google.com/github/Sara-Iftikhar/AutoTab/blob/master/notebooks/regression_cv.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="open in colab"/></a>

```python
from ai4water.datasets import busan_beach
from skopt.plots import plot_objective
from autotab import OptimizePipeline

data = busan_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]

transformations = ['minmax', 'zscore', 'log', 'log10', 'sqrt', 'robust', 'quantile', 'none', 'scale']

pl = OptimizePipeline(
    inputs_to_transform=data.columns.tolist()[0:-1],
    parent_iterations=400,
    child_iterations=20,
    parent_algorithm='bayes',
    child_algorithm="random",
    cv_parent_hpo=True,
    eval_metric='mse',
    monitor=['r2', 'r2_score'],
    input_transformations = transformations,
    output_transformations = transformations,
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

get version information
```python
pl._version_info()
```

perform optimization 
```python
results = pl.fit(data=data, process_results=False)
```

print optimization report
```python
print(pl.report())
```

show convergence plot
```python
pl.optimizer_._plot_convergence(save=False)
```

```python
pl.optimizer_._plot_parallel_coords(figsize=(16, 8), save=False)
```

```python
_ = pl.optimizer_._plot_distributions(save=False)
```

```python
pl.optimizer_.plot_importance(save=False)
```

```python
pl.optimizer_.plot_importance(save=False, plot_type="bar")
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
pl.dumbbell_plot(data=data, metric_name='r2')
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

get best pipeline with respect to evaluation metric
```python
pl.get_best_pipeline_by_metric('r2')
```

build fit and evaluate the best pipeline
```python
model = pl.bfe_best_model_from_scratch(data=data)
```

```python
pl.evaluate_model(model, data=data)
```

```python
pl.evaluate_model(model, data=data, metric_name='r2_score')
```

```python
pl.evaluate_model(model, data=data, metric_name='r2')
```

get best pipeline with respect to $R^2$
```python
pl.get_best_pipeline_by_metric('r2')
```

```python
model = pl.bfe_best_model_from_scratch(data=data, metric_name='r2')
```

```python
pl.evaluate_model(model, data=data, metric_name='r2')
```

```python
print(f"all results are save in {pl.path} folder")
```