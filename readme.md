# automl

optimize pipeline for any machine learning mdoel using hierarchical optimization method

# Installation

This package can be installed using pip from pypi using following command

    pip install automl
    
or using github link for the latest code

	python -m pip install git+https://github.com/Sara-Iftikhar/automl.git

or using setup file, go to folder where repo is downloaded

    python setup.py install

```python
from ai4water.datasets import busan_beach

from automl import OptimizePipeline

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