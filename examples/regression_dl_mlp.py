"""
==========================
regression (deep learning)
==========================
"""

from ai4water.datasets import busan_beach
from skopt.plots import plot_objective
from autotab import OptimizePipeline

data = busan_beach()
data.shape

##############################################

pl = OptimizePipeline(
    inputs_to_transform=data.columns.tolist()[0:-1],
    outputs_to_transform=data.columns.tolist()[-1:],
    parent_iterations=30,
    child_iterations=0,
    parent_algorithm='bayes',
    child_algorithm="random",
    eval_metric='mse',
    monitor=['r2', 'nse'],
    input_transformations=['minmax', 'zscore', 'log', 'log10', 'sqrt', 'robust', 'quantile', 'none', 'scale'],
    output_transformations=['minmax', 'zscore', 'log', 'log10', 'sqrt', 'robust', 'quantile', 'none', 'scale'],
    models=["MLP"],

    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True,
    category="DL",
    epochs=100,
    train_fraction=1.0,
    val_fraction=0.3,

)

##############################################

pl._version_info()

##############################################
results = pl.fit(data=data, process_results=False)

pl.optimizer._plot_convergence(save=False)

##############################################

pl.optimizer._plot_parallel_coords(figsize=(16, 8), save=False)

##############################################

_ = pl.optimizer._plot_distributions(save=False)

##############################################

pl.optimizer.plot_importance(save=False)

##############################################

pl.optimizer.plot_importance(save=False, plot_type="bar")

##############################################

_ = plot_objective(results)

##############################################

pl.optimizer._plot_evaluations(save=False)

##############################################

pl.optimizer._plot_edf(save=False)

##############################################

pl.get_best_pipeline_by_metric()

##############################################

model = pl.bfe_best_model_from_scratch(data=data, fit_on_all_train_data=False)

##############################################

pl.evaluate_model(model, data=data)

##############################################

pl.evaluate_model(model, data, 'nse')

##############################################

pl.evaluate_model(model, data, 'r2')

##############################################

model = pl.bfe_best_model_from_scratch(data, 'r2')

##############################################

pl.evaluate_model(model, data, 'r2')

##############################################

print(f"all results are save in {pl.path} folder")
