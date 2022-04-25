"""
===========
regression
===========
"""

from ai4water.datasets import busan_beach
from skopt.plots import plot_objective
from autotab import OptimizePipeline

data = busan_beach()

pl = OptimizePipeline(
    inputs_to_transform=data.columns.tolist()[0:-1],
    parent_iterations=30,
    child_iterations=5,
    parent_algorithm='bayes',
    child_algorithm='random',
    eval_metric='r2_score',
    monitor=['r2', 'nse'],
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
    split_random=True,
    train_fraction=1.0,
)

results = pl.fit(data=data)

##############################################

pl.optimizer._plot_convergence(save=False)

##############################################

pl.optimizer._plot_parallel_coords(figsize=(16, 8), save=False)

##############################################

pl.optimizer._plot_distributions(save=False)

##############################################3

pl.optimizer.plot_importance(save=False)

###########################################

_ = plot_objective(results)

###########################################

pl.optimizer._plot_evaluations(save=False)


###########################################

pl.optimizer._plot_edf(save=False)

##############################################

pl.bfe_all_best_models()

##############################################

pl.dumbbell_plot()

##############################################

pl.dumbbell_plot('r2')
##############################################

pl.taylor_plot()

##############################################

pl.compare_models()

##############################################

pl.compare_models(plot_type="bar_chart")

#################################################

pl.cleanup()