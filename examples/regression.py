"""
===========
regression
===========
"""

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

pl.fit(data=data)

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