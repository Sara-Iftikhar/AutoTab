"""
===========
regression
===========
"""

from ai4water.datasets import busan_beach
from skopt.plots import plot_objective
from autotab import OptimizePipeline

data = busan_beach()

print(data.shape)
###########################################

print(data.head())
##########################################


kws = {
'inputs_to_transform': data.columns.tolist()[0:-1],
'outputs_to_transform': data.columns.tolist()[-1:],
'parent_iterations': 10,
'child_iterations': 0,  # don't optimize hyperparamters only for demonstration
'parent_algorithm': 'bayes',
'child_algorithm': 'random',
'eval_metric': 'rmse',
    'cv_parent_hpo': True,
    'cross_validator': {"KFold": {"n_splits": 5}},
'monitor': ['r2', 'r2_score'],
'models': [ "LinearRegression",
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

'input_features': data.columns.tolist()[0:-1],
'output_features': data.columns.tolist()[-1:],
'split_random': True,
    'seed':2809,
}

with OptimizePipeline(**kws) as pl:
    pl.remove_transformation('box-cox')
    pl._pp_plots = ["regression", "prediction", "residual", "edf"]

    pl.change_transformation_behavior('yeo-johnson', {'pre_center': True})

    results = pl.fit(data=data, process_results=False)

##############################################

# plot the convergence plot to illustrate how much improvement occurred w.r.t evaluation metric

pl.optimizer_._plot_convergence(save=False, grid=True)

# %%
pl.optimizer_._plot_convergence(save=False, original=True, grid=True)

##############################################

# show searched space

pl.optimizer_._plot_parallel_coords(figsize=(16, 8), save=False)

##############################################

pl.optimizer_._plot_distributions(save=False)

##############################################3

pl.optimizer_.plot_importance(save=False)

###########################################

# plot first order and second order partial dependence plots gaussian process
_ = plot_objective(results)

###########################################

pl.optimizer_._plot_evaluations(save=False)

###########################################

pl.optimizer_._plot_edf(save=False)

##############################################

pl.bfe_all_best_models(data=data)

##############################################

pl.dumbbell_plot(data=data, save=False, upper_limit=1e15)

##############################################

pl.dumbbell_plot(data=data, metric_name='r2', save=False)

##############################################

pl.taylor_plot(data=data, save=False)

##############################################

pl.compare_models()

##############################################

# compare the performance of models
pl.compare_models(plot_type="bar_chart")

##############################################

# compare the performance of models w.r.t R2
pl.compare_models("r2", plot_type="bar_chart")

# %%
model = pl.be_best_model_from_config(data=data, metric_name="r2_score")

# %%
model.evaluate_on_test_data(data=data, metrics="r2_score")
# %%
model = pl.bfe_best_model_from_scratch(metric_name='r2_score', data=data)

# %%
model.evaluate_on_training_data(data=data, metrics="r2_score")

# %%
model.evaluate_on_test_data(data=data, metrics="r2_score")

#################################################

print(f"all results are save in {pl.path} folder")

#################################################

# remove all the files/folders which are now nomore required.
pl.cleanup()