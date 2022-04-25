


from ai4water.datasets import busan_beach

from autotab import OptimizePipeline

inputs = ['tide_cm', 'wat_temp_c', 'sal_psu',
          'pcp3_mm',  # 'pcp6_mm', 'pcp12_mm',
          'pcp_mm', 'air_temp_c', 'rel_hum']

data = busan_beach(inputs=inputs)


def build_basic(parent_algorithm="random",
                child_algorithm="random",
                parent_iterations=4,
                child_iterations=4,
              eval_metric="mse",
                models = None,
              **kwargs
              ):

    inputs_to_transform = inputs
    if 'inputs_to_transform' in kwargs:
        inputs_to_transform = kwargs.pop('inputs_to_transform')
    pl = OptimizePipeline(
        inputs_to_transform=inputs_to_transform,
        parent_iterations=parent_iterations,
        child_iterations=child_iterations,
        parent_algorithm=parent_algorithm,
        child_algorithm=child_algorithm,
        eval_metric=eval_metric,
        monitor=['r2', 'nse'],
        models=models or [
            "LinearRegression",
            "LassoLars",
            "Lasso",
            "RandomForestRegressor"
        ],
        input_features=data.columns.tolist()[0:-1],
        output_features=data.columns.tolist()[-1:],
        split_random=True,
        train_fraction=1.0,
        **kwargs
    )


    return pl


def run_basic(**kwargs):

    pl = build_basic(**kwargs)
    pl.fit(
        data=data
    )

    return pl


