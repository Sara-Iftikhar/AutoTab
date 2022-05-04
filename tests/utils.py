
from ai4water.datasets import busan_beach

from autotab import OptimizePipeline

inputs = ['tide_cm', 'wat_temp_c', 'sal_psu',
          'pcp3_mm',  # 'pcp6_mm', 'pcp12_mm',
          'pcp_mm', 'air_temp_c', 'rel_hum']

rgr_data = busan_beach(inputs=inputs)


def build_basic(parent_algorithm="random",
                child_algorithm="random",
                parent_iterations=4,
                child_iterations=4,
                eval_metric="mse",
                models = None,
                train_fraction=0.7,
                **kwargs
              ):

    input_features = rgr_data.columns.tolist()[0:-1]
    if 'input_features' in kwargs:
        input_features = kwargs.pop('input_features')

    output_features = rgr_data.columns.tolist()[-1:]
    if 'output_features' in kwargs:
        output_features = kwargs.pop('output_features')

    inputs_to_transform = input_features
    if 'inputs_to_transform' in kwargs:
        inputs_to_transform = kwargs.pop('inputs_to_transform')

    if 'monitor' in kwargs:
        monitor = kwargs.pop('monitor')
    else:
        monitor = ['r2', 'nse']

    pl = OptimizePipeline(
        inputs_to_transform=inputs_to_transform,
        parent_iterations=parent_iterations,
        child_iterations=child_iterations,
        parent_algorithm=parent_algorithm,
        child_algorithm=child_algorithm,
        eval_metric=eval_metric,
        monitor=monitor,
        models=models or [
            "LinearRegression",
            "LassoLars",
            "Lasso",
            "RandomForestRegressor"
        ],
        input_features=input_features,
        output_features=output_features,
        split_random=True,
        train_fraction=train_fraction,
        **kwargs
    )


    return pl


def run_basic(data=None, process_results=True, **kwargs):

    if data is None:
        data = rgr_data

    pl = build_basic(**kwargs)
    pl.fit(
        data=data,
        process_results=process_results
    )

    return pl


