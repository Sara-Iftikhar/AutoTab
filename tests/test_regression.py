import unittest
import site
import warnings

site.addsitedir(r"E:\AA\automl")


def warn(*args, **kwargs):
    pass


warnings.warn = warn

from automl import OptimizePipeline

from ai4water.datasets import arg_beach


inputs = ['tide_cm', 'wat_temp_c', 'sal_psu',
          'pcp3_mm',  # 'pcp6_mm', 'pcp12_mm',
          'pcp_mm', 'air_temp_c', 'rel_hum']

data = arg_beach(inputs=inputs)


def run_basic(parent_algorithm="random", child_algorithm="random",
              parent_iterations=4, child_iterations=4,
              parent_val_metric="mse", child_val_metric="mse",
              **kwargs
              ):

    pl = OptimizePipeline(
        inputs_to_transform=inputs,
        parent_iterations=parent_iterations,
        child_iterations=child_iterations,
        parent_algorithm=parent_algorithm,
        child_algorithm=child_algorithm,
        parent_val_metric=parent_val_metric,
        child_val_metric=child_val_metric,
        monitor=['r2', 'nse'],
        models=[
            "LinearRegression",
            "LassoLars",
            "Lasso",
            "PoissonRegressor"
        ],
        input_features=data.columns.tolist()[0:-1],
        output_features=data.columns.tolist()[-1:],
        train_data="random",
        val_data="same",
        val_fraction=0.0,
        **kwargs
    )

    results = pl.fit(
        data=data
    )
    return pl


class TestMetrics(unittest.TestCase):
    """test different val_metrics for parent and child hpos"""
    def test_r2_as_val_metric(self):

        run_basic(parent_val_metric="r2", child_val_metric="r2",
                  parent_iterations=10, child_iterations=25)

        return


class TestRegression(unittest.TestCase):

    def test_basic(self):
        pl = run_basic()

        itr = pl.get_best_metric_iteration('nse')
        assert isinstance(itr, int)

        nse = pl.get_best_metric('nse')
        assert isinstance(nse, float)

        best_pipeline = pl.get_best_pipeline_by_metric('nse')
        assert isinstance(best_pipeline, dict)
        for k in ['x_transformation', 'y_transformation', 'model', 'path']:
            assert k in best_pipeline

        best_nse, best_pl = pl.get_best_pipeline_by_model("LinearRegression", 'nse')
        assert isinstance(best_nse, float)

        assert isinstance(best_pl, dict)
        for k in ['x_transformation', 'y_transformation', 'model', 'path']:
            assert k in best_pipeline
        return

    def test_y_transformations(self):
        output_transformations = ['sqrt', 'log', 'log10']
        pl = run_basic(parent_algorithm="bayes",
                            parent_iterations=12,
                            outputs_to_transform='tetx_coppml',
                            output_transformations=output_transformations
                            )

        y_transformation = pl.parent_suggestions[1]['y_transformation'][0]['method']
        assert y_transformation in output_transformations
        return


if __name__ == "__main__":
    unittest.main()
