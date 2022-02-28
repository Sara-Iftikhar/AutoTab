import os
import unittest
import site
import warnings

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
site.addsitedir(package_path)


def warn(*args, **kwargs):
    pass

warnings.warn = warn

import matplotlib.pyplot as plt

from ai4water.datasets import busan_beach

from automl import OptimizePipeline


inputs = ['tide_cm', 'wat_temp_c', 'sal_psu',
          'pcp3_mm',  # 'pcp6_mm', 'pcp12_mm',
          'pcp_mm', 'air_temp_c', 'rel_hum']

data = busan_beach(inputs=inputs)


def build_basic(parent_algorithm="random", child_algorithm="random",
              parent_iterations=4, child_iterations=4,
              eval_metric="mse",
                models = None,
              **kwargs
              ):

    pl = OptimizePipeline(
        inputs_to_transform=inputs,
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
    results = pl.fit(
        data=data
    )

    return pl


class TestMetrics(unittest.TestCase):
    """test different val_metrics for parent and child hpos"""
    def test_r2_as_val_metric(self):

        run_basic(eval_metric="r2",
                  parent_iterations=10, child_iterations=25)

        return


class TestRegression(unittest.TestCase):

    show = True

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

    def test_change_child_iter(self):
        """check that we can change the hpo iterations for a model"""
        pl = build_basic(models = ['Lasso', 'RandomForestRegressor'])
        pl.change_child_iteration({"RandomForestRegressor": 10})
        pl.fit(data=data)
        assert pl.child_val_scores_.shape[1] == 10
        return

    def test_remove_model(self):
        """test that we can remove a model which is already being considered"""
        pl = build_basic()
        pl.remove_model("LinearRegression")
        assert "LinearRegression" not in pl.models
        assert "LinearRegression" not in pl._child_iters
        assert "LinearRegression" not in pl.estimator_space
        return

    def test_change_model_space(self):
        """test that we can change space of a model"""
        pl = build_basic(models = ['Lasso', 'RandomForestRegressor'])
        space = {'max_depth': [5,10, 15, 20],
                 'n_estimators': [5,10, 15, 20]}
        pl.update_model_space({"RandomForestRegressor": space})
        pl.fit(data=data)
        assert len(pl.estimator_space['RandomForestRegressor']['param_space'])==2
        return

    def test_baseline_results(self):
        pl = build_basic()
        val_scores, metrics = pl.baseline_results(data=data)
        assert isinstance(val_scores, dict)
        assert len(val_scores) == 4
        assert isinstance(metrics, dict)
        assert len(metrics) == 4
        for k,v in metrics.items():
            assert len(v) == 2
        return

    def test_dumbbell_plot(self):
        pl = run_basic(parent_iterations=7,
            models=[
            "LinearRegression",
            "LassoLars",
            "Lasso",
            "RandomForestRegressor",
            "HistGradientBoostingRegressor",
        ])
        ax = pl.dumbbell_plot('r2', show=self.show)
        assert isinstance(ax, plt.Axes)
        pl.cleanup()
        return

    def test_bar_plot(self):
        pl = run_basic(parent_iterations=7,
            models=[
            "LinearRegression",
            "LassoLars",
            "Lasso",
            "RandomForestRegressor",
            "HistGradientBoostingRegressor",
        ])
        ax = pl.compare_models('r2', show=self.show)
        assert isinstance(ax, plt.Axes)
        pl.cleanup()
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

    def test_tpe(self):
        pl = run_basic(parent_algorithm="tpe",
        parent_iterations=12, eval_metric="nse",)
        pl.post_fit()
        pl.cleanup()

        return

    def test_single_model(self):
        pl = run_basic(models=["Lasso"])
        pl.post_fit()
        pl.cleanup()
        return


class TestRepeatition(unittest.TestCase):
    """Should raise value error if some model is repeated"""
    def test_model_names(self):
        self.assertRaises(ValueError, build_basic,
                          models=['Lasso', 'LassoLars', 'LassoCV', 'Lasso'])
        return


if __name__ == "__main__":
    unittest.main()
