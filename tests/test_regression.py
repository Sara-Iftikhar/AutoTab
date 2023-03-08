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
from ai4water.preprocessing import DataSet

from utils import run_basic, build_basic, rgr_data

ds = DataSet(rgr_data)
train_x, train_y = ds.training_data()
val_x, val_y = ds.validation_data()
test_x, test_y = ds.test_data()

class TestRegression(unittest.TestCase):

    show = False

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
        pl.cleanup()
        return

    def test_basic_with_xy(self):

        pl = build_basic()

        pl.fit(x=train_x, y=train_y, validation_data=(val_x, val_y),
               process_results=False)

        pl.cleanup()
        return

    def test_change_child_iter(self):
        """check that we can change the hpo iterations for a model"""
        pl = build_basic(models = ['Lasso', 'RandomForestRegressor'])
        pl.change_child_iteration({"RandomForestRegressor": 10})
        pl.fit(data=rgr_data, process_results=False)
        assert pl.child_val_scores_.shape[1] == 10
        pl.cleanup()
        return

    def test_remove_model(self):
        """test that we can remove a model which is already being considered"""
        pl = build_basic()
        pl.remove_model("LinearRegression")
        assert "LinearRegression" not in pl.models
        assert "LinearRegression" not in pl._child_iters
        assert "LinearRegression" not in pl.model_space
        return

    def test_change_model_space(self):
        """test that we can change space of a model"""
        pl = build_basic(models = ['Lasso', 'RandomForestRegressor'])
        space = {'max_depth': [5,10, 15, 20],
                 'n_estimators': [5,10, 15, 20]}
        pl.update_model_space({"RandomForestRegressor": space})
        pl.fit(data=rgr_data, process_results=False)
        assert len(pl.model_space['RandomForestRegressor']['param_space'])==2
        return

    def test_baseline_results(self):
        pl = build_basic()
        pl.reset()
        val_scores, metrics = pl.baseline_results(data=rgr_data)
        assert isinstance(val_scores, dict)
        assert len(val_scores) == 4
        assert isinstance(metrics, dict)
        assert len(metrics) == 4
        for k,v in metrics.items():
            assert len(v) == 3, f"key {k} val: {v}"
        pl.cleanup()
        return

    def test_dumbbell_plot(self):
        pl = run_basic(parent_iterations=7,
            models=[
            "LinearRegression",
            "LassoLars",
            "Lasso",
            "RandomForestRegressor",
            "HistGradientBoostingRegressor",
        ],
                       child_iterations=0,
                       process_results=False
                       )
        ax = pl.dumbbell_plot(data=rgr_data, metric_name='r2', show=self.show)
        assert isinstance(ax, plt.Axes), ax
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
        ],
                       child_iterations=0,
                       process_results=False,
                       )
        ax = pl.compare_models('r2', "bar_chart", show=self.show)
        assert isinstance(ax, plt.Axes)
        ax = pl.compare_models('r2', show=self.show)
        assert isinstance(ax, plt.Axes)
        pl.cleanup()
        return

    def test_y_transformations(self):
        output_transformations = ['sqrt', 'log', 'log10']
        pl = run_basic(parent_algorithm="bayes",
                       parent_iterations=12,
                       outputs_to_transform='tetx_coppml',
                       output_transformations=output_transformations,
                       child_iterations = 0,
                       process_results=False
                            )

        y_transformation = pl.parent_suggestions_[1]['y_transformation'][0]['method']
        assert y_transformation in output_transformations
        pl.cleanup()
        return

    def test_tpe(self):
        pl = run_basic(parent_algorithm="tpe",
                       parent_iterations=12,
                       eval_metric="nse",
                       child_iterations=0,
                       process_results=False)
        pl.post_fit(data=rgr_data, show=self.show)
        pl.cleanup()

        return

    def test_single_model(self):
        pl = run_basic(models=["Lasso"],
                       process_results=False)
        pl.post_fit(data=rgr_data, show=self.show)
        pl.cleanup()
        return


if __name__ == "__main__":
    unittest.main()
