import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

import os
import unittest

from autotab import OptimizePipeline
from autotab.utils import Callbacks
from ai4water import Model
from ai4water.preprocessing import DataSet

from utils import run_basic, build_basic, rgr_data, make_kws

ds = DataSet(rgr_data, verbosity=0)
train_x, train_y = ds.training_data()
val_x, val_y = ds.validation_data()
test_x, test_y = ds.test_data()


class TestMisc(unittest.TestCase):

    show = False

    def test_r2_as_val_metric(self):
        """test a specifc val metric"""
        run_basic(eval_metric="r2",
                  parent_iterations=4,
                  child_iterations=0,
                  process_results=False,
                  parent_algorithm="random"
                  )

        return

    def test_cv(self):
        run_basic(eval_metric="r2",
                  cv_parent_hpo=True,
                  parent_iterations=3,
                  parent_algorithm="random",
                  child_iterations=0,
                  cross_validator={"KFold": {"n_splits": 4}},
                  input_transformations=['log', 'log10', 'sqrt', 'robust'],
                  process_results=False
                  )
        return

    def test_from_config(self):
        pl = run_basic(
            eval_metric="r2",
            child_iterations=0,
            process_results=False,
            parent_algorithm="random",
            parent_iterations=3,
        )

        pl2 = OptimizePipeline.from_config_file(os.path.join(pl.path, "config.json"))
        pl2.post_fit(data=rgr_data, show=self.show)
        pl.dumbbell_plot(data=rgr_data, save=False, upper_limit=1e15, show=self.show)
        pl.taylor_plot(data=rgr_data, save=False, show=self.show)
        pl.compare_models(show=self.show)
        pl.compare_models(plot_type="bar_chart", show=self.show)
        pl._pp_plots = []
        model = pl.bfe_best_model_from_scratch(metric_name='r2', data=rgr_data)
        assert isinstance(model, Model)
        return

    def test_from_config_xy(self):
        pl = build_basic(
            eval_metric="r2",
            child_iterations=0,
            parent_iterations=3,
            parent_algorithm="random",
        )
        pl.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), process_results=False)

        pl2 = OptimizePipeline.from_config_file(os.path.join(pl.path, "config.json"))
        pl2.post_fit(x=train_x, y=train_y, test_data=(test_x, test_y), show=self.show)
        return

    def test_model_names(self):
        """Should raise value error if some model is repeated"""
        self.assertRaises(ValueError, build_basic,
                          models=['Lasso', 'LassoLars', 'LassoCV', 'Lasso'])
        return

    def test_zero_child_iter(self):
        pl = build_basic(parent_iterations=4,
                       child_iterations=0,
                       parent_algorithm="random",
                       process_results=False)
        for v in pl._child_iters.values():
            assert v == 0

        return

    def test_zero_child_iter_xy(self):
        pl = build_basic(parent_iterations=4,
                       child_iterations=0,
                       parent_algorithm="random",
                       process_results=False
                       )
        for v in pl._child_iters.values():
            assert v == 0
        return

    def test_grouped_transformations(self):
        pl = run_basic(
            inputs_to_transform={
                'group1': ['tide_cm', 'wat_temp_c', 'sal_psu', 'pcp3_mm'],
                'group2': ['pcp_mm', 'air_temp_c', 'rel_hum']
            },
            child_iterations=0,
            process_results=False,
            parent_algorithm="random",
            parent_iterations=4,
        )
        pl.post_fit(data=rgr_data, show=self.show)
        pl.cleanup()
        return

    def test_no_model(self):

        pl = OptimizePipeline(input_features=['a'], output_features="", models=[])
        assert len(pl.models) == 0
        return

    def test_no_train_on_all_data(self):
        """test a specifc val metric"""
        pl = build_basic(
            eval_metric="r2",
            child_iterations=0,
            parent_algorithm="random",
            parent_iterations=4,
        )
        pl.fit(data=rgr_data, process_results=False)
        pl.post_fit(data=rgr_data, fit_on_all_train_data=False, show=self.show)

        return

    def test_remove_transformation(self):
        pl = build_basic()
        pl.remove_transformation('box-cox')
        for sp in pl.space():
            assert 'box-cox' not in sp.categories

        pl.remove_transformation(['yeo-johnson', 'log'])
        for sp in pl.space():
            assert 'yeo-johnson' not in sp.categories
            assert 'log' not in sp.categories

        pl.remove_transformation('log2', 'tide_cm')
        for sp in pl.space():
            if sp.name not in ["model"]:
                if sp.name == "tide_cm":
                    assert 'log2' not in sp.categories
                else:
                    assert 'log2' in sp.categories

        pl.remove_transformation('log10', ['tide_cm', 'wat_temp_c'])

        for sp in pl.space():
            if sp.name not in ['model']:
                if sp.name in ['tide_cm', 'wat_temp_c']:
                    assert 'log10' not in sp.categories
                else:
                    assert 'log10' in sp.categories
        return

    def test_no_transformation_on_inputs(self):
        """do not apply any transformation on input features"""
        pl = build_basic(inputs_to_transform=[])
        space = pl.space()
        assert len(space) == 1
        return

    def test_contex_manager_no_model(self):

        class MyCallbacks(Callbacks):
            def on_eval_begin(self, model, iter_num=None, x=None, y=None, validation_data=None) ->None:
                print("raising Value Error")
                raise ValueError
        kws = make_kws()

        def call():
            with OptimizePipeline(**kws) as pl:
                pl.fit(data = rgr_data, callbacks=MyCallbacks())

        self.assertRaises(ValueError, call)

        return

    def test_save_results_without_fit(self):

        pl = build_basic()

        pl.save_results()
        pl.report()
        pl._save_config()
        return

    def test_input_transformations(self):
        """makes sure that the transformations which are not part of 'input_transformations'
        argument do not appear in space.
        """
        pl = build_basic(input_transformations=[
            "minmax", "center", "scale", "zscore",  "box-cox", "robust", "log", "log2", "log10",
            "sqrt", "pareto", "none"
        ])
        for sp in pl.space():
            assert all([trans not in sp.categories for trans in [
                'yeo-johnson', 'quntile', 'qunatile_normal', 'vast']])
        return

    def test_input_transformations1(self):
        """makes sure that the transformations which are not part of 'input_transformations'
        argument do not appear in space. Test it by using pl.remove_transformation() method
        """
        pl = build_basic(input_transformations=[
            "minmax", "center", "scale", "zscore",  "box-cox",  "robust", "log", "log2", "log10",
            "sqrt", "pareto", "none"
        ])
        pl.remove_transformation('box-cox', 'tide_cm')
        for sp in pl.space():
            for trans in ['yeo-johnson', 'quntile', 'qunatile_normal', 'vast']:
                assert trans not in sp.categories, f"{trans} in categories {sp.categories}"
        return

    def test_input_transformations2(self):
        """makes sure that the transformations which are not part of 'input_transformations'
        argument do not appear in space. but they should appear in output_transformations
        if not removed from there
        """
        pl = build_basic(input_transformations=[
            "minmax", "center", "scale", "zscore",  "box-cox",  "robust", "log", "log2", "log10"],
            outputs_to_transform = ['tetx_coppml']
        )
        pl.remove_transformation('box-cox', 'tide_cm')
        for sp in pl.space():
            if sp.name != "tetx_coppml":
                for trans in ['yeo-johnson', 'quntile', 'qunatile_normal', 'vast']:
                    assert trans not in sp.categories, f"for {sp.name} {trans} in categories {sp.categories}"
        return

    def test_input_transformations3(self):
        """makes sure that the transformations which are not part of 'input_transformations'
        argument do not appear in space. but they should appear in output_transformations
        if not removed from there
        """
        pl = build_basic(input_transformations=[
            "minmax", "center", "scale", "zscore",  "box-cox",  "robust", "log", "log2", "log10"],
            output_transformations = ["minmax", "center", "scale", "zscore"]
        )
        pl.remove_transformation('box-cox', 'tide_cm')
        for sp in pl.space():

            if sp.name == "tetx_coppml":
                for trans in ['yeo-johnson', 'quntile', 'qunatile_normal', 'vast', 'log']:
                    assert trans not in sp.categories, f"{trans} in {sp.categories}"

            else:
                for trans in ['yeo-johnson', 'quntile', 'qunatile_normal', 'vast']:
                    assert trans not in sp.categories, f"for {sp.name} {trans} in categories {sp.categories}"

            if sp.name not in  ["tide_cm", 'model']:
                # box-cox is removed for tide_cm, it should be in output space
                assert 'box-cox' in sp.categories, f"{sp.name}"
        return


if __name__ == "__main__":
    unittest.main()
