import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

import os
import unittest

from autotab import OptimizePipeline
from autotab.utils import Callbacks
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
                  parent_iterations=10,
                  child_iterations=0,
                  process_results=False
                  )

        return

    def test_cv(self):
        run_basic(eval_metric="r2",
                  cv_parent_hpo=True,
                  parent_iterations=3,
                  parent_algorithm="random",
                  child_iterations=0,
                  cross_validator={"KFold": {"n_splits": 5}},
                  input_transformations=['log', 'log10', 'sqrt', 'robust'],
                  process_results=False
                  )
        return

    def test_from_config(self):
        pl = run_basic(
            eval_metric="r2",
            child_iterations=0,
            process_results=False,
        )

        pl2 = OptimizePipeline.from_config_file(os.path.join(pl.path, "config.json"))
        pl2.post_fit(data=rgr_data, show=self.show)
        return

    def test_from_config_xy(self):
        pl = build_basic(
            eval_metric="r2",
            child_iterations=0,
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
            process_results=False
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
        )
        pl.fit(data=rgr_data, process_results=False)
        pl.post_fit(data=rgr_data, fit_on_all_train_data=False, show=self.show)

        return

    def test_remove_transformation(self):
        pl = build_basic()
        pl.remove_transformation('box-cox')
        pl.remove_transformation(['yeo-johnson', 'log'])
        pl.remove_transformation('log2', 'tide_cm')
        pl.remove_transformation('log10', ['tide_cm', 'wat_temp_c'])
        return

    def test_no_transformation_on_inputs(self):
        """do not apply any transformation on input features"""
        pl = build_basic(inputs_to_transform=[])
        space = pl.space()
        assert len(space) == 1
        return

    def test_contex_manager_no_model(self):

        class MyCallbacks(Callbacks):
            def on_eval_begin(self, iter_num=None, x=None, y=None, validation_data=None) ->None:
                print("raising Value Error")
                raise ValueError
        kws = make_kws()

        def call():
            with OptimizePipeline(**kws) as pl:
                pl.fit(data = rgr_data, callbacks=MyCallbacks())

        self.assertRaises(ValueError, call)

        return


if __name__ == "__main__":
    unittest.main()
