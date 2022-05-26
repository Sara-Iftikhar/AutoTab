import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

import os
import unittest

from autotab import OptimizePipeline
from ai4water.preprocessing import DataSet

from utils import run_basic, build_basic, rgr_data

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
        pl = run_basic(parent_iterations=4,
                       child_iterations=0,
                       parent_algorithm="random",
                       process_results=False)
        assert pl.child_iter_ == pl.child_iterations
        assert pl.child_val_scores_.size == 0

        return

    def test_zero_child_iter_xy(self):
        pl = run_basic(parent_iterations=4,
                       child_iterations=0,
                       parent_algorithm="random",
                       process_results=False
                       )
        assert pl.child_iter_ == pl.child_iterations
        assert pl.child_val_scores_.size == 0
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


if __name__ == "__main__":
    unittest.main()
