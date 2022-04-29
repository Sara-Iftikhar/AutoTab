
import unittest

from utils import run_basic, build_basic, rgr_data


class TestMisc(unittest.TestCase):

    def test_r2_as_val_metric(self):
        """test a specifc val metric"""
        run_basic(eval_metric="r2",
                  parent_iterations=10, child_iterations=0)

        return

    def test_cv(self):
        run_basic(eval_metric="r2",
                  cv_parent_hpo=True,
                  parent_iterations=10,
                  child_iterations=0,
                  cross_validator={"KFold": {"n_splits": 5}}
                  )
        return

    def test_model_names(self):
        """Should raise value error if some model is repeated"""
        self.assertRaises(ValueError, build_basic,
                          models=['Lasso', 'LassoLars', 'LassoCV', 'Lasso'])

    def test_zero_child_iter(self):
        pl = run_basic(parent_iterations=14, child_iterations=0)
        pl.post_fit(data=rgr_data, show=False)
        pl.cleanup()
        return

    def test_grouped_transformations(self):
        pl = run_basic(
            inputs_to_transform={
                'group1': ['tide_cm', 'wat_temp_c', 'sal_psu', 'pcp3_mm'],
                'group2': ['pcp_mm', 'air_temp_c', 'rel_hum']
            },
            child_iterations=0
        )
        pl.post_fit(data=rgr_data, show=False)
        pl.cleanup()


if __name__ == "__main__":
    unittest.main()
