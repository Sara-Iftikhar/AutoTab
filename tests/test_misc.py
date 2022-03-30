
import unittest

from utils import run_basic, build_basic


class TestMisc(unittest.TestCase):

    def test_r2_as_val_metric(self):
        """test different val_metrics for parent and child hpos"""
        run_basic(eval_metric="r2",
                  parent_iterations=10, child_iterations=25)

        return

    def test_model_names(self):
        """Should raise value error if some model is repeated"""
        self.assertRaises(ValueError, build_basic,
                          models=['Lasso', 'LassoLars', 'LassoCV', 'Lasso'])

    def test_zero_child_iter(self):
        res = run_basic(parent_iterations=14, child_iterations=0)

        return


if __name__ == "__main__":
    unittest.main()
