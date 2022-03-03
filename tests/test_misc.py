
import unittest

from utils import run_basic, build_basic


class TestMetrics(unittest.TestCase):
    """test different val_metrics for parent and child hpos"""
    def test_r2_as_val_metric(self):

        run_basic(eval_metric="r2",
                  parent_iterations=10, child_iterations=25)

        return


class TestRepeatition(unittest.TestCase):
    """Should raise value error if some model is repeated"""
    def test_model_names(self):
        self.assertRaises(ValueError, build_basic,
                          models=['Lasso', 'LassoLars', 'LassoCV', 'Lasso'])


if __name__ == "__main__":
    unittest.main()
