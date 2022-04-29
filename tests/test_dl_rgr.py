
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

import unittest

from utils import run_basic, rgr_data


class TestMisc(unittest.TestCase):

    def test_1model_both_xy_trans(self):
        """test a specifc val metric"""
        pl = run_basic(models = [
            "MLP",
        ],
            parent_iterations=10,
            outputs_to_transform=["tetx_coppml"],
            child_iterations=0,
            epochs=20,
            category="DL",
        )
        pl.post_fit(data=rgr_data, show=False)

        return

    def test_1model_no_child_iters(self):
        """test a specifc val metric"""
        run_basic(models = [
            "MLP",
        ],
            parent_iterations=10,
            child_iterations=0,
            epochs=20,
            category="DL",
        )

        return

    def test_1model_with_child_iters(self):
        """test a specifc val metric"""
        run_basic(models = [
            "MLP",
        ],
            parent_iterations=10,
            child_iterations=5,
            child_algorithm="random",
            epochs=20,
            category="DL",
        )

        return

    def test_2model_no_child_iters(self):
        """test a specifc val metric"""
        run_basic(models = [
            "MLP",
            "CNN",
        ],
            parent_iterations=10,
            child_iterations=0,
            epochs=20,
            ts_args={"lookback":5},
            category="DL",
        )

        return


if __name__ == "__main__":
    unittest.main()