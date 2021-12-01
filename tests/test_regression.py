import unittest

from automl import OptimizePipeline

from ai4water.datasets import arg_beach


inputs = ['tide_cm', 'wat_temp_c', 'sal_psu',
          'pcp3_mm',  # 'pcp6_mm', 'pcp12_mm',
          'pcp_mm', 'air_temp_c', 'rel_hum']

data = arg_beach(inputs=inputs)


def run_basic(parent_algorithm="random", child_algorithm="random",
              parent_iterations=4, child_iterations=4,
              parent_val_metric="mse", child_val_metric="mse",
              ):

    pl = OptimizePipeline(
        data=data,
        features=inputs,
        parent_iterations=parent_iterations,
        child_iterations=child_iterations,
        parent_algorithm=parent_algorithm,
        child_algorithm=child_algorithm,
        parent_val_metric=parent_val_metric,
        child_val_metric=child_val_metric,
        monitor=['r2', 'nse'],
        models=[
            "RandomForestRegressor",
            "XGBRegressor",
            "LGBMRegressor",
            "CatBoostRegressor"
        ],
        train_data="random",
        val_data="same",
        val_fraction=0.0,
    )

    results = pl.fit(
    )
    return results


class TestRegression(unittest.TestCase):

    def test_basic(self):
        run_basic()

        return

    def test_r2_as_val_metric(self):

        run_basic(parent_val_metric="r2", child_val_metric="r2")

        return


if __name__ == "__main__":
    unittest.main()
