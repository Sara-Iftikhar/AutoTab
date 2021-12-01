import unittest

from automl import OptimizePipeline

from ai4water.datasets import arg_beach


inputs = ['tide_cm', 'wat_temp_c', 'sal_psu',
          'pcp3_mm',  # 'pcp6_mm', 'pcp12_mm',
          'pcp_mm', 'air_temp_c', 'rel_hum']

data = arg_beach(inputs=inputs)


class TestRegression(unittest.TestCase):

    def test_basic(self):
        pl = OptimizePipeline(
            data=data,
            features=inputs,
            parent_iterations=4,
            child_iterations=4,
            parent_algorithm="random",
            child_algorithm="random",
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

        return


if __name__ == "__main__":
    unittest.main()
