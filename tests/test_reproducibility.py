
import os
import unittest
import site


package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
site.addsitedir(package_path)

from ai4water.datasets import busan_beach

from autotab import OptimizePipeline


inputs = ['tide_cm', 'wat_temp_c', 'sal_psu',
          'pcp3_mm',  # 'pcp6_mm', 'pcp12_mm',
          'pcp_mm', 'air_temp_c', 'rel_hum']

data = busan_beach(inputs=inputs)


class TestMLRegression(unittest.TestCase):

    def test_rgr(self):

        pl = OptimizePipeline(input_transformations=['log', 'log2', 'sqrt', 'none', 'log10',
                'minmax', 'scale', 'center', 'zscore', 'robust'],
                input_features=inputs,
                inputs_to_transform=inputs,
                output_features=data.columns.tolist()[-1:],
                models=['LinearRegression', 
                    'XGBRegressor', 
                    'CatBoostRegressor', 
                    'RandomForestRegressor',
                    "LGBMRegressor",],
                split_random=True,
                output_transformations=['log', 'log10', 'log2', 'sqrt', 'none'],
                parent_iterations=12,
                child_iterations=5,
                seed=891,
                eval_metric="r2_score",
                child_algorithm="random",
                )
        
        pl.fit(data=data)

        best_val = pl.optimizer_.best_xy()['y']

        pipeline = pl.get_best_pipeline_by_metric(metric_name='r2_score')

        model = pl.build_model(model=pipeline['model'],
                                x_transformation=pipeline['x_transformation'],
                                y_transformation=pipeline['y_transformation'],
                                val_metric='r2_score',
                                prefix=None,
                                )

        model.fit(data=data)

        val_score = pl.evaluate_model(model,
                                      metric_name='r2_score',
                                      data='validation')

        self.assertAlmostEqual(val_score + best_val, 1.0, places=5)

        pl.cleanup()
        return


if __name__ == '__main__':
    unittest.main()