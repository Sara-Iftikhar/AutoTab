
import os
import unittest
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

from ai4water.datasets import MtropicsLaos

from autotab import OptimizePipeline


cls_data = MtropicsLaos().make_classification(lookback_steps=1,)
inputs_cls = cls_data.columns.tolist()[0:-1]
outputs_cls = cls_data.columns.tolist()[-1:]


class TestBinaryCls(unittest.TestCase):
    """test binary classification"""

    def test_basic(self):
        pl = OptimizePipeline(
            inputs_to_transform=inputs_cls,
            input_features = inputs_cls,
            output_features=outputs_cls,
            mode="classification",
            eval_metric="accuracy",
            train_fraction=1.0,
            val_fraction=0.3,
            models=["ExtraTreeClassifier",
                    "RandomForestClassifier",
                    "XGBClassifier",
                    "CatBoostClassifier",
                    "LGBMClassifier",
                    "GradientBoostingClassifier",
                    "HistGradientBoostingClassifier",
                    "ExtraTreesClassifier",
                    "RidgeClassifier",
                    #"NuSVC",
                    "SVC",
                    "KNeighborsClassifier",
                    ],
            parent_iterations=12,
            child_iterations=0,
        )

        results = pl.fit(data=cls_data)
        # pl.post_fit() TODO
        pl.cleanup()

        return


if __name__ == "__main__":
    unittest.main()
