import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

import unittest

from utils import run_basic, classification_data

bin_data = classification_data(2)
bin_input_features = bin_data.columns.tolist()[0:-1]
bin_output_features = bin_data.columns.tolist()[-1:]

multi_cls_data = classification_data(4)
multi_cls_input_features = multi_cls_data.columns.tolist()[0:-1]
multi_cls_output_features = multi_cls_data.columns.tolist()[-1:]


class TestClassification(unittest.TestCase):

    show = False
    def test_binary(self):
        pl = run_basic(models=[
            "MLP",
        ],
            input_features=bin_input_features,
            output_features=bin_output_features,
            parent_iterations=10,
            child_iterations=0,
            parent_algorithm="bayes",
            loss="binary_crossentropy",
            epochs=10,
            category="DL",
            mode="classification",
            num_classes=2,
            monitor="f1_score",
            eval_metric="accuracy",
            data=bin_data,
            process_results=False
        )

        pl.post_fit(data=bin_data, show=self.show)

        return

    def test_multi_class(self):
        run_basic(models=[
            "MLP",
        ],
            input_features=multi_cls_input_features,
            output_features=multi_cls_output_features,
            parent_algorithm="bayes",
            loss="categorical_crossentropy",
            parent_iterations=10,
            child_iterations=0,
            epochs=20,
            category="DL",
            mode="classification",
            num_classes = 4,
            eval_metric="accuracy",
            monitor="f1_score",
            data=multi_cls_data,
        )

        return

if __name__ == "__main__":
    unittest.main()