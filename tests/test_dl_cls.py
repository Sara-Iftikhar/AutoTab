import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from utils import run_basic


def classification_data(n_classes):
    input_features = [f'input_{n}' for n in range(10)]
    outputs = ['target']
    X, y = make_classification(n_samples=100,
                               n_features=len(input_features),
                               n_informative=n_classes,
                               n_classes=n_classes,
                               random_state=1,
                               n_redundant=0,
                               n_repeated=0
                               )
    y = y.reshape(-1, 1)

    df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=input_features + outputs)

    return df

bin_data = classification_data(2)
bin_input_features = bin_data.columns.tolist()[0:-1]
bin_output_features = bin_data.columns.tolist()[-1:]

multi_cls_data = classification_data(4)
multi_cls_input_features = multi_cls_data.columns.tolist()[0:-1]
multi_cls_output_features = multi_cls_data.columns.tolist()[-1:]


class TestClassification(unittest.TestCase):

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
        )

        pl.post_fit(data=bin_data, show=False)

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