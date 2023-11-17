"""Boosted binary-classifier ensemble."""
from math import log
from typing import Generic, TypeVar
import numpy
from .binary_classifier import BinaryClassifier
from .util import Floats, Ints, get_accuracy

GenericBinaryClassifier = TypeVar(
    "GenericBinaryClassifier", bound=BinaryClassifier
)


class BinaryClassifierEnsemble(
    Generic[GenericBinaryClassifier]
):
    """An ensemble of binary classifiers."""

    def __init__(
        self,
        model: type[GenericBinaryClassifier],
        max_models: int,
    ):
        self.model = model
        self.max_models = max_models

        self.models: list[GenericBinaryClassifier] = []
        self.model_weights: list[float] = []
        self.weights: Floats | None = None
        self.accuracy: float | None = None

    def predict(self, instances: Floats) -> Ints:
        """Predict the labels of a set of instances by averaging."""
        assert self.weights is not None

        predictions: Ints = numpy.array(
            [
                model.predict(instances)
                for model in self.models
            ]
        )
        predictions = numpy.average(
            predictions,
            axis=0,
            weights=self.model_weights,
        )
        return numpy.array(
            [
                1 if prediction > 0.5 else 0
                for prediction in predictions
            ]
        )

    def train(
        self, instances: Floats, labels: Ints
    ) -> None:
        """Train the ensemble."""
        self.weights = numpy.array(
            [
                1 / len(instances)
                for _ in range(len(instances))
            ]
        )

        for _ in range(self.max_models):
            model = self.model()
            model.train(instances, labels, self.weights)

            predictions = model.predict(instances)

            weighted_error = get_weighted_error(
                labels, predictions, self.weights
            )
            if weighted_error == 0 or weighted_error > 0.5:
                break

            self.models.append(model)

            model_weight = get_model_weight(weighted_error)

            self.model_weights.append(model_weight)

            self.weights = update_weights(
                self.weights,
                labels,
                predictions,
                weighted_error,
            )

        self.accuracy = get_accuracy(
            labels, self.predict(instances)
        )


def get_weighted_error(
    labels: Ints,
    predictions: Ints,
    weights: Floats,
) -> float:
    """Get the weighted error of a set of predictions."""
    return sum(
        weight
        for label, prediction, weight in zip(
            labels, predictions, weights
        )
        if label != prediction
    )


def get_model_weight(weighted_error: float) -> float:
    """Get the weight of a model."""
    return log((1 - weighted_error) / weighted_error) / 2


def update_weight(
    weight: float,
    label: int,
    prediction: int,
    weighted_error: float,
) -> float:
    """Update the weight of an instance."""
    if label == prediction:
        return weight / (2 * (1 - weighted_error))
    return weight / (2 * weighted_error)


def update_weights(
    weights: Floats,
    labels: Ints,
    predictions: Ints,
    weighted_error: float,
) -> Floats:
    """Update the weights of a set of instances."""
    return numpy.array(
        [
            update_weight(
                weight, label, prediction, weighted_error
            )
            for weight, label, prediction in zip(
                weights, labels, predictions
            )
        ]
    )
