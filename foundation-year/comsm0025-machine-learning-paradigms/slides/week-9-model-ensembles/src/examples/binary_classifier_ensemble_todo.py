"""Boosted binary-classifier ensemble (TODO)."""
from typing import Generic, TypeVar
from .binary_classifier import BinaryClassifier
from .util import Floats, Ints

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
        raise NotImplementedError

    def predict(self, instances: Floats) -> Ints:
        """Predict the labels of a set of instances by averaging."""
        raise NotImplementedError

    def train(
        self, instances: Floats, labels: Ints
    ) -> None:
        """Train the ensemble."""
        raise NotImplementedError


def get_weighted_error(
    labels: Ints,
    predictions: Ints,
    weights: Floats,
) -> float:
    """Get the weighted error of a set of predictions."""
    raise NotImplementedError


def get_model_weight(weighted_error: float) -> float:
    """Get the weight of a model."""
    raise NotImplementedError


def update_weight(
    weight: float,
    label: int,
    prediction: int,
    weighted_error: float,
) -> float:
    """Update the weight of an instance."""
    raise NotImplementedError


def update_weights(
    weights: Floats,
    labels: Ints,
    predictions: Ints,
    weighted_error: float,
) -> Floats:
    """Update the weights of a set of instances."""
    raise NotImplementedError
