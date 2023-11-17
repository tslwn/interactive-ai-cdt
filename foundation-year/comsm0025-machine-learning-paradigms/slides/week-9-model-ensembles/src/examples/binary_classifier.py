"""Binary classifiers."""
from abc import ABC, abstractmethod
import numpy
from .util import Floats, Ints, get_accuracy


class BinaryClassifier(ABC):
    """An abstract binary classifier."""

    @abstractmethod
    def train(
        self,
        instances: Floats,
        labels: Ints,
        weights: Floats,
    ) -> None:
        """Train the classifier."""

    @abstractmethod
    def predict(self, instances: Floats) -> Ints:
        """Predict the labels of a set of instances."""


class BasicLinearClassifier(BinaryClassifier):
    """A basic weighted linear classifier."""

    def __init__(self) -> None:
        self.weights: Floats | None = None
        self.bias: float | None = None
        self.accuracy: float | None = None

    def train(
        self,
        instances: Floats,
        labels: Ints,
        weights: Floats,
    ):
        p = numpy.average(
            instances[labels == 1],
            weights=weights[labels == 1],
            axis=0,
        )
        n = numpy.average(
            instances[labels == 0],
            weights=weights[labels == 0],
            axis=0,
        )
        self.weights = p - n
        self.bias = (numpy.dot(p, p) - numpy.dot(n, n)) / 2
        self.accuracy = get_accuracy(
            labels, self.predict(instances)
        )

    def predict(self, instances: Floats) -> Ints:
        assert self.weights is not None
        assert self.bias is not None
        return numpy.array(
            [
                1
                if numpy.dot(self.weights, instance)
                > self.bias
                else 0
                for instance in instances
            ]
        )
