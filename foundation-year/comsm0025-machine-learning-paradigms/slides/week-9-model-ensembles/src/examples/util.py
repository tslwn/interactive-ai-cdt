"""Utilities."""
from typing import Any
import numpy

Floats = numpy.ndarray[Any, numpy.dtype[numpy.float_]]
Ints = numpy.ndarray[Any, numpy.dtype[numpy.int_]]


def get_accuracy(labels: Ints, predictions: Ints) -> float:
    """Get the accuracy of a set of predictions."""
    return sum(predictions == labels) / len(labels)
