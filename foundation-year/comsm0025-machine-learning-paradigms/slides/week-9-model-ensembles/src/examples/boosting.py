"""A simple implementation of a boosting algorithm."""
# pylint: disable=redefined-outer-name
from matplotlib.axes import Axes
import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from .binary_classifier import BasicLinearClassifier
from .binary_classifier_ensemble import (
    BinaryClassifierEnsemble,
)
from .util import Floats, Ints


def get_data(
    n_samples: int = 100,
) -> tuple[Floats, Ints]:
    """Get a set of data instances and labels."""
    data = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
    )
    return data[0], data[1]


def plot_data(
    ax: Axes,
    instances: Floats,
    labels: Ints,
    predictions: Ints,
):
    """Plot a set of data instances, labels, and predictions."""
    true_positives = numpy.logical_and(
        labels == 1, predictions == 1
    )
    true_negatives = numpy.logical_and(
        labels == 0, predictions == 0
    )
    false_positives = numpy.logical_and(
        labels == 0, predictions == 1
    )
    false_negatives = numpy.logical_and(
        labels == 1, predictions == 0
    )

    ax.scatter(
        instances[true_positives, 0],
        instances[true_positives, 1],
        label=f"True Positive (n = {len(instances[true_positives])})",
        s=5,
    )
    ax.scatter(
        instances[true_negatives, 0],
        instances[true_negatives, 1],
        label=f"True Negative (n = {len(instances[true_negatives])})",
        s=5,
    )
    ax.scatter(
        instances[false_positives, 0],
        instances[false_positives, 1],
        label=f"False Positive (n = {len(instances[false_positives])})",
        s=5,
    )
    ax.scatter(
        instances[false_negatives, 0],
        instances[false_negatives, 1],
        label=f"False Negative (n = {len(instances[false_negatives])})",
        s=5,
    )


def get_decision_boundary(
    min_x: float,
    max_x: float,
    weights: Floats,
    bias: float,
) -> tuple[Floats, Floats]:
    """Get the line defined by a set of weights and a bias."""
    line_x = numpy.linspace(min_x, max_x)
    line_y = (-weights[0] * line_x + bias) / weights[1]
    return line_x, line_y


def plot_decision_boundary(
    ax: Axes,
    min_x: float,
    max_x: float,
    index: int,
    model: BasicLinearClassifier,
    model_weight: float,
) -> None:
    """Plot the decision boundary of a model."""
    assert model.weights is not None
    assert model.bias is not None
    x, y = get_decision_boundary(
        min_x,
        max_x,
        model.weights,
        model.bias,
    )
    ax.plot(
        x,
        y,
        label=f"Model {index} (acc={model.accuracy:.3f}, alpha={model_weight:.3f})",
    )


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})

    fig, axs = plt.subplots(1, 1)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    fig.suptitle(
        "Boosted ensembles of basic linear classifiers"
    )

    axs = (
        [axs] if isinstance(axs, Axes) else axs.reshape(-1)
    )

    ax: Axes
    for ax in axs:
        instances, labels = get_data(2000)

        min_x = min(instances[:, 0])
        max_x = max(instances[:, 0])
        min_y = min(instances[:, 1])
        max_y = max(instances[:, 1])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        ensemble = BinaryClassifierEnsemble(
            BasicLinearClassifier,
            10,
        )

        ensemble.train(instances, labels)

        predictions = ensemble.predict(instances)

        plot_data(ax, instances, labels, predictions)

        ax.set_title(f"acc={ensemble.accuracy:.3f}")

        for model_index, model in enumerate(
            ensemble.models
        ):
            plot_decision_boundary(
                ax,
                min_x,
                max_x,
                model_index,
                model,
                ensemble.model_weights[model_index],
            )

        ax.legend(fontsize="small")

    plt.show()
