class Model:
    def train(
        self,
        instances: list[list[float]],
        labels: list[int],
    ):
        raise NotImplementedError

    def predict(
        self, instances: list[list[float]]
    ) -> list[int]:
        raise NotImplementedError


class Ensemble:
    def __init__(
        self,
        models: list[Model],
        model_weights: list[float],
    ):
        raise NotImplementedError

    def predict(
        self, instances: list[list[float]]
    ) -> list[int]:
        raise NotImplementedError


def boost(
    instances: list[list[float]],
    labels: list[int],
    n_models: int,
) -> Ensemble:
    models = []

    model_weights = []

    weights = [
        1 / len(instances) for _ in range(len(instances))
    ]

    for _ in range(n_models):
        model = Model()

        model.train(instances, labels)
        models.append(model)

        predictions = model.predict(instances)

        weighted_error = get_weighted_error(
            labels, predictions, weights
        )

        model_weight = get_model_weight(weighted_error)
        model_weights.append(model_weight)

        weights = update_weights(
            weights, labels, predictions, weighted_error
        )

    return Ensemble(models, model_weights)


def get_weighted_error(
    labels: list[int],
    predictions: list[int],
    weights: list[float],
) -> float:
    raise NotImplementedError


def get_model_weight(
    weighted_error: float,
) -> float:
    raise NotImplementedError


def update_weight(
    weight: float,
    label: int,
    prediction: int,
    weighted_error: float,
) -> float:
    raise NotImplementedError


def update_weights(
    weights: list[float],
    labels: list[int],
    predictions: list[int],
    weighted_error: float,
) -> list[float]:
    return [
        update_weight(w_i, y_i, y_hat_i, weighted_error)
        for w_i, y_i, y_hat_i in zip(
            weights, labels, predictions
        )
    ]
