import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import numpy as np

def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_mnist()

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        print(np.max(parameters[0]))
        print(np.min(parameters[0]))
        print(parameters)
        parameters = [np.around(x,5) for x in parameters]
        print("After rounding")
        print(parameters)
        print(np.max(parameters[0]))
        print(np.min(parameters[0]))
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression(solver='lbfgs')
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server("127.0.0.1:4466", strategy=strategy, config={"num_rounds": 5})