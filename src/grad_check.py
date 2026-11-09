"""Task 3.2: compare the output-layer gradients with hand calculations."""
import json
import numpy as np
import torch

from .config import PREPARED_FILE, RESULTS_DIR, SEED
from .model import ECGNet, cross_entropy
from .prepare import prepare_data


def run_gradient_check():
    if not PREPARED_FILE.exists():
        prepare_data()

    data = np.load(PREPARED_FILE)
    x = torch.as_tensor(data["X_train"][:64], dtype=torch.float32)
    y = torch.as_tensor(data["y_train"][:64], dtype=torch.long)

    torch.manual_seed(SEED)
    model = ECGNet()
    logits, _, a2 = model.forward_with_activations(x)
    loss = cross_entropy(logits, y)
    loss.backward()

    # Manual output-layer gradient: (softmax(logits) - one_hot(y))^T A2 / B.
    batch_size = x.shape[0]
    probabilities = torch.softmax(logits.detach(), dim=1)
    one_hot = torch.zeros_like(probabilities)
    one_hot.scatter_(1, y.view(-1, 1), 1.0)
    d_logits = (probabilities - one_hot) / batch_size
    manual_weight_gradient = d_logits.T @ a2.detach()
    manual_bias_gradient = d_logits.sum(dim=0)

    weight_difference = torch.max(
        torch.abs(manual_weight_gradient - model.weights[2].grad)
    ).item()
    bias_difference = torch.max(
        torch.abs(manual_bias_gradient - model.biases[2].grad)
    ).item()

    result = {
        "batch_size": int(batch_size),
        "initial_loss_on_first_64_training_rows": float(loss.item()),
        "max_absolute_difference_W3": float(weight_difference),
        "max_absolute_difference_b3": float(bias_difference),
        "passes_1e-5_tolerance": bool(max(weight_difference, bias_difference) < 1e-5),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "gradient_check.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("Task 3 - Output-layer gradient check")
    print(f"Initial loss (first 64 rows): {loss.item():.8f}")
    print(f"Largest |manual - autograd| for W[3]: {weight_difference:.10e}")
    print(f"Largest |manual - autograd| for b[3]: {bias_difference:.10e}")
    print("Passes 1e-5 tolerance:", result["passes_1e-5_tolerance"])
    return result


if __name__ == "__main__":
    run_gradient_check()
