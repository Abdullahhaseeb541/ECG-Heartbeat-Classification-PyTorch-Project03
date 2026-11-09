"""Small synthetic checks; these do not replace training on the Kaggle dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from src.model import ECGNet, cross_entropy
from src.inference import predict_batch, read_labeled_csv, read_signal_csv
from src.metrics import balanced_accuracy, confusion_matrix


def main():
    torch.manual_seed(0)
    model = ECGNet()
    assert model.parameter_count() == 32645, model.parameter_count()

    x = torch.randn(64, 187)
    y = torch.randint(0, 5, (64,), dtype=torch.long)
    logits, _, activation_2 = model.forward_with_activations(x)
    assert logits.shape == (64, 5)
    loss = cross_entropy(logits, y)
    assert torch.isfinite(loss)
    loss.backward()

    probabilities = torch.softmax(logits.detach(), dim=1)
    one_hot = torch.zeros_like(probabilities)
    one_hot.scatter_(1, y.view(-1, 1), 1.0)
    d_logits = (probabilities - one_hot) / len(y)
    manual_weight = d_logits.T @ activation_2.detach()
    manual_bias = d_logits.sum(dim=0)
    weight_error = (manual_weight - model.weights[2].grad).abs().max().item()
    bias_error = (manual_bias - model.biases[2].grad).abs().max().item()
    assert weight_error < 1e-5, weight_error
    assert bias_error < 1e-5, bias_error

    csv_signal = np.linspace(0, 1, 187, dtype=np.float32)
    csv_rows = np.vstack((
        np.append(csv_signal, 0),
        np.append(csv_signal, 4),
    ))
    labeled_bytes = "\n".join(
        ",".join(str(value) for value in row) for row in csv_rows
    ).encode("utf-8")
    parsed_signals, parsed_labels = read_labeled_csv(labeled_bytes)
    assert parsed_signals.shape == (2, 187)
    assert parsed_labels.tolist() == [0, 4]
    assert read_signal_csv(",".join(str(value) for value in csv_signal).encode()).shape == (187,)

    batch_predictions, batch_probabilities = predict_batch(model, parsed_signals, batch_size=1)
    assert batch_predictions.shape == (2,)
    assert batch_probabilities.shape == (2, 5)
    assert np.allclose(batch_probabilities.sum(axis=1), 1.0)

    partial_matrix = confusion_matrix(np.array([0, 0, 4]), np.array([0, 4, 4]))
    assert balanced_accuracy(partial_matrix) == 0.75

    print("Synthetic smoke test passed.")
    print("Parameters:", model.parameter_count())
    print("Loss:", float(loss.detach()))
    print("Gradient max differences:", weight_error, bias_error)
    print("Upload validation and batched inference checks passed.")
    print("Note: no real ECG accuracy is measured by this test.")


if __name__ == "__main__":
    main()
