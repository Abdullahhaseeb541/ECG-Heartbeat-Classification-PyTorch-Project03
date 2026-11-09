"""Task 2 and Task 3: manually parameterised ECG neural network and loss."""
import math
import torch
from torch import nn

from .config import N_SAMPLES, N_CLASSES, HIDDEN_UNITS


class ECGNet(nn.Module):
    """Fully connected 187-128-64-5 network made without ready-made layers."""

    def __init__(self):
        super().__init__()
        layer_sizes = (N_SAMPLES, *HIDDEN_UNITS, N_CLASSES)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He normal: standard deviation sqrt(2 / fan_in), as specified.
            weight = nn.Parameter(
                torch.randn(fan_out, fan_in) * math.sqrt(2.0 / fan_in)
            )
            bias = nn.Parameter(torch.zeros(fan_out))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward_with_activations(self, x):
        """Return logits and hidden activations (the latter support Task 3.2)."""
        z1 = x @ self.weights[0].T + self.biases[0]
        a1 = torch.clamp(z1, min=0)

        z2 = a1 @ self.weights[1].T + self.biases[1]
        a2 = torch.clamp(z2, min=0)

        logits = a2 @ self.weights[2].T + self.biases[2]
        return logits, a1, a2

    def forward(self, x):
        """Return the five raw output logits for each row in x."""
        logits, _, _ = self.forward_with_activations(x)
        return logits

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def cross_entropy(logits, targets):
    """Stable mean softmax cross-entropy using torch.logsumexp."""
    log_normalizer = torch.logsumexp(logits, dim=1)
    correct_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    return (log_normalizer - correct_logits).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    network = ECGNet()
    example = torch.randn(8, N_SAMPLES)
    output = network(example)
    example_loss = cross_entropy(output, torch.zeros(8, dtype=torch.long))
    print("Output shape:", tuple(output.shape))
    print("Trainable parameters:", network.parameter_count())
    print("Example loss:", float(example_loss.detach()))
