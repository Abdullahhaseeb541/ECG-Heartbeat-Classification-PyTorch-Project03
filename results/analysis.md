# Task 6 – Analysis

All numerical values below are generated from the actual run of this project. Do not replace them with the expected range in the assignment manual.

## 6.1 Always predicting N

The always-N baseline is **82.76%** on the test set. The neural network test accuracy is **97.78%**. The baseline is already high because class N is much more common than the other classes. Therefore, overall accuracy alone can hide poor recognition of minority heartbeat types. Balanced accuracy and per-class accuracy are also necessary.

## 6.2 Lowest per-class accuracy and most common confusion

The lowest per-class accuracy belongs to **F (F - Fusion beat)**, at **66.05%**. Among the incorrect predictions for that true class, the most common predicted class is **N (N - Normal beat)**, with **36** examples. Two beat types may be difficult to separate because their waveform shapes can overlap in morphology, while the rare class provides fewer examples from which the network can learn a stable pattern. This is a plausible explanation, not a clinical conclusion.

## 6.3 Overfitting or underfitting

At the selected best epoch (**25**), training loss is **0.0516** and validation loss is **0.0676** (validation minus training = **0.0160**). At the selected epoch, the training/validation loss gap is 0.0160. The saved curves should be inspected for a widening gap; the best validation epoch is used to reduce the risk of selecting a later over-fitted checkpoint. Refer to `chosen_loss_curves.png` and `chosen_accuracy_curves.png` when explaining the final judgement.

## 6.4 Why zero initialisation was not used and why He initialisation is suitable

If every weight started at zero, neurons in the same layer would receive identical gradients and remain identical. They would fail to learn different features. He initialisation uses a variance of approximately `2 / fan_in` (standard deviation `sqrt(2 / fan_in)`) to help maintain a useful signal scale through layers that use ReLU, reducing the risk that activations or gradients become extremely small or large at the start of training.

## 6.5 Momentum and weight decay

Momentum keeps a running velocity. When consecutive mini-batch gradients point in opposite directions, the new gradient partially cancels the previous velocity, smoothing the update and reducing rapid oscillation. Weight decay adds `lambda * weight` to a weight's gradient. Therefore, even when its data gradient is zero, a non-zero weight is gradually pulled toward zero. The assignment applies this decay to weights, not biases.

## 6.6 Helping the rare F class

One possible change is class-weighted cross-entropy: assign a larger loss weight to errors on rare classes such as F. This makes mistakes on under-represented classes contribute more strongly to the objective. Another option is to oversample minority-class training examples. Any such change must be evaluated using validation data and must not use the test set for tuning.

## Test metrics

- Test rows: 21,892
- Test accuracy: 97.78%
- Balanced accuracy: 85.23%
- Always-N accuracy: 82.76%

| Class | Test examples | Per-class accuracy |
|---|---:|---:|
| N | 18,118 | 99.33% |
| S | 556 | 70.32% |
| V | 1,448 | 93.16% |
| F | 162 | 66.05% |
| Q | 1,608 | 97.26% |

The full confusion matrix is saved in `confusion_matrix.csv` and `test_confusion_matrix.png`.
