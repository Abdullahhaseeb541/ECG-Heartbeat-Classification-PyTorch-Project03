"""Generate result-dependent analysis and communication drafts after testing."""
from pathlib import Path
import json
import numpy as np

from .config import RESULTS_DIR, CLASS_CODES, CLASS_NAMES


def _pct(value):
    return f"{100.0 * float(value):.2f}%"


def generate_reports(training, test, matrix):
    """Create Task 6 analysis, a Medium article draft, and a LinkedIn draft."""
    matrix = np.asarray(matrix, dtype=np.int64)
    class_acc = np.asarray(test["per_class_accuracy"], dtype=float)
    counts = matrix.sum(axis=1)
    lowest_class = int(np.argmin(class_acc))
    off_diagonal = matrix[lowest_class].copy()
    off_diagonal[lowest_class] = 0
    confused_class = int(np.argmax(off_diagonal))
    confusion_count = int(off_diagonal[confused_class])

    always_n = float(test["always_N_accuracy"])
    model_accuracy = float(test["test_accuracy"])
    gap = float(training["validation_loss_at_best_epoch"] - training["train_loss_at_best_epoch"])
    best_epoch = int(training["best_epoch"])
    final_val_loss = float(training["learning_rate_runs"][
        list(training["candidate_learning_rates"]).index(training["chosen_learning_rate"])
    ]["history"]["validation_loss"][-1])
    best_val_loss = float(training["validation_loss_at_best_epoch"])
    best_train_loss = float(training["train_loss_at_best_epoch"])
    last_train_loss = float(training["learning_rate_runs"][
        list(training["candidate_learning_rates"]).index(training["chosen_learning_rate"])
    ]["history"]["train_loss"][-1])

    if best_epoch < int(training["epochs"]) and final_val_loss > best_val_loss + 1e-3 and last_train_loss < best_train_loss:
        overfit_comment = (
            f"There is evidence of overfitting after epoch {best_epoch}: training loss continues "
            f"to decrease to {last_train_loss:.4f}, while validation loss rises from its best "
            f"value of {best_val_loss:.4f} to {final_val_loss:.4f} by the final epoch."
        )
    elif gap > 0.15:
        overfit_comment = (
            f"At the selected epoch, validation loss ({training['validation_loss_at_best_epoch']:.4f}) "
            f"is noticeably above training loss ({training['train_loss_at_best_epoch']:.4f}), "
            "which suggests a generalisation gap and possible overfitting. The full curves should "
            "also be inspected before making a strong conclusion."
        )
    else:
        overfit_comment = (
            f"At the selected epoch, the training/validation loss gap is {gap:.4f}. The saved "
            "curves should be inspected for a widening gap; the best validation epoch is used "
            "to reduce the risk of selecting a later over-fitted checkpoint."
        )

    analysis = f"""# Task 6 – Analysis

All numerical values below are generated from the actual run of this project. Do not replace them with the expected range in the assignment manual.

## 6.1 Always predicting N

The always-N baseline is **{_pct(always_n)}** on the test set. The neural network test accuracy is **{_pct(model_accuracy)}**. The baseline is already high because class N is much more common than the other classes. Therefore, overall accuracy alone can hide poor recognition of minority heartbeat types. Balanced accuracy and per-class accuracy are also necessary.

## 6.2 Lowest per-class accuracy and most common confusion

The lowest per-class accuracy belongs to **{CLASS_CODES[lowest_class]} ({CLASS_NAMES[lowest_class]})**, at **{_pct(class_acc[lowest_class])}**. Among the incorrect predictions for that true class, the most common predicted class is **{CLASS_CODES[confused_class]} ({CLASS_NAMES[confused_class]})**, with **{confusion_count:,}** examples. Two beat types may be difficult to separate because their waveform shapes can overlap in morphology, while the rare class provides fewer examples from which the network can learn a stable pattern. This is a plausible explanation, not a clinical conclusion.

## 6.3 Overfitting or underfitting

At the selected best epoch (**{best_epoch}**), training loss is **{best_train_loss:.4f}** and validation loss is **{best_val_loss:.4f}** (validation minus training = **{gap:.4f}**). {overfit_comment} Refer to `chosen_loss_curves.png` and `chosen_accuracy_curves.png` when explaining the final judgement.

## 6.4 Why zero initialisation was not used and why He initialisation is suitable

If every weight started at zero, neurons in the same layer would receive identical gradients and remain identical. They would fail to learn different features. He initialisation uses a variance of approximately `2 / fan_in` (standard deviation `sqrt(2 / fan_in)`) to help maintain a useful signal scale through layers that use ReLU, reducing the risk that activations or gradients become extremely small or large at the start of training.

## 6.5 Momentum and weight decay

Momentum keeps a running velocity. When consecutive mini-batch gradients point in opposite directions, the new gradient partially cancels the previous velocity, smoothing the update and reducing rapid oscillation. Weight decay adds `lambda * weight` to a weight's gradient. Therefore, even when its data gradient is zero, a non-zero weight is gradually pulled toward zero. The assignment applies this decay to weights, not biases.

## 6.6 Helping the rare F class

One possible change is class-weighted cross-entropy: assign a larger loss weight to errors on rare classes such as F. This makes mistakes on under-represented classes contribute more strongly to the objective. Another option is to oversample minority-class training examples. Any such change must be evaluated using validation data and must not use the test set for tuning.

## Test metrics

- Test rows: {int(test['test_rows']):,}
- Test accuracy: {_pct(model_accuracy)}
- Balanced accuracy: {_pct(test['balanced_accuracy'])}
- Always-N accuracy: {_pct(always_n)}

| Class | Test examples | Per-class accuracy |
|---|---:|---:|
"""
    for index, code in enumerate(CLASS_CODES):
        analysis += f"| {code} | {int(counts[index]):,} | {_pct(class_acc[index])} |\n"
    analysis += "\nThe full confusion matrix is saved in `confusion_matrix.csv` and `test_confusion_matrix.png`.\n"
    (RESULTS_DIR / "analysis.md").write_text(analysis, encoding="utf-8")

    # A complete article draft in the requested 800–1500 word range, with
    # measured metrics inserted only after the real test script has run.
    blog = f"""# Classifying Heartbeats from ECG Signals with a Neural Network Built in PyTorch

## Introduction

An electrocardiogram (ECG) records the electrical activity of the heart. A heartbeat is represented by a waveform, and different heartbeat categories can have different shapes. Recognising unusual beats is an important signal-processing and classification problem. In this project, I built a small fully connected neural network to classify one heartbeat into one of five categories: N (normal), S (supraventricular ectopic), V (ventricular ectopic), F (fusion), and Q (unknown or paced).

The main question was not simply whether the network could achieve a high overall accuracy. The dataset is imbalanced: normal beats are much more common than some of the abnormal categories. A classifier can therefore appear successful while still missing many examples from a rare class. For this reason, the evaluation includes a confusion matrix, per-class accuracy, and balanced accuracy as well as ordinary accuracy.

## Dataset and preparation

The project uses the ECG Heartbeat Categorization Dataset from Kaggle, based on single-beat segments from the MIT-BIH Arrhythmia Database. The supplied MIT-BIH CSV files contain 187 signal values followed by one class label in each row. The signal values are already scaled, so they are used as provided rather than being scaled again. The two PTBDB files mentioned on the Kaggle page are not used for this assignment.

The original training CSV contains {int(training['train_rows']) + int(training['validation_rows']):,} rows. It is divided using the exact seed-42 NumPy permutation in the assignment into {int(training['train_rows']):,} training rows and {int(training['validation_rows']):,} validation rows, while the official test file contains {int(test['test_rows']):,} rows. The official Kaggle test file is kept separate. It is not used to choose the learning rate, select an epoch, or update the model.

Figure 1 (`example_beats.png`) plots one heartbeat from each of the five categories. The examples show that ECG classes are identified from waveform morphology rather than from a single sample. Some beats have visibly different shapes, while other categories can have overlapping or subtle patterns. A single example is useful for visual understanding, but it cannot represent every beat in its class.

## Network architecture

The network is a fully connected feed-forward model with the architecture **187–128–64–5**. The input layer receives 187 signal samples. The first hidden layer contains 128 neurons and uses ReLU; the second hidden layer contains 64 neurons and also uses ReLU. The output layer contains five neurons and returns raw logits, one for each heartbeat category.

To follow the assignment's from-scratch requirement, every weight matrix and bias vector is created as an `nn.Parameter`. No `nn.Linear`, `nn.Sequential`, ready-made ReLU layer, or external machine-learning classifier is used. Weights are initialised with the He normal rule, using standard deviation `sqrt(2 / fan_in)`, and biases start at zero. The model has **{training['trainable_parameters']:,} trainable parameters**.

## Loss, gradient check, and training

The loss is softmax cross-entropy. It is calculated with `torch.logsumexp` rather than directly exponentiating all logits, which provides a numerically stable expression. Automatic differentiation calculates the gradients, but the output-layer gradients are also calculated manually for the first 64 training examples. The numerical difference is recorded in `gradient_check.json`; both maximum differences should be below the assignment's `1e-5` tolerance.

Training uses mini-batches of 128 examples for 30 epochs. The three required learning rates—0.01, 0.1, and 0.5—are tested using a fresh model initialised with seed 0 for each run. The parameter update is implemented manually. It combines the gradient with L2 weight decay for weight matrices, updates a momentum velocity with beta 0.9, and subtracts the resulting step from the parameters. Biases do not receive weight decay. The selected learning rate is the one with the lowest validation loss, and the checkpoint is taken from that run's lowest-validation-loss epoch.

Figure 2 (`learning_rate_validation_loss.png`) compares validation loss for all three candidate learning rates. Figures 3 and 4 (`chosen_loss_curves.png` and `chosen_accuracy_curves.png`) show the training and validation curves for the selected run. They help assess convergence and whether a gap develops between training and validation performance.

## Results

The selected learning rate was **{training['chosen_learning_rate']}**, and the lowest validation loss occurred at epoch **{training['best_epoch']}**. At that epoch, training accuracy was **{_pct(training['train_accuracy_at_best_epoch'])}** and validation accuracy was **{_pct(training['validation_accuracy_at_best_epoch'])}**.

On the untouched test set, the model achieved **{_pct(test['test_accuracy'])}** accuracy and **{_pct(test['balanced_accuracy'])}** balanced accuracy. The always-N baseline achieved **{_pct(test['always_N_accuracy'])}**. Per-class test accuracy was:

| Class | Per-class accuracy |
|---|---:|
"""
    for index, code in enumerate(CLASS_CODES):
        blog += f"| {code} | {_pct(class_acc[index])} |\n"
    blog += f"""

Figure 5 (`test_confusion_matrix.png`) visualises the test confusion matrix. Rows represent the true heartbeat category and columns represent the predicted category. The weakest class was **{CLASS_CODES[lowest_class]}**, and its most frequent incorrect prediction was **{CLASS_CODES[confused_class]}**. The precise counts are available in `confusion_matrix.csv`.

## What the results mean

The always-N baseline demonstrates why overall accuracy can be misleading for an imbalanced dataset. A model that predicts only the majority class may obtain a high score without learning to recognise the less common heartbeat types. Per-class accuracy measures recognition separately for each true class, while balanced accuracy averages those five values so that each class has equal importance. These measures give a more complete picture than a single accuracy value.

The weakest class is likely more difficult because it has fewer training examples and its waveform may overlap with the morphology of another category. This is a possible data-and-model explanation rather than a medical interpretation. A future experiment could use class-weighted cross-entropy or oversample rare classes, with all design choices made using training and validation data only.

The training and validation loss gap at the selected epoch was **{gap:.4f}**. The interpretation of this gap should be checked against the plotted curves: a rising validation loss while training loss continues to fall is evidence of overfitting. Selecting the epoch with the lowest validation loss helps avoid simply choosing the final epoch.

## Conclusion and limitations

This project demonstrates the full workflow for a small neural network built from manually created PyTorch parameters: loading and splitting ECG data, constructing the forward pass, implementing stable cross-entropy, checking a gradient by hand, writing the mini-batch update, selecting a learning rate with validation data, and reporting class-aware test metrics. The final Streamlit interface can display a test-set heartbeat or accept a one-row CSV and show the predicted class with probabilities for all five classes.

This is an educational machine-learning demonstration, not a medical device and not a substitute for professional diagnosis. Performance on this dataset does not guarantee the same accuracy on new hospitals, sensors, populations, or recordings. A reliable clinical system would require broader external validation, careful review of false negatives, and appropriate regulatory and clinical assessment.

**Figures to include:** `example_beats.png`, `learning_rate_validation_loss.png`, `chosen_loss_curves.png`, `chosen_accuracy_curves.png`, and `test_confusion_matrix.png`.
"""
    (RESULTS_DIR.parent / "communication" / "medium_blog_draft.md").write_text(blog, encoding="utf-8")

    linkedin = f"""I’ve completed an educational machine-learning project on **ECG Heartbeat Classification** using a neural network built from scratch in PyTorch.

The model classifies 187-sample ECG beats into five categories (N, S, V, F, and Q). I implemented the network parameters, stable softmax cross-entropy, a manual gradient check, and mini-batch training with momentum and weight decay. I also evaluated the model with a confusion matrix, per-class accuracy, and balanced accuracy to account for class imbalance.

**Test accuracy:** {_pct(test['test_accuracy'])}  
**Balanced accuracy:** {_pct(test['balanced_accuracy'])}  
**Selected learning rate:** {training['chosen_learning_rate']}

The project helped me understand why high overall accuracy does not always mean a classifier performs equally well on every class.

Medium article: [Add your published Medium link]  
GitHub repository: [Add your public GitHub repository link]

#MachineLearning #PyTorch #ECG #DeepLearning #SignalProcessing #StudentProject
"""
    (RESULTS_DIR.parent / "communication" / "linkedin_post_draft.md").write_text(linkedin, encoding="utf-8")
    return {"analysis": analysis, "blog_path": "communication/medium_blog_draft.md",
            "linkedin_path": "communication/linkedin_post_draft.md"}
