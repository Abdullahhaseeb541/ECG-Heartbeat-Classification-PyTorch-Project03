# Classifying Heartbeats from ECG Signals with a Neural Network Built in PyTorch

## Introduction

An electrocardiogram (ECG) records the electrical activity of the heart. A heartbeat is represented by a waveform, and different heartbeat categories can have different shapes. Recognising unusual beats is an important signal-processing and classification problem. In this project, I built a small fully connected neural network to classify one heartbeat into one of five categories: N (normal), S (supraventricular ectopic), V (ventricular ectopic), F (fusion), and Q (unknown or paced).

The main question was not simply whether the network could achieve a high overall accuracy. The dataset is imbalanced: normal beats are much more common than some of the abnormal categories. A classifier can therefore appear successful while still missing many examples from a rare class. For this reason, the evaluation includes a confusion matrix, per-class accuracy, and balanced accuracy as well as ordinary accuracy.

## Dataset and preparation

The project uses the ECG Heartbeat Categorization Dataset from Kaggle, based on single-beat segments from the MIT-BIH Arrhythmia Database. The supplied MIT-BIH CSV files contain 187 signal values followed by one class label in each row. The signal values are already scaled, so they are used as provided rather than being scaled again. The two PTBDB files mentioned on the Kaggle page are not used for this assignment.

The original training CSV contains 87,554 rows. It is divided using the exact seed-42 NumPy permutation in the assignment into 78,799 training rows and 8,755 validation rows, while the official test file contains 21,892 rows. The official Kaggle test file is kept separate. It is not used to choose the learning rate, select an epoch, or update the model.

Figure 1 (`example_beats.png`) plots one heartbeat from each of the five categories. The examples show that ECG classes are identified from waveform morphology rather than from a single sample. Some beats have visibly different shapes, while other categories can have overlapping or subtle patterns. A single example is useful for visual understanding, but it cannot represent every beat in its class.

## Network architecture

The network is a fully connected feed-forward model with the architecture **187–128–64–5**. The input layer receives 187 signal samples. The first hidden layer contains 128 neurons and uses ReLU; the second hidden layer contains 64 neurons and also uses ReLU. The output layer contains five neurons and returns raw logits, one for each heartbeat category.

To follow the assignment's from-scratch requirement, every weight matrix and bias vector is created as an `nn.Parameter`. No `nn.Linear`, `nn.Sequential`, ready-made ReLU layer, or external machine-learning classifier is used. Weights are initialised with the He normal rule, using standard deviation `sqrt(2 / fan_in)`, and biases start at zero. The model has **32,645 trainable parameters**.

## Loss, gradient check, and training

The loss is softmax cross-entropy. It is calculated with `torch.logsumexp` rather than directly exponentiating all logits, which provides a numerically stable expression. Automatic differentiation calculates the gradients, but the output-layer gradients are also calculated manually for the first 64 training examples. The numerical difference is recorded in `gradient_check.json`; both maximum differences should be below the assignment's `1e-5` tolerance.

Training uses mini-batches of 128 examples for 30 epochs. The three required learning rates—0.01, 0.1, and 0.5—are tested using a fresh model initialised with seed 0 for each run. The parameter update is implemented manually. It combines the gradient with L2 weight decay for weight matrices, updates a momentum velocity with beta 0.9, and subtracts the resulting step from the parameters. Biases do not receive weight decay. The selected learning rate is the one with the lowest validation loss, and the checkpoint is taken from that run's lowest-validation-loss epoch.

Figure 2 (`learning_rate_validation_loss.png`) compares validation loss for all three candidate learning rates. Figures 3 and 4 (`chosen_loss_curves.png` and `chosen_accuracy_curves.png`) show the training and validation curves for the selected run. They help assess convergence and whether a gap develops between training and validation performance.

## Results

The selected learning rate was **0.5**, and the lowest validation loss occurred at epoch **25**. At that epoch, training accuracy was **98.48%** and validation accuracy was **98.05%**.

On the untouched test set, the model achieved **97.78%** accuracy and **85.23%** balanced accuracy. The always-N baseline achieved **82.76%**. Per-class test accuracy was:

| Class | Per-class accuracy |
|---|---:|
| N | 99.33% |
| S | 70.32% |
| V | 93.16% |
| F | 66.05% |
| Q | 97.26% |


Figure 5 (`test_confusion_matrix.png`) visualises the test confusion matrix. Rows represent the true heartbeat category and columns represent the predicted category. The weakest class was **F**, and its most frequent incorrect prediction was **N**. The precise counts are available in `confusion_matrix.csv`.

## What the results mean

The always-N baseline demonstrates why overall accuracy can be misleading for an imbalanced dataset. A model that predicts only the majority class may obtain a high score without learning to recognise the less common heartbeat types. Per-class accuracy measures recognition separately for each true class, while balanced accuracy averages those five values so that each class has equal importance. These measures give a more complete picture than a single accuracy value.

The weakest class is likely more difficult because it has fewer training examples and its waveform may overlap with the morphology of another category. This is a possible data-and-model explanation rather than a medical interpretation. A future experiment could use class-weighted cross-entropy or oversample rare classes, with all design choices made using training and validation data only.

The training and validation loss gap at the selected epoch was **0.0160**. The interpretation of this gap should be checked against the plotted curves: a rising validation loss while training loss continues to fall is evidence of overfitting. Selecting the epoch with the lowest validation loss helps avoid simply choosing the final epoch.

## Conclusion and limitations

This project demonstrates the full workflow for a small neural network built from manually created PyTorch parameters: loading and splitting ECG data, constructing the forward pass, implementing stable cross-entropy, checking a gradient by hand, writing the mini-batch update, selecting a learning rate with validation data, and reporting class-aware test metrics. The final Streamlit interface can display a test-set heartbeat or accept a one-row CSV and show the predicted class with probabilities for all five classes.

This is an educational machine-learning demonstration, not a medical device and not a substitute for professional diagnosis. Performance on this dataset does not guarantee the same accuracy on new hospitals, sensors, populations, or recordings. A reliable clinical system would require broader external validation, careful review of false negatives, and appropriate regulatory and clinical assessment.

**Figures to include:** `example_beats.png`, `learning_rate_validation_loss.png`, `chosen_loss_curves.png`, `chosen_accuracy_curves.png`, and `test_confusion_matrix.png`.
