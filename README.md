# ECG Insight — ECG Heartbeat Classification with PyTorch

<p align="center">
  <strong>A from-scratch neural network for five-class ECG heartbeat classification, with a Streamlit application for interactive prediction and evaluation.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Project" src="https://img.shields.io/badge/Project-ECG%20Classification-147B68">
</p>

## Overview

**ECG Insight** is a machine-learning project that classifies individual electrocardiogram (ECG) heartbeat segments into five categories. Each heartbeat is represented by **187 numerical signal samples**, which are processed by a fully connected neural network implemented with manually defined PyTorch parameters.

The project covers the complete workflow: loading and inspecting the dataset, preparing training and validation splits, implementing the network and loss function, checking gradients, training with momentum and weight decay, evaluating on a held-out test set, and serving predictions through a Streamlit web interface.

The application provides three ways to interact with the model: browse an official test-set row, evaluate a labeled CSV file, or predict a single uploaded heartbeat.

> **Important:** This is an educational machine-learning project. It is not a medical device and must not be used to diagnose, monitor, or make treatment decisions.

## Application Preview

The screenshot below is a preview of the Streamlit interface included with this project. It shows the official test-row explorer. Replace the image file with your latest full application screenshot whenever you are ready; keep the same filename and path and the Markdown below will continue to work.

<p align="center">
  <img src="docs/images/app_demo.png" alt="ECG Insight Streamlit application preview showing the official test-set heartbeat selector" width="900">
</p>

<p align="center"><em>ECG Insight — official test-set heartbeat explorer (demo screenshot).</em></p>

### Replace the preview with your own screenshot

1. Run the application locally using the instructions below.
2. Capture a screenshot of the complete UI that you want to show on GitHub.
3. Save or rename the screenshot to `app_demo.png`.
4. Replace the existing file at `docs/images/app_demo.png` in your repository.
5. Commit and push the replacement image. **No README edit is required** as long as the path and filename stay the same.

The image is embedded using this standard GitHub Markdown syntax:

```markdown
<p align="center">
  <img src="docs/images/app_demo.png" alt="ECG Insight Streamlit application preview" width="900">
</p>
```

## Model at a Glance

| Component | Configuration |
|---|---|
| Input | 187 ECG signal values per heartbeat |
| Hidden layers | 128 and 64 neurons |
| Hidden activation | ReLU |
| Output layer | 5 class logits |
| Architecture | `187 → 128 → 64 → 5` |
| Trainable parameters | **32,645** |
| Weight initialization | He normal initialization |
| Loss | Numerically stable softmax cross-entropy |
| Training framework | PyTorch |

### Heartbeat classes

| Label | Class | Description |
|---:|:---:|---|
| 0 | **N** | Normal beat |
| 1 | **S** | Supraventricular ectopic beat |
| 2 | **V** | Ventricular ectopic beat |
| 3 | **F** | Fusion beat |
| 4 | **Q** | Unknown / paced beat |

The model returns five class probabilities for each input heartbeat. The predicted class is the class with the highest probability (argmax over the output logits/softmax probabilities).

## Dataset

This project uses the **ECG Heartbeat Categorization Dataset** published on Kaggle and based on the MIT-BIH heartbeat data:

- [Kaggle — ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

Download the dataset and place these two files in the project's `data/` directory:

- `mitbih_train.csv`
- `mitbih_test.csv`

The CSV files are headerless. Each row contains 187 ECG signal values followed by one integer class label (0–4), giving 188 columns in total. The supplied signal data spans **0 to 1** in the training file; the preparation pipeline retains the provided scaled values rather than applying another scaler.

### Dataset split used in the recorded run

The original training CSV contains 87,554 rows. The project reserves 10% of that file for validation using a NumPy permutation with seed 42. The separate Kaggle test file remains held out for final evaluation.

| Split | Rows | Purpose |
|---|---:|---|
| Training | 78,799 | Learn model parameters |
| Validation | 8,755 | Compare learning rates and select the checkpoint |
| Official test | 21,892 | Final held-out evaluation |

The test CSV is not used for gradient updates, learning-rate selection, or checkpoint selection. Do not merge it into the training data or create a new random split from it when reproducing the reported final test result.

## Recorded Results

The following figures are taken from the `results/` files in the project archive. They describe the saved run included with this project; results may differ if the model is retrained with a different software environment, device, or configuration.

### Training and model selection

| Metric | Recorded value |
|---|---:|
| Candidate learning rates | `0.01`, `0.1`, `0.5` |
| Selected learning rate | **0.5** |
| Best validation epoch | **25** |
| Training loss at selected epoch | 0.05160 |
| Validation loss at selected epoch | 0.06764 |
| Training accuracy at selected epoch | 98.48% |
| Validation accuracy at selected epoch | 98.05% |
| Batch size | 128 |
| Maximum epochs per run | 30 |
| Momentum coefficient | 0.9 |
| Weight decay | 0.0001 |
| Training device in recorded run | CPU |

### Official test-set performance

| Metric | Recorded value |
|---|---:|
| Test examples | 21,892 |
| Correct predictions | 21,407 |
| Incorrect predictions | 485 |
| **Overall test accuracy** | **97.78%** |
| **Balanced accuracy** | **85.23%** |
| Always-predict-N baseline | 82.76% |

### Per-class test performance

| Class | Test examples | Correct predictions | Per-class accuracy |
|:---:|---:|---:|---:|
| N | 18,118 | 17,996 | 99.33% |
| S | 556 | 391 | 70.32% |
| V | 1,448 | 1,349 | 93.16% |
| F | 162 | 107 | 66.05% |
| Q | 1,608 | 1,564 | 97.26% |

**Interpreting the results:** The dataset is strongly imbalanced, with Normal beats making up most of the test examples. This is why the overall accuracy is higher than balanced accuracy. The lower results for S and F show that the model does not perform equally well across every class. Per-class metrics and the confusion matrix should therefore be considered alongside overall accuracy.

## Streamlit Application Features

The Streamlit interface has three prediction/evaluation modes:

1. **Official test row** — Select a 1-based row number from the held-out test file and load that heartbeat. The application displays the ground-truth label, predicted class, class probabilities, and signal waveform.
2. **Evaluate labeled CSV** — Upload a headerless CSV with 187 signal values and a final class label (0–4) per row. The app calculates accuracy, balanced accuracy, per-class accuracy, and a confusion matrix, and allows row-by-row predictions to be downloaded. Uploaded data is evaluated only; it does not retrain the model.
3. **Predict one heartbeat** — Upload a CSV containing exactly one row of 187 numeric signal values, without a label. The model returns the predicted class and probability distribution. Because the true label is not supplied, this mode cannot calculate accuracy by itself.

## Project Structure

The following tree shows the main files and folders used in the VS Code project. Dataset files, the virtual environment, and generated binary artifacts are shown as local/runtime items; they should generally not be committed to GitHub.

```text
ECG_Heartbeat_Classification_Project03/
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme and UI configuration
│
├── app/
│   ├── __init__.py
│   └── app.py                      # Main Streamlit interface
│
├── communication/
│   ├── README.md                   # Communication deliverable notes
│   ├── linkedin_post_draft.md      # LinkedIn post draft
│   └── medium_blog_draft.md        # Medium article draft
│
├── data/                           # Local dataset directory (not committed)
│   ├── mitbih_train.csv            # Download from Kaggle
│   ├── mitbih_test.csv             # Download from Kaggle
│   ├── one_heartbeat/              # Optional single-beat CSV examples
│   └── test_samples/                # Optional generated test subsets
│
├── docs/
│   └── images/
│       └── app_demo.png            # GitHub README application screenshot
│
├── results/                        # Generated outputs and saved model
│   ├── best_model.pt               # Selected model checkpoint
│   ├── prepared_data.npz           # Prepared train/validation/test arrays
│   ├── training_summary.json       # Training configuration and history
│   ├── test_metrics.json           # Overall and per-class test metrics
│   ├── confusion_matrix.csv        # Confusion matrix values
│   ├── per_class_accuracy.csv      # Per-class evaluation metrics
│   ├── test_predictions.csv        # Row-level test predictions/probabilities
│   ├── gradient_check.json         # Manual/autograd gradient comparison
│   ├── example_beats.png           # Example ECG waveform plot
│   ├── learning_rate_validation_loss.png
│   ├── chosen_loss_curves.png
│   ├── chosen_accuracy_curves.png
│   └── test_confusion_matrix.png   # Test confusion-matrix figure
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Shared paths and model/training settings
│   ├── load_data.py                # Load original Kaggle CSV files
│   ├── prepare.py                  # Validate and split the dataset
│   ├── plot_beats.py               # Plot one heartbeat per class
│   ├── model.py                    # Manual 187-128-64-5 network and loss
│   ├── grad_check.py               # Output-layer gradient verification
│   ├── train.py                    # Manual mini-batch training and selection
│   ├── metrics.py                  # Accuracy and confusion-matrix metrics
│   ├── inference.py                # CSV validation and batch inference
│   ├── test_model.py               # Final held-out test evaluation
│   └── reporting.py                # Analysis and communication drafts
│
├── tests/
│   └── smoke_test.py               # Lightweight synthetic code checks
│
├── .gitignore
├── app.py                          # Streamlit application entry point
├── make_test_samples.py            # Create balanced/random test subsets
├── requirements.txt                # Python dependencies
├── run_pipeline.py                 # Run preparation, checks, training, testing
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd ECG_Heartbeat_Classification_Project03
```

Replace the repository URL with your own GitHub repository address.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes NumPy, pandas, Matplotlib, PyTorch, and Streamlit. For GPU training, install a PyTorch build compatible with your CUDA driver and hardware; otherwise, the project can run on CPU.

### 4. Add the dataset

Download the Kaggle dataset and copy the two required files into `data/`:

```text
data/
├── mitbih_train.csv
└── mitbih_test.csv
```

The data loader checks that each CSV has 188 columns and reports a clear error if either file is missing or has an unexpected shape.

## Run the Project

Run commands from the repository root (the directory containing `README.md`).

### Run the complete pipeline

```bash
python run_pipeline.py --device auto
```

Use `--device cuda` to explicitly request a supported CUDA device, or `--device cpu` to force CPU execution:

```bash
python run_pipeline.py --device cuda
```

The pipeline runs data preparation, example waveform plotting, gradient checking, learning-rate experiments, model selection, and final test evaluation in order.

### Or run each stage separately

**1. Load and prepare data**

```bash
python -m src.load_data
python -m src.plot_beats
python -m src.prepare
```

**2. Inspect the model and verify gradients**

```bash
python -m src.model
python -m src.grad_check
```

**3. Train and select the model**

```bash
python -m src.train --device auto
```

**4. Evaluate the official test file**

```bash
python -m src.test_model --device auto
```

Keep the official test file out of training and validation-based model selection. Run final test evaluation after choosing the learning rate and checkpoint.

### Launch the Streamlit application

After the data has been prepared and a model checkpoint has been created, start the web application:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, which can be opened in a browser.

## Generated Outputs

The project writes generated files to `results/`:

- `prepared_data.npz` — prepared training, validation and official test arrays.
- `data_summary.json` and `table1_data.csv` — dataset shapes, split sizes, signal range and class counts.
- `gradient_check.json` — manual versus automatic output-layer gradient differences.
- `best_model.pt` — selected trained model checkpoint.
- `training_summary.json` and `table2_training_summary.csv` — training configuration, learning-rate choice and selected checkpoint metrics.
- `learning_rate_comparison.csv` — validation results for candidate learning rates.
- `learning_rate_validation_loss.png` — comparison of validation-loss curves.
- `chosen_loss_curves.png` and `chosen_accuracy_curves.png` — selected run's training/validation curves.
- `test_metrics.json`, `per_class_accuracy.csv`, and `confusion_matrix.csv` — final evaluation metrics.
- `test_predictions.csv` — row-level predictions and class probabilities.
- `test_confusion_matrix.png` — confusion-matrix visualization.
- `analysis.md` — result-dependent discussion generated after testing.

The project also creates Medium and LinkedIn draft files in `communication/` after the test evaluation step.

## Create Smaller Test Samples (Optional)

After placing `mitbih_test.csv` in `data/`, you can create a balanced 500-row subset and a random 1,000-row subset:

```bash
python make_test_samples.py
```

The files are saved under `data/test_samples/`. These are **subsets of the official test data**, not independent new test examples. They are useful for demonstrating the upload workflow, but should not be presented as a second independent test evaluation.

## Reproducibility and Validation Checks

The implementation includes several checks intended to make the workflow easier to inspect:

- The dataset loader validates the 188-column CSV format.
- The validation split uses a fixed seed (`42`).
- Model initialization uses a fixed seed (`0`) for each learning-rate run.
- The network has 32,645 trainable parameters.
- The output-layer gradient check compares manually calculated gradients against PyTorch autograd with a `1e-5` tolerance.
- Training uses the specified batch size, learning-rate candidates, momentum, and weight decay.
- The checkpoint is selected using validation loss.
- The official test set is reserved for final evaluation.
- `tests/smoke_test.py` runs lightweight synthetic checks; it does not measure ECG accuracy.

## Limitations

- The dataset is imbalanced, so overall accuracy alone does not describe performance on minority classes.
- The recorded model has lower per-class accuracy for S and F than for N, V, and Q.
- Evaluation on this benchmark does not establish generalization to other hospitals, devices, recording protocols, or patient populations.
- Softmax probabilities are model scores and should not be interpreted as calibrated medical confidence.
- This application is for educational demonstration only and is not intended for clinical use.

## Future Improvements

Potential follow-up experiments include class-weighted cross-entropy, carefully controlled oversampling, calibration analysis, external-dataset evaluation, and model comparisons. Any model-selection decisions should be made using training and validation data, leaving a separate test set untouched for final reporting.

## Acknowledgements

- [ECG Heartbeat Categorization Dataset — Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)

## License

No license is specified in this repository. Add a `LICENSE` file before allowing others to reuse, modify, or redistribute the project.

---

<p align="center"><strong>ECG Insight</strong> · An educational ECG classification project built with PyTorch and Streamlit.</p>
