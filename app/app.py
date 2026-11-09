"""Professional Streamlit interface for ECG heartbeat classification."""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import RESULTS_DIR, PREPARED_FILE, CLASS_CODES, CLASS_NAMES, N_SAMPLES
from src.inference import predict_batch, read_labeled_csv, read_signal_csv
from src.metrics import accuracy, balanced_accuracy, confusion_matrix, per_class_accuracy
from src.model import ECGNet

CHECKPOINT_PATH = RESULTS_DIR / "best_model.pt"
TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.json"
FS = 125

st.set_page_config(
    page_title="ECG Insight | Heartbeat AI",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner=False)
def load_model():
    model = ECGNet()
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_test_data():
    with np.load(PREPARED_FILE) as archive:
        x_test = archive["X_test"].astype(np.float32, copy=True)
        y_test = archive["y_test"].astype(np.int64, copy=True)
    return x_test, y_test


def predict_one(model, signal):
    tensor = torch.as_tensor(signal, dtype=torch.float32).reshape(1, N_SAMPLES)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
        predicted = int(torch.argmax(logits, dim=1).item())
    return predicted, probabilities


def plot_signal(signal):
    time_ms = np.arange(N_SAMPLES) * (1000.0 / FS)
    fig, axis = plt.subplots(figsize=(11, 3.1))
    axis.plot(time_ms, signal, linewidth=1.7)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Scaled amplitude")
    axis.set_title("ECG waveform · 187 samples at 125 Hz", loc="left", fontweight="bold")
    axis.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def plot_probabilities(probabilities):
    fig, axis = plt.subplots(figsize=(7, 2.8))
    colors = [
        "#147b68" if value == max(probabilities) else "#9cb4c7"
        for value in probabilities
    ]
    bars = axis.bar(CLASS_CODES, probabilities, color=colors, width=0.62)
    axis.bar_label(bars, labels=[f"{value:.1%}" for value in probabilities], padding=3)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Probability")
    axis.set_title("Predicted class probabilities", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    with st.sidebar:
        st.title("Heartbeat AI")
        st.caption("ECG Insight · Machine Learning · Project 03")
        st.divider()
        mode = st.segmented_control(
            "Prediction mode",
            ["Official test row", "Evaluate labeled CSV", "Predict one heartbeat"],
            default="Official test row",
            key="prediction_mode",
            width="stretch",
        )
        st.divider()
        st.markdown("**Model pipeline**")
        st.caption("187 signal values · 128 → 64 hidden units · 5 heartbeat classes")
        st.divider()
        st.caption("Academic demonstration only. Not intended for clinical diagnosis.")

    st.title("Understand every heartbeat.")
    st.caption(
        "Explore an official test sample, evaluate a labeled test CSV, "
        "or predict a single 187-point heartbeat."
    )

    if mode == "Official test row" and not PREPARED_FILE.exists():
        st.error("Prepared data was not found. Complete Task 1 first: download both Kaggle CSV files into `data/`, then run `python -m src.prepare`.")
        st.code("python -m src.prepare\npython -m src.grad_check\npython -m src.train\npython -m src.test_model", language="bash")
        return
    if not CHECKPOINT_PATH.exists():
        st.warning("No trained model checkpoint exists yet. Run `python -m src.train` and `python -m src.test_model` from the project root, then refresh this page.")
        return

    try:
        model = load_model()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Unable to load the trained model: {error}")
        return

    X_test = np.empty((0, N_SAMPLES), dtype=np.float32)
    y_test = np.empty(0, dtype=np.int64)
    if mode == "Official test row":
        try:
            X_test, y_test = load_test_data()
        except (OSError, KeyError, ValueError) as error:
            st.error(f"Unable to load the prepared official test set: {error}")
            return

    test_accuracy_value = None
    balanced_value = None
    if TEST_METRICS_PATH.exists():
        try:
            metrics = json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            st.warning(f"Could not read the saved official test metrics: {error}")
        else:
            test_accuracy_value = metrics.get("test_accuracy")
            balanced_value = metrics.get("balanced_accuracy")

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Official test samples",
        f"{len(X_test):,}" if mode == "Official test row" else "Select official mode",
        border=True,
    )
    metric_cols[1].metric("Signal points", "187", border=True)
    metric_cols[2].metric(
        "Test accuracy",
        f"{test_accuracy_value:.2%}" if test_accuracy_value is not None else "Not evaluated",
        border=True,
    )
    metric_cols[3].metric(
        "Balanced accuracy",
        f"{balanced_value:.2%}" if balanced_value is not None else "Not evaluated",
        border=True,
    )

    if mode == "Evaluate labeled CSV":
        with st.container(border=True):
            st.subheader("Evaluate a labeled test file")
            st.caption(
                "Upload a headerless CSV with 187 signal values and a final class label (0–4) per row. "
                "The uploaded data is used only for evaluation; the model is not retrained."
            )
            uploaded_test = st.file_uploader(
                "Choose a labeled ECG test CSV",
                type=["csv"],
                accept_multiple_files=False,
                key="labeled_test_csv",
            )

        if uploaded_test is None:
            st.info("Upload a labeled CSV to calculate its accuracy and class-wise results.")
            return

        try:
            uploaded_signals, uploaded_labels = read_labeled_csv(uploaded_test.getvalue())
        except ValueError as error:
            st.error(f"Could not evaluate this CSV: {error}")
            return

        with st.spinner(f"Evaluating {len(uploaded_signals):,} labeled heartbeats..."):
            uploaded_predictions, uploaded_probabilities = predict_batch(model, uploaded_signals)

        uploaded_matrix = confusion_matrix(uploaded_labels, uploaded_predictions)
        uploaded_accuracy = accuracy(uploaded_labels, uploaded_predictions)
        uploaded_balanced_accuracy = balanced_accuracy(uploaded_matrix)
        class_counts = uploaded_matrix.sum(axis=1)
        class_accuracies = per_class_accuracy(uploaded_matrix)
        class_accuracies[class_counts == 0] = np.nan

        st.success(f"Evaluation complete for {len(uploaded_labels):,} rows. The model was not retrained.")
        uploaded_metric_cols = st.columns(4)
        uploaded_metric_cols[0].metric("Uploaded samples", f"{len(uploaded_labels):,}", border=True)
        uploaded_metric_cols[1].metric("Accuracy", f"{uploaded_accuracy:.2%}", border=True)
        uploaded_metric_cols[2].metric("Balanced accuracy", f"{uploaded_balanced_accuracy:.2%}", border=True)
        uploaded_metric_cols[3].metric(
            "Classes present",
            f"{int(np.count_nonzero(class_counts))} / {len(CLASS_CODES)}",
            border=True,
        )
        st.caption("Balanced accuracy averages only classes represented in this uploaded test file.")

        table_col, matrix_col = st.columns(2)
        with table_col:
            st.subheader("Per-class performance")
            st.dataframe(
                pd.DataFrame({
                    "Class": CLASS_CODES,
                    "Heartbeat type": [name.split(" - ", 1)[1] for name in CLASS_NAMES],
                    "Test examples": class_counts,
                    "Accuracy": [
                        f"{value:.2%}" if np.isfinite(value) else "N/A"
                        for value in class_accuracies
                    ],
                }),
                hide_index=True,
                alt="Accuracy for each represented heartbeat class",
            )
        with matrix_col:
            st.subheader("Confusion matrix")
            confusion_frame = pd.DataFrame(
                uploaded_matrix,
                index=CLASS_CODES,
                columns=CLASS_CODES,
            )
            confusion_frame.index.name = "True class"
            st.dataframe(
                confusion_frame,
                alt="Uploaded test-set confusion matrix with true classes as rows",
            )

        uploaded_results = pd.DataFrame({
            "row_number_1_based": np.arange(1, len(uploaded_labels) + 1),
            "true_class": [CLASS_CODES[value] for value in uploaded_labels],
            "predicted_class": [CLASS_CODES[value] for value in uploaded_predictions],
        })
        for class_id, code in enumerate(CLASS_CODES):
            uploaded_results[f"probability_{code}"] = uploaded_probabilities[:, class_id]
        st.download_button(
            "Download row-by-row predictions",
            data=uploaded_results.to_csv(index=False).encode("utf-8"),
            file_name="uploaded_test_predictions.csv",
            mime="text/csv",
            width="stretch",
        )
        return

    signal = None
    true_class = None
    row_number = None
    if mode == "Official test row":
        with st.container(border=True):
            st.subheader("Quick start · Explore the official test set")
            st.caption(
                "Each row is one real heartbeat from the assignment's held-out test CSV. "
                "Choose a row and load it to refresh the waveform and prediction below."
            )

            with st.form("official_test_row_form", border=False):
                row_col, button_col = st.columns(
                    [1, 1],
                    vertical_alignment="bottom",
                )
                with row_col:
                    st.subheader(f"Valid record range: 1-{len(X_test):,}")
                    row_number = st.number_input(
                        "Test-set row number (1-based)",
                        min_value=1,
                        max_value=len(X_test),
                        value=1,
                        step=1,
                        key="official_test_row_number",
                        help=(
                            "This is the row's position in mitbih_test.csv, not a "
                            "heartbeat measurement. Valid rows are 1 through "
                            f"{len(X_test):,}."
                        ),
                    )
                with button_col:
                    st.form_submit_button(
                        "Load selected heartbeat",
                        type="primary",
                        icon=":material/monitor_heart:",
                        width="stretch",
                    )

            st.caption(
                "1. Choose a test row  ·  2. Select **Load selected heartbeat**  ·  "
                "3. Review its true class, model prediction, probabilities, and waveform."
            )
            row_number = int(row_number)
            signal = X_test[row_number - 1]
            true_class = int(y_test[row_number - 1])
            st.subheader("Heartbeat waveform")
            st.caption("The plotted signal is the same 187-value input passed to the neural network.")
            fig = plot_signal(signal)
            st.pyplot(fig, width="stretch", alt="Official test ECG heartbeat waveform")
            plt.close(fig)

        left, right = st.columns([1.25, 1], vertical_alignment="top")
        with left:
            with st.container(border=True):
                st.subheader(f"Selected heartbeat · test row {row_number:,}")
                st.metric(
                    "Ground-truth class",
                    CLASS_CODES[true_class],
                    help=CLASS_NAMES[true_class],
                    border=True,
                )
                st.caption(
                    f"{CLASS_NAMES[true_class].split(' - ', 1)[1]} · "
                    "from the official held-out test set"
                )
        with right:
            with st.container(border=True):
                st.subheader("Official test data")
                st.caption(
                    "The model was trained using the training split only. "
                    "These held-out rows are used for testing."
                )
                test_csv = pd.DataFrame(
                    np.column_stack((X_test, y_test))
                ).to_csv(index=False, header=False).encode("utf-8")
                st.download_button(
                    "Download test-set CSV",
                    data=test_csv,
                    file_name="mitbih_test.csv",
                    mime="text/csv",
                    width="stretch",
                )
                prediction_file = RESULTS_DIR / "test_predictions.csv"
                if prediction_file.exists():
                    st.download_button(
                        "Download test predictions",
                        data=prediction_file.read_bytes(),
                        file_name="test_predictions.csv",
                        mime="text/csv",
                        width="stretch",
                    )
    else:
        with st.container(border=True):
            st.subheader("Upload one heartbeat")
            st.caption(
                "Upload a headerless CSV containing exactly one row and 187 numeric "
                "signal values. Do not include the label column."
            )
            uploaded = st.file_uploader(
                "Choose a one-row ECG CSV",
                type=["csv"],
                accept_multiple_files=False,
            )
            if uploaded is not None:
                try:
                    signal = read_signal_csv(uploaded.getvalue())
                except ValueError as error:
                    st.error(f"Could not read this file: {error}")

    if signal is None:
        st.info("Select a test row or upload a valid one-row CSV to display a prediction.")
        return

    predicted, probabilities = predict_one(model, signal)
    confidence = float(probabilities[predicted])
    result_cols = st.columns([1, 1.35])
    with result_cols[0]:
        with st.container(border=True):
            st.subheader("Predicted heartbeat class")
            st.metric("Class", CLASS_CODES[predicted], border=True)
            st.caption(CLASS_NAMES[predicted].split(" - ", 1)[1])
            st.progress(confidence, text=f"Model probability · {confidence:.2%}")
            if true_class is not None:
                if predicted == true_class:
                    st.success(f"Correct prediction · actual class is {CLASS_CODES[true_class]}.")
                else:
                    st.warning(
                        f"Prediction differs from ground truth · actual {CLASS_CODES[true_class]}, "
                        f"predicted {CLASS_CODES[predicted]}."
                    )
            st.download_button(
                "Download this 187-point signal",
                data=pd.DataFrame(signal.reshape(1, -1)).to_csv(
                    index=False,
                    header=False,
                ).encode("utf-8"),
                file_name=f"ecg_heartbeat_{row_number or 'uploaded'}.csv",
                mime="text/csv",
                width="stretch",
            )
    with result_cols[1]:
        with st.container(border=True):
            st.subheader("Class probability distribution")
            probability_fig = plot_probabilities(probabilities)
            st.pyplot(
                probability_fig,
                width="stretch",
                alt="Predicted probability for each heartbeat class",
            )
            plt.close(probability_fig)
            st.dataframe(
                pd.DataFrame({
                    "Class": CLASS_CODES,
                    "Heartbeat type": [
                        name.split(" - ", 1)[1] for name in CLASS_NAMES
                    ],
                    "Probability": [f"{value:.2%}" for value in probabilities],
                }),
                hide_index=True,
                alt="Predicted probabilities and heartbeat class names",
            )

    if mode == "Predict one heartbeat":
        st.subheader("Heartbeat waveform")
        st.caption("The plotted signal is the same 187-value input passed to the neural network.")
        fig = plot_signal(signal)
        st.pyplot(fig, width="stretch", alt="Uploaded ECG heartbeat waveform")
        plt.close(fig)
    st.caption("This model is an educational classification project. Probabilities are model outputs, not clinical risk estimates or a diagnosis.")


if __name__ == "__main__":
    main()
