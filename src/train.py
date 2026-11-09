"""Task 4: train from scratch with mini-batches, momentum and weight decay.

No torch.optim class is used. The parameter update follows the equation in
Project 03 exactly, and weight decay is applied to weights but not biases.
"""
import argparse
import csv
import json
import math
import time
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import (
    PREPARED_FILE, RESULTS_DIR, SEED, BATCH_SIZE, EPOCHS,
    LEARNING_RATES, MOMENTUM, WEIGHT_DECAY,
)
from .model import ECGNet, cross_entropy
from .prepare import prepare_data


def choose_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation has no CUDA device.")
    return device


def load_prepared_tensors(device):
    if not PREPARED_FILE.exists():
        prepare_data()
    with np.load(PREPARED_FILE) as data:
        tensors = {
            key: torch.as_tensor(data[key], dtype=(torch.long if key.startswith("y_") else torch.float32), device=device)
            for key in ("X_train", "y_train", "X_val", "y_val")
        }
    return tensors


def evaluate(model, x, y, batch_size=4096):
    """Calculate full-split mean loss and accuracy without storing all logits."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_rows = len(y)

    with torch.no_grad():
        for start in range(0, total_rows, batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]
            logits = model(xb)
            batch_loss = cross_entropy(logits, yb)
            total_loss += batch_loss.item() * len(yb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()

    return total_loss / total_rows, total_correct / total_rows


def _cpu_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_one_learning_rate(alpha, tensors, device, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train one fresh, seed-0 model using the assignment's manual update."""
    torch.manual_seed(SEED)
    model = ECGNet().to(device)
    initial_train_loss, _ = evaluate(model, tensors["X_train"], tensors["y_train"])

    # Parameter order is weights first, followed by biases, as defined in ECGNet.
    weight_velocities = [torch.zeros_like(parameter) for parameter in model.weights]
    bias_velocities = [torch.zeros_like(parameter) for parameter in model.biases]

    history = {
        "epoch": [], "train_loss": [], "validation_loss": [],
        "train_accuracy": [], "validation_accuracy": [],
    }
    best_validation_loss = math.inf
    best_epoch = None
    best_state = None
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(SEED)
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(tensors["y_train"]), generator=shuffle_generator).to(device)

        for start in range(0, len(permutation), batch_size):
            batch_indices = permutation[start:start + batch_size]
            xb = tensors["X_train"].index_select(0, batch_indices)
            yb = tensors["y_train"].index_select(0, batch_indices)

            model.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss for learning rate {alpha} at epoch {epoch}. "
                    "Check the update rule or try the specified candidate rates again."
                )
            loss.backward()

            # Manual momentum + L2 decay update; biases receive no weight decay.
            with torch.no_grad():
                for parameter, velocity in zip(model.weights, weight_velocities):
                    gradient = parameter.grad + WEIGHT_DECAY * parameter
                    velocity.mul_(MOMENTUM).add_(gradient, alpha=(1.0 - MOMENTUM))
                    parameter.add_(velocity, alpha=-alpha)

                for parameter, velocity in zip(model.biases, bias_velocities):
                    gradient = parameter.grad
                    velocity.mul_(MOMENTUM).add_(gradient, alpha=(1.0 - MOMENTUM))
                    parameter.add_(velocity, alpha=-alpha)

        train_loss, train_accuracy = evaluate(model, tensors["X_train"], tensors["y_train"])
        validation_loss, validation_accuracy = evaluate(model, tensors["X_val"], tensors["y_val"])

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["validation_loss"].append(float(validation_loss))
        history["train_accuracy"].append(float(train_accuracy))
        history["validation_accuracy"].append(float(validation_accuracy))

        print(
            f"alpha={alpha:<4} | epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.5f} | val loss={validation_loss:.5f} | "
            f"train acc={train_accuracy:.4%} | val acc={validation_accuracy:.4%}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model)

    elapsed = time.perf_counter() - started
    return {
        "alpha": float(alpha),
        "history": history,
        "initial_train_loss": float(initial_train_loss),
        "best_validation_loss": float(best_validation_loss),
        "best_epoch": int(best_epoch),
        "validation_accuracy_at_best_epoch": float(history["validation_accuracy"][best_epoch - 1]),
        "train_loss_at_best_epoch": float(history["train_loss"][best_epoch - 1]),
        "validation_loss_at_best_epoch": float(history["validation_loss"][best_epoch - 1]),
        "train_accuracy_at_best_epoch": float(history["train_accuracy"][best_epoch - 1]),
        "elapsed_seconds": float(elapsed),
        "best_state_dict": best_state,
    }



def train(alpha):
    """Assignment-style train(alpha) entry point for one fresh candidate run."""
    device = choose_device("auto")
    tensors = load_prepared_tensors(device)
    return train_one_learning_rate(alpha, tensors, device)

def _save_plots(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Task 4.2: compare all three validation-loss curves.
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for result in results:
        history = result["history"]
        axis.plot(history["epoch"], history["validation_loss"], linewidth=2,
                  label=f"Learning rate = {result['alpha']}")
    axis.set_title("Validation Loss for the Three Learning Rates")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean softmax cross-entropy")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_rate_validation_loss.png", dpi=180)
    plt.close(fig)

    chosen = min(results, key=lambda item: item["best_validation_loss"])
    history = chosen["history"]

    # Task 4.3: loss chart for the selected learning rate.
    fig, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(history["epoch"], history["train_loss"], linewidth=2, label="Training loss")
    axis.plot(history["epoch"], history["validation_loss"], linewidth=2, label="Validation loss")
    axis.axvline(chosen["best_epoch"], linestyle="--", linewidth=1,
                 label=f"Best epoch: {chosen['best_epoch']}")
    axis.set_title(f"Training and Validation Loss (learning rate = {chosen['alpha']})")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean softmax cross-entropy")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "chosen_loss_curves.png", dpi=180)
    plt.close(fig)

    # Task 4.3: accuracy chart for the selected learning rate.
    fig, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(history["epoch"], history["train_accuracy"], linewidth=2, label="Training accuracy")
    axis.plot(history["epoch"], history["validation_accuracy"], linewidth=2, label="Validation accuracy")
    axis.axvline(chosen["best_epoch"], linestyle="--", linewidth=1,
                 label=f"Best epoch: {chosen['best_epoch']}")
    axis.set_title(f"Training and Validation Accuracy (learning rate = {chosen['alpha']})")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.02)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "chosen_accuracy_curves.png", dpi=180)
    plt.close(fig)
    return chosen


def run_training(device_name="auto"):
    device = choose_device(device_name)
    tensors = load_prepared_tensors(device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    network = ECGNet()
    parameter_count = network.parameter_count()
    del network

    print(f"Device: {device}")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Settings: batch={BATCH_SIZE}, epochs={EPOCHS}, momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}")

    all_started = time.perf_counter()
    runs = []
    for alpha in LEARNING_RATES:
        print(f"\nStarting a fresh model with learning rate {alpha}...")
        runs.append(train_one_learning_rate(alpha, tensors, device))

    chosen = _save_plots(runs)
    torch.save(chosen["best_state_dict"], RESULTS_DIR / "best_model.pt")

    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    total_training_seconds = time.perf_counter() - all_started
    public_runs = []
    for result in runs:
        public_runs.append({key: value for key, value in result.items() if key != "best_state_dict"})

    summary = {
        "architecture": [187, 128, 64, 5],
        "trainable_parameters": int(parameter_count),
        "train_rows": int(len(tensors["y_train"])),
        "validation_rows": int(len(tensors["y_val"])),
        "initial_loss": float(runs[0]["initial_train_loss"]),
        "candidate_learning_rates": list(LEARNING_RATES),
        "chosen_learning_rate": float(chosen["alpha"]),
        "best_epoch": int(chosen["best_epoch"]),
        "train_loss_at_best_epoch": float(chosen["train_loss_at_best_epoch"]),
        "validation_loss_at_best_epoch": float(chosen["validation_loss_at_best_epoch"]),
        "train_accuracy_at_best_epoch": float(chosen["train_accuracy_at_best_epoch"]),
        "validation_accuracy_at_best_epoch": float(chosen["validation_accuracy_at_best_epoch"]),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "device": str(device),
        "hardware": hardware,
        "total_training_seconds": float(total_training_seconds),
        "learning_rate_runs": public_runs,
        "checkpoint": "results/best_model.pt",
    }
    gradient_check_path = RESULTS_DIR / "gradient_check.json"
    if gradient_check_path.exists():
        with gradient_check_path.open("r", encoding="utf-8") as file:
            gradient_check = json.load(file)
        summary["gradient_check_W3_max_difference"] = gradient_check["max_absolute_difference_W3"]
        summary["gradient_check_b3_max_difference"] = gradient_check["max_absolute_difference_b3"]

    with (RESULTS_DIR / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # Table 2 from the assignment: selected network/training details.
    table2_rows = [
        ("trainable_parameters", parameter_count),
        ("initial_loss", runs[0]["initial_train_loss"]),
        ("gradient_check_W3_max_difference", summary.get("gradient_check_W3_max_difference", "run src.grad_check.py first")),
        ("gradient_check_b3_max_difference", summary.get("gradient_check_b3_max_difference", "run src.grad_check.py first")),
        ("chosen_learning_rate", chosen["alpha"]),
        ("best_epoch", chosen["best_epoch"]),
        ("train_loss_at_best_epoch", chosen["train_loss_at_best_epoch"]),
        ("validation_loss_at_best_epoch", chosen["validation_loss_at_best_epoch"]),
        ("train_accuracy_at_best_epoch", chosen["train_accuracy_at_best_epoch"]),
        ("validation_accuracy_at_best_epoch", chosen["validation_accuracy_at_best_epoch"]),
        ("training_time_seconds", total_training_seconds),
        ("hardware", hardware),
    ]
    with (RESULTS_DIR / "table2_training_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item", "value"])
        writer.writerows(table2_rows)

    # Table 3 from the assignment: one row per candidate learning rate.
    with (RESULTS_DIR / "learning_rate_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["learning_rate", "lowest_validation_loss", "validation_accuracy_at_that_epoch", "best_epoch"])
        for item in runs:
            writer.writerow([
                item["alpha"], item["best_validation_loss"],
                item["validation_accuracy_at_best_epoch"], item["best_epoch"],
            ])

    print("\nTraining and model selection complete.")
    print(f"Chosen learning rate: {chosen['alpha']}")
    print(f"Best epoch: {chosen['best_epoch']}")
    print(f"Best validation loss: {chosen['best_validation_loss']:.6f}")
    print(f"Checkpoint: {RESULTS_DIR / 'best_model.pt'}")
    print(f"Total time: {total_training_seconds / 60:.2f} minutes on {hardware}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ECGNet model from scratch.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="Use GPU when available, or explicitly select cpu/cuda.")
    args = parser.parse_args()
    run_training(args.device)
