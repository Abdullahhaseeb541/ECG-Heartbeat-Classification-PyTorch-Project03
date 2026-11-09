"""Run Tasks 1–5 in order from the project root.

Example: python run_pipeline.py --device cuda
"""
import argparse

from src.prepare import prepare_data
from src.plot_beats import plot_one_per_class
from src.load_data import load_raw
from src.grad_check import run_gradient_check
from src.train import run_training
from src.test_model import run_test


def main():
    parser = argparse.ArgumentParser(description="Run the ECG assignment pipeline.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    print("\n========== TASK 1: DATA PREPARATION ==========")
    prepare_data()
    print("\n========== TASK 1.5: EXAMPLE BEATS ==========")
    plot_one_per_class(load_raw()["train"])
    print("\n========== TASK 3.2: GRADIENT CHECK ==========")
    run_gradient_check()
    print("\n========== TASK 4: TRAINING / MODEL SELECTION ==========")
    run_training(args.device)
    print("\n========== TASK 5: ONE-TIME TEST EVALUATION ==========")
    run_test(args.device)
    print("\nPipeline completed. Check the results/ and communication/ folders.")


if __name__ == "__main__":
    main()
