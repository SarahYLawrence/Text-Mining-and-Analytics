---

# ERisk Test Results Evaluation

## Introduction
This repository contains Python scripts for evaluating test results obtained from the ERisk project. The scripts are designed to calculate various error metrics between human-provided and machine-generated answers.

## Requirements
- Python 3.x
- NumPy
- Math module

## Usage
1. Clone the repository to your local machine.
2. Place the human-provided and machine-generated answer data files in the same directory as the scripts.
3. Update the file paths in the scripts to point to your data files.
4. Run the scripts using Python.

## Files
- `Evaluation-Metrics.py`: This script contains functions to calculate various error metrics, including Mean Zero-One Error (MZOE), Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Macroaveraged Mean Absolute Error (MAEmacro).
- `main.py`: This script demonstrates how to use the functions in `Evaluation-Metrics.py` to evaluate test results obtained in different years.
- `Human_answers2022.py`, `Human_answers2023.py`: These files contain the human-provided answers for the years 2022 and 2023, respectively.
- `Machine_answer2022.py`, `Machine_answer2023.py`: These files contain the machine-generated answers for the years 2022 and 2023, respectively.

## Instructions
1. Update the file paths in `Evaluation-Metrics.py` to point to your human-provided and machine-generated answer data files.
2. Run `main.py` using Python.
3. The script will output the evaluation results for each year, including MZOE, MAE, RMSE, and MAEmacro.

## Example Usage
```bash
python main.py
```
