
# NumCompute Stream

NumCompute Stream is a streaming decision tree–based machine learning framework built using plain Python, NumPy, and matplotlib. It extends the original NumCompute package by adding support for incremental learning, decision tree classification, ensemble learning, streaming metrics, preprocessing, benchmarking, and visualisation.

This project was developed for the Programming for AI assignment. External machine learning and data processing libraries such as pandas, scikit-learn, PyTorch, and TensorFlow are not used.

## Features

* Custom CSV loading using `io.py`
* Train/test splitting and streaming chunk creation
* Streaming-compatible preprocessing using `.partial_fit()`
* Decision tree classifier implemented from scratch
* Bagging/random forest style ensemble classifier
* Pipeline system for preprocessing and model training
* Streaming trainer for chunk-wise learning and logging
* Accuracy, precision, recall, F1 score, rolling accuracy, and confusion matrix metrics
* Streaming statistics including mean, variance, min, max, histogram, and quantile support
* Built-in matplotlib visualisation functions
* Unit tests covering normal cases and edge cases
* Benchmarks for model comparison and vectorised computation

## Project Structure

```text
Assignment_2_Numcompute/
│
├── numcompute_stream/
│   ├── __init__.py
│   ├── io.py
│   ├── stream.py
│   ├── tree.py
│   ├── ensemble.py
│   ├── preprocessing.py
│   ├── stats.py
│   ├── metrics.py
│   ├── pipeline.py
│   └── visualise.py
│
├── tests/
│   ├── test_io.py
│   ├── test_stats.py
│   ├── test_metrics.py
│   ├── test_preprocessing.py
│   ├── test_tree.py
│   ├── test_ensemble.py
│   ├── test_pipeline.py
│   └── test_stream.py
│
├── demo/
│   ├── stream_demo.ipynb
│   └── sample_data.csv
│
├── benchmark/
│   ├── benchmark_models.py
│   └── benchmark_vectorisation.py
│
├── README.md
├── requirements.txt
└── report.pdf
```

## Requirements

The project uses only the permitted libraries:

* Python
* NumPy
* matplotlib

For testing and running the notebook demo, the project also uses:

* pytest
* notebook
* ipykernel

Install requirements with:

```bash
pip install -r requirements.txt
```

On Windows PowerShell with a full Python path:

```powershell
& C:\Users\Hp\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pip install -r requirements.txt
```

## Running Tests

To run all unit tests:

```bash
python -m pytest tests -v
```

On Windows PowerShell:

```powershell
& C:\Users\Hp\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest tests -v
```

To run one test file:

```bash
python -m pytest tests/test_ensemble.py -v
```

On Windows PowerShell:

```powershell
& C:\Users\Hp\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest tests/test_ensemble.py -v
```

The test suite covers I/O, preprocessing, streaming statistics, metrics, decision tree, ensemble model, pipeline, and stream trainer behaviour. It also checks edge cases such as missing values, invalid shapes, invalid parameters, prediction before fitting, zero-variance scaling, and streaming updates.

## Running the Demo

Open the notebook:

```text
demo/stream_demo.ipynb
```

In VS Code:

1. Open `demo/stream_demo.ipynb`
2. Select the Python kernel
3. Click `Run All`

The demo performs the following steps:

1. Loads `demo/sample_data.csv` using the custom `load_csv()` function from `io.py`
2. Splits the dataset into training and testing sets
3. Splits the training data into chunks to simulate streaming data
4. Creates a single decision tree pipeline
5. Creates an ensemble pipeline
6. Trains both models incrementally using `.partial_fit()`
7. Logs accuracy after each chunk
8. Visualises accuracy and error over time
9. Shows predictions and final test accuracy

## Running Benchmarks

The `benchmark/` folder contains two benchmark scripts.

### Model Benchmark

This compares a single decision tree with a bagging ensemble under streaming conditions.

Run:

```bash
python benchmark/benchmark_models.py
```

On Windows PowerShell:

```powershell
& C:\Users\Hp\AppData\Local\Python\pythoncore-3.14-64\python.exe benchmark/benchmark_models.py
```

This benchmark reports:

* Model name
* Accuracy
* Training time
* Prediction time

### Vectorisation Benchmark

This compares Python loop-based calculations with NumPy vectorised calculations.

Run:

```bash
python benchmark/benchmark_vectorisation.py
```

On Windows PowerShell:

```powershell
& C:\Users\Hp\AppData\Local\Python\pythoncore-3.14-64\python.exe benchmark/benchmark_vectorisation.py
```

This benchmark reports:

* Python loop time
* NumPy vectorised time
* Correctness checks

## Example Usage

```python
import numpy as np

from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline

X = np.array([
    [1.0, 2.0],
    [1.0, np.nan],
    [5.0, 6.0],
    [6.0, 5.0],
])

y = np.array([0, 0, 1, 1])

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=2)),
])

pipeline.partial_fit(X, y)
predictions = pipeline.predict(X)

print(predictions)
```

## Main Modules

### `io.py`

Handles CSV loading, saving CSV files, train/test splitting, and chunk creation.

### `preprocessing.py`

Includes streaming-compatible preprocessing classes:

* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`

### `tree.py`

Implements a decision tree classifier from scratch using Gini impurity or entropy.

### `ensemble.py`

Implements a bagging/random forest style ensemble classifier using multiple decision trees and majority voting.

### `pipeline.py`

Connects preprocessing steps and the model into one consistent workflow.

### `stream.py`

Provides `StreamTrainer` for chunk-wise model training, metric updates, and logging.

### `metrics.py`

Provides classification metrics including accuracy, precision, recall, F1 score, rolling accuracy, and confusion matrix.

### `stats.py`

Provides streaming statistics including mean, variance, standard deviation, min, max, quantiles, and histograms.

### `visualise.py`

Provides matplotlib plots for:

* Metric history over time
* Model comparison
* Error over time
* Predictions vs ground truth
* Confusion matrix

## Assignment Requirements Covered

This project covers the required assignment components:

* Streaming learning through `.partial_fit()` and `.update()` methods
* Decision tree classifier implemented from scratch
* Ensemble model using multiple decision trees
* Custom preprocessing, statistics, and metrics
* Built-in visualisation using matplotlib
* NumPy vectorised operations
* Numerical stability for NaNs, zero variance, and invalid inputs
* At least 30 unit tests
* Benchmark comparison for base model vs ensemble model
* Benchmark comparison for loop vs vectorised operations
* Demo notebook showing pipeline usage, logs, visualisations, and predictions

## Notes

The current `.partial_fit()` implementation stores previously seen chunks and retrains the model using accumulated data. This is streaming-compatible for demonstration and testing, but a future improvement would be to implement a more advanced true online decision tree algorithm.

## Author

Sadman Sami
Master of Artificial Intelligence and Machine Learning
University of Adelaide
