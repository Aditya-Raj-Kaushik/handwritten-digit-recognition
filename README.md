# Handwritten Digit Recognition – Classical Machine Learning

This repository contains my solution for the **Virutalyyst AI Engineer assignment**, where the goal was to build a handwritten digit recognition system using **only classical machine learning techniques**.

The focus of this project is not just on achieving accuracy, but on demonstrating a clear understanding of the **end-to-end machine learning workflow**, including preprocessing, model design, evaluation, and honest analysis of results.

---

## Problem Statement

The task is to classify handwritten digit images (0–9) from the MNIST dataset provided in **CSV format**.
Each image is a 28×28 grayscale image flattened into 784 pixel values.

**Input:** Pixel values (0–255)
**Output:** Predicted digit label (0–9)

As per the assignment constraints:

* No neural networks or deep learning models were used
* No pre-trained models were used
* Only classical ML algorithms were implemented

---

## Dataset

The dataset is a CSV version of the MNIST dataset downloaded manually from Kaggle.
It contains:

* One label column (`label`)
* 784 pixel columns (`pixel0` to `pixel783`)

The notebook assumes the dataset file is available locally in the working directory.

---

## Approach and Workflow

The project follows a structured and logical machine learning pipeline:

1. Load and inspect the dataset
2. Perform exploratory data analysis

   * Dataset size
   * Class distribution
   * Sample image visualization
3. Normalize pixel values to the range [0, 1]
4. Split data into training and testing sets using stratification
5. Apply PCA (optional) to reduce dimensionality
6. Train and evaluate multiple classical ML models
7. Compare models using accuracy and confusion matrices
8. Analyze misclassified samples
9. Draw conclusions and discuss limitations

A clear execution flow and explanations are included directly in the notebook.

---

## Models Used

### 1. K-Nearest Neighbors (KNN)

* Implemented **from scratch using NumPy**
* Uses Euclidean distance and majority voting
* The value of `k` was selected through limited empirical testing

### 2. Support Vector Machine (SVM)

* RBF kernel
* Hyperparameters (`C` and `gamma`) tuned using a small search space
* Achieved the best individual model performance

### 3. Decision Tree

* Tuned `max_depth` and `min_samples_split`
* Included for interpretability and comparison

### 4. Voting Ensemble (Bonus)

* Hard voting across KNN, SVM, and Decision Tree
* Improved robustness, though it did not outperform the best SVM model

---

## Evaluation

Models are evaluated using:

* Overall accuracy
* Confusion matrices
* Visual inspection of misclassified digits

Misclassification analysis helped identify common errors caused by:

* Similar digit shapes (e.g., 4 vs 9, 5 vs 6)
* Variations in handwriting style
* Incomplete or faint strokes

---

## Results Summary

* **SVM** performed best as an individual model
* **KNN** showed reasonable accuracy but was computationally expensive
* **Decision Tree** was interpretable but prone to overfitting
* **Ensemble learning** improved stability but did not exceed SVM accuracy

The results are discussed honestly without over-optimizing metrics.

---

## Limitations

* Flattened pixel data does not preserve spatial relationships
* Scratch KNN does not scale well for large datasets
* Ensemble uses equal voting instead of weighted voting
* Hyperparameter tuning was empirical, not exhaustive

---

## Future Improvements

If more time or relaxed constraints were available:

* Use weighted or soft voting ensembles
* Apply feature extraction methods such as HOG
* Perform cross-validation for tuning
* Explore shallow neural networks (if permitted)

---

## How to Run

1. Place the MNIST CSV file in the same directory as the notebook
2. Open `Assignment_Aditya_Virutalyyst.ipynb`
3. Run all cells from top to bottom

The PDF file is an exported version of the executed notebook.

---

## Author

**Aditya Raj Kaushik**

---


