# 💰 Adult Census Income Classification

**Name:** Aditya Shukla  

**Registration Number:** 23BAI10155

**Application Number:** IN26011099

**Batch Number:** 1(A)

**Email ID:** aditya.23bai10155@vitbhopal.ac.in 

A machine learning project that predicts whether an individual's annual income exceeds **$50K** based on demographic and employment features from the **Adult Census Income Dataset** (Kaggle).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Author](#author)

---

## 🔍 Overview

Income inequality analysis is a well‑known problem in socio‑economic research. This project builds and compares **five classical supervised‑learning classifiers** to predict whether a person earns more than $50K per year using census data. The end-to-end pipeline covers data acquisition, cleaning, feature engineering, model training, and performance evaluation.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Adult Census Income – Kaggle](https://www.kaggle.com/datasets/priyamchoksi/adult-census-income-dataset) |
| **Records** | ~32,561 |
| **Features** | 14 (age, workclass, education, marital‑status, occupation, etc.) |
| **Target** | `income` — binary label (`>50K` / `<=50K`) |

### Key Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the individual |
| `workclass` | Categorical | Employment type (Private, Self‑emp, Govt, etc.) |
| `education` | Categorical | Highest level of education attained |
| `education.num` | Numeric | Education level (numeric) |
| `marital.status` | Categorical | Marital status |
| `occupation` | Categorical | Type of occupation |
| `relationship` | Categorical | Family relationship role |
| `race` | Categorical | Ethnicity |
| `sex` | Categorical | Gender |
| `capital.gain` | Numeric | Capital gains recorded |
| `capital.loss` | Numeric | Capital losses recorded |
| `hours.per.week` | Numeric | Average working hours per week |
| `native.country` | Categorical | Country of origin |

---

## 🛠️ Project Workflow

```
1. Dataset Understanding
   └─ Load data, inspect dimensions, check class distribution

2. Data Cleaning
   ├─ Strip whitespace from categorical columns
   ├─ Replace '?' markers with NaN
   ├─ Impute missing values using column mode
   └─ Remove duplicate records

3. Feature Engineering
   ├─ Binary-encode the target variable (>50K → 1, <=50K → 0)
   ├─ One-hot encode categorical features (drop_first)
   ├─ Stratified train/test split (80/20)
   └─ Standard scaling of feature set

4. Model Building
   └─ Train five classifiers on scaled training data

5. Performance Evaluation
   └─ Compare models across Accuracy, Precision, Recall, F1, and ROC-AUC
```

---

## 🤖 Models Used

| # | Algorithm | Description |
|---|---|---|
| 1 | **Logistic Regression** | Linear model for binary classification |
| 2 | **Decision Tree** | Tree-based model using feature splits |
| 3 | **Random Forest** | Ensemble of decision trees for improved generalization |
| 4 | **K-Nearest Neighbors** | Instance-based learning using distance metrics |
| 5 | **Support Vector Machine (SVM)** | Finds optimal hyperplane for class separation |

---

## 📈 Results

All models are evaluated on the **test set** using the following metrics:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Ratio of true positives among predicted positives |
| **Recall** | Ratio of true positives among actual positives |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |

> Detailed comparison results are available inside the Jupyter Notebook.

---

## 🧰 Tech Stack

| Tool / Library | Purpose |
|---|---|
| **Python 3** | Programming language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | ML models, preprocessing, metrics |
| **KaggleHub** | Automated dataset download |
| **Jupyter Notebook** | Interactive development environment |

---

## 📁 Project Structure

```
Adult Census Income Classification/
├── Adult Census Income Classification.ipynb   # Main notebook with full pipeline
├── README.md                      # Project documentation (this file)
```