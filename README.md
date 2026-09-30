# Industrial Machine Failure Prediction Using Machine Learning

## LearnDepth Track 1 – Final Capstone | Problem 21

### Project Overview

Industrial machines can experience failures due to factors such as temperature, rotational speed, torque, and tool wear. Predicting machine failure in advance can help in taking preventive actions and reducing unexpected downtime.

This project develops a machine learning-based system to predict whether an industrial machine is likely to experience a failure based on its operating conditions.

The project follows an end-to-end machine learning workflow, starting from data preprocessing and exploratory data analysis to model development, evaluation, and deployment using Streamlit.

---

## Problem Statement

Predict whether a machine is likely to experience a failure based on its operating parameters.

---

## Project Objectives

- Analyze the industrial machine dataset.
- Perform data preprocessing and exploratory data analysis.
- Identify relevant features for machine failure prediction.
- Develop classification models for predicting machine failure.
- Handle the class imbalance in the dataset.
- Compare different machine learning models using suitable evaluation metrics.
- Develop a simple Streamlit application for making predictions.

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

### Dataset Size

- Rows: 10,000
- Columns: 14
- Target variable: `Machine failure`

### Important Features

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

### Target

`Machine failure`

- `0` – No Failure
- `1` – Failure

The dataset contains significantly more no-failure samples than failure samples, so class imbalance was considered during model development.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked the dataset shape and data types.
3. Checked for missing values.
4. Converted the categorical `Type` feature into numerical form using one-hot encoding.
5. Removed identifier columns such as:
   - `UDI`
   - `Product ID`
6. Selected relevant operating features for prediction.
7. Split the dataset into training and testing sets using an 80:20 ratio.
8. Used stratified splitting to preserve the target class distribution.
9. Applied feature scaling for models that require scaled inputs.

---

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the dataset and identify relationships between machine operating parameters and machine failures.

The analysis included:

- Class distribution analysis
- Feature statistics
- Mean comparison between failure and no-failure cases
- Correlation analysis
- Data visualization
- Feature relationship analysis

The target dataset was found to be imbalanced, with failure cases representing a small portion of the total observations.

---

## Machine Learning Models

The following classification models were developed and evaluated:

### 1. Logistic Regression

Used as a baseline classification model for predicting machine failure.

### 2. K-Nearest Neighbors (KNN)

Used to classify machine conditions based on the similarity between observations.

### 3. Decision Tree

Used to model relationships between machine operating parameters and failure outcomes.

### 4. Balanced Logistic Regression

A class-weighted version of Logistic Regression was developed to give more importance to the minority failure class.

### 5. Balanced Decision Tree

A class-weighted Decision Tree was also developed to address the class imbalance.

---

## Handling Class Imbalance

The dataset contains many more non-failure cases than failure cases.

To address this issue, class weighting was applied to selected models using:

```python
class_weight='balanced'
