#  Wine Quality Classification 

This repository contains the final project for IT542: Innovative Algorithms. The goal is to analyze and classify wine quality based on physicochemical features using machine learning algorithms.

##  Dataset

- Source: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Files used: `winequality-red.csv` and/or `winequality-white.csv`
- Variables:
  - 11 input features (e.g., alcohol, sulphates, residual sugar)
  - 1 output variable: `quality` (score from 3 to 8)

The dataset contains **1599 samples** with no missing values. A binary classification version of the `quality` variable was created:

- `quality_label`: `{ "good", "bad" }`
- Distribution: 855 good, 744 bad (moderately balanced)

##  Exploratory Data Analysis (EDA)

- Summary statistics and variable types reviewed
- No missing values found
- Duplicate rows identified: 240
- Key variables: `alcohol`, `sulphates`, and `residual sugar` show high variance

##  Data Preprocessing

- Converted multiclass `quality` into binary `quality_label`
- Checked for duplicates and removed as needed
- Ensured balanced dataset for modeling

##  Machine Learning Approach

> *(You can describe here the models used — e.g., Logistic Regression, Random Forest, SVM, etc. If not yet added, you can update this section later.)*

- Data split into training and test sets
- Models evaluated using accuracy, confusion matrix, and classification report

##  Results Summary

> *(Optional: Add accuracy results or visuals if available)*

- Sample result:
  - Accuracy: ~X%
  - F1-score: ~Y%

##  Included Files

- `IT542_Final_Sumeyye_Albayrak.ipynb` – Main Jupyter notebook
- `winequality-red.csv` – Dataset file (if uploaded)
- `README.md` – Project documentation

##  Author

**Sümeyye Albayrak**  
IT542 – Innovative Algorithms (Final Project, 2024)

