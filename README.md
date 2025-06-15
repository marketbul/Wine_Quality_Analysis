# 🍷 Wine Quality Analysis & Prediction App

An interactive Streamlit application that performs *Exploratory Data Analysis (EDA)* and builds a *machine learning model* to predict the quality of red wine based on its physicochemical properties.

[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-%23FF4B4B?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/Machine%20Learning-scikit--learn-%23F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📌 Project Description

This project explores the *Red Wine Quality Dataset* from the UCI Machine Learning Repository. The dataset contains chemical and physical characteristics of different red wine samples, along with quality ratings assigned by wine tasters.

### 🎯 Goals:
- Perform in-depth *EDA* to identify patterns, correlations, and outliers.
- Visualize distributions, relationships, and aggregated statistics.
- Build and evaluate a *Random Forest Regression model*.
- Enable users to *predict wine quality* by inputting feature values in real-time via sliders.

---

## 🔍 Features

- 📊 Summary statistics, correlation heatmaps, scatter, violin, and box plots
- 🧠 Machine Learning using RandomForestRegressor with model evaluation metrics (MSE, R²)
- 🚀 Interactive user input with *quality prediction*
- 🎨 Modern UI with custom styling for improved user experience
- 🔍 Outlier detection using Z-score method

---

## 🗂 Dataset

- *Source*: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- *File Used*: winequality-red.csv
- *Attributes*:
  - Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, etc.
  - Target variable: quality (score from 0 to 10)

---

## 🛠 Tech Stack

- *Frontend/UI*: Streamlit
- *EDA*: Pandas, NumPy, Seaborn, Matplotlib
- *ML Model*: Scikit-learn (RandomForestRegressor)
- *Preprocessing*: StandardScaler, Train/Test Split

---

