# 🚀 Remaining Useful Life (RUL) Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting the **Remaining Useful Life (RUL)** of engines using sensor data. The goal is to estimate how many cycles an engine can operate before failure, enabling predictive maintenance.

The project follows an **end-to-end ML pipeline**, starting from exploratory data analysis in Jupyter Notebook to modularized production-ready code with API deployment and experiment tracking.

---

## 📊 Dataset Understanding & Exploration

Initial experimentation was done in a **Jupyter Notebook** to understand the data and derive insights.

### ✔️ Key Steps Performed:

- **Train-Test Analysis**
  - Verified differences between train and test datasets

- **Missing Value Check**
  - Ensured dataset integrity (no null values)

- **Constant Sensor Removal**
  - Used `nunique > 1` instead of `std = 0`
  - Reason: Some sensors had fractional values where std ≠ 0 but were still constant
  - Visualized removed features

- **RUL Calculation**
  - Computed Remaining Useful Life for each engine cycle

---

## 📈 Exploratory Data Analysis (EDA)

### 🔹 Survival Analysis
- Analyzed how long engines typically last
- Found average lifespan ≈ **206 cycles**
- Visualized distribution of engine lifetimes

### 🔹 Lifespan Extremes
- Plotted shortest vs longest engine lifespans  
- Observation: Same unit appearing in both extremes due to cycle-based grouping interpretation

### 🔹 Correlation Analysis
- Generated correlation heatmap to identify important sensors
- Focused on correlation with RUL
- Did not reduce dimensions due to:
  - Limited number of sensors
  - Large dataset size

### 🔹 Trend Visualization
- Selected **top 5 sensors**
- Plotted trends for **every 10th engine unit**
- Applied **Rolling Mean (Moving Average)**:
  - Captures overall trends
  - Reduces noise compared to raw values

---

## ⚙️ Feature Engineering

- Applied **Rolling Mean (RM)** for temporal smoothing
- Applied feature scaling consistently on train and test data

---

## 🤖 Model Training

### 🔹 Baseline Model
- **Linear Regression**

**Performance:**
- R² Score: **0.72**
- RMSE: **21**

### 🔍 Insight
- Model underperformed because:
  - Data is **not linearly related**
  - Sensor relationships with RUL are complex

---

## 🏗️ Modular Pipeline Development

After experimentation, the project was refactored into a **modular ML pipeline**.

### 🔹 Preprocessing Pipeline
Includes:
- Feature engineering (Rolling Mean)
- Scaling
- Data transformations

### 🔹 Training Pipeline
- Applies preprocessing on training data
- Accepts:
  - Model type
  - Hyperparameters (via config/input)
- Trains model
- Evaluates performance
- Saves pipeline as a **pickle file**

### 🔹 Testing Pipeline
- Applies same preprocessing on test data
- Includes validation checks:
  - Column consistency
  - Data shape verification
- Outputs predictions and evaluation metrics

---

## 🔁 Experiment Tracking

Integrated **MLflow** for:
- Tracking experiments
- Logging parameters
- Comparing model performance
- Managing runs efficiently

---

## 🌐 API Deployment

- Built a simple API to serve predictions
- Integrated trained pipeline for real-time inference

---


