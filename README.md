# Air Quality Prediction — Machine Learning Capstone

## 📌 Overview

This project focuses on predicting **Air Quality Index (AQI)** using machine learning techniques. The project uses historical air-quality data containing pollutant measurements and other environmental information.

The project implements multiple **regression and classification algorithms** and compares their performance using standard evaluation metrics.

---

## 🎯 Problem Statement

Air pollution is a major environmental concern, and monitoring air quality helps identify potentially harmful conditions.

The objective of this project is to develop machine learning models that can:

* Predict the numerical **AQI value**.
* Classify air quality into different **AQI categories**.
* Compare multiple machine learning algorithms.
* Identify the models that perform well on the given dataset.

---

## 📂 Dataset

The project uses the **City Day** air-quality dataset:

**File:** `city_day.csv`

The dataset contains approximately **29,531 records and 16 columns**, including pollutant measurements such as:

* PM2.5
* PM10
* NO
* NO2
* NOx
* NH3
* CO
* SO2
* O3
* Benzene
* Toluene
* Xylene

The dataset also contains:

* `City`
* `Date`
* `AQI`
* `AQI_Bucket`

---

## 📊 Exploratory Data Analysis

The project performs several EDA steps to understand the dataset, including:

* Dataset structure and summary
* Missing-value analysis
* Duplicate-value checking
* Outlier inspection
* Feature distributions
* AQI distribution
* AQI category distribution
* Correlation heatmap
* Scatter plots between pollutant measurements and AQI

Observations are provided alongside major visualizations to explain the patterns found in the data.

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

* Handling missing values using appropriate imputation techniques
* Checking and treating outliers
* Encoding categorical features
* Scaling numerical features
* Removing unnecessary columns
* Separating features and target variables
* Performing an 80:20 train-test split

For classification, a **stratified split** is used to maintain the distribution of AQI categories.

Preprocessing transformations are fitted only on the training data to prevent **data leakage**.

---

## ⚙️ Feature Engineering

The `Date` column is converted into a datetime format and additional temporal features are extracted:

* Year
* Month
* Day
* Day of Week

The original `Date` column is then removed from the modeling features.

Additional engineered features can be incorporated as the project develops.

---

## 📈 Model Evaluation

The performance of all models is summarized using comparison tables.

For regression, the models are compared based on:

| Metric | Description                                                 |
| ------ | ----------------------------------------------------------- |
| R²     | Measures how much variance in AQI is explained by the model |
| RMSE   | Measures the square-rooted average squared prediction error |
| MAE    | Measures the average absolute prediction error              |

For classification, the models are compared using:

| Metric           | Description                                                       |
| ---------------- | ----------------------------------------------------------------- |
| Accuracy         | Percentage of correctly classified observations                   |
| Precision        | Correct positive predictions relative to all positive predictions |
| Recall           | Correct positive predictions relative to all actual positives     |
| Weighted F1      | Weighted combination of precision and recall                      |
| Confusion Matrix | Shows correct and incorrect predictions by class                  |

---

## 📉 Visualizations

The notebook includes visualizations such as:

* Feature distribution plots
* AQI distribution
* AQI category distribution
* Correlation heatmap
* Pollutant vs AQI scatter plots
* Predicted vs Actual AQI
* Regression residual plot
* Random Forest feature importance
* Classification confusion matrices

These visualizations are used to interpret both the dataset and model performance.

##

---

## 👥 Team

**Team Members:**

* Tharun G Nambiar
* Adhitya Ranjit
* Adwit Singh

---

## 📌 Conclusion

This project provides a comparative study of multiple machine learning approaches for air-quality prediction. By combining exploratory data analysis, preprocessing, feature engineering, regression, and classification, the project aims to identify useful patterns in air-quality data and evaluate how different machine learning algorithms perform on the same dataset.
