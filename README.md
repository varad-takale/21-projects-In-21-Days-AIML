# 21-projects-In-21-Days-AIML
# Project 1: End-to-End Exploratory Data Analysis (EDA) on the Titanic Dataset

📅 **Day 1 of #21DaysAIMLChallenge**

## 📌 Overview

This project performs a complete, end-to-end Exploratory Data Analysis (EDA) on the classic Titanic dataset. The goal is to understand the passengers aboard the Titanic, uncover patterns in survival, and engineer meaningful features that could later be used for predictive modeling.

## 🎯 Objectives

- Load and inspect the raw Titanic dataset
- Clean missing/inconsistent data
- Explore individual features (univariate analysis)
- Explore relationships between features and survival (bivariate & multivariate analysis)
- Engineer new features to extract additional insight
- Study correlations between numerical features
- Generate an automated data profiling report

## 🛠️ Tech Stack

- **Python**
- **Pandas** & **NumPy** – data manipulation
- **Matplotlib** & **Seaborn** – data visualization
- **ydata-profiling** – automated EDA report generation

## 🔍 Workflow

1. **Setup** – Import required libraries
2. **Data Loading & Inspection** – Load the dataset and examine shape, types, and summary statistics
3. **Data Cleaning** – Handle missing values in `Age`, `Embarked`, and `Cabin`; create a `Has_Cabin` indicator
4. **Univariate Analysis** – Visualize distributions of categorical (`Survived`, `Pclass`, `Sex`, `Embarked`, etc.) and numerical (`Age`, `Fare`) features
5. **Bivariate Analysis** – Examine survival rate against `Pclass`, `Sex`, `Embarked`, and cabin availability
6. **Feature Engineering** – Create `FamilySize`, `IsAlone`, and extract passenger `Title` from names
7. **Multivariate Analysis** – Combine features (e.g., `Pclass` + `Sex`, `Age` + `Sex` + `Survived`) for deeper insights
8. **Correlation Analysis** – Visualize a correlation heatmap of numerical features
9. **Automated Profiling** – Generate a detailed profiling report using `ydata-profiling`

## 📊 Key Insights

- Survival rate was significantly higher for **women** and **1st class passengers**
- Passengers with a **recorded cabin** had a higher survival rate
- **Family size** and being **alone** influenced survival probability
- Passenger **title** (e.g., Mr., Mrs., Miss, Master) correlates strongly with survival outcome

## 📂 Dataset

The dataset used is the well-known [Titanic dataset](https://www.kaggle.com/c/titanic/data), containing passenger details such as age, sex, class, fare, and survival status.

## 🚀 How to Run

1. Clone this repository
2. Open the notebook (`project1.ipynb`) in Jupyter Notebook or Google Colab
3. Run the cells sequentially — the dataset will be fetched automatically
4. Install dependencies if needed:
   ```bash
   pip install pandas numpy matplotlib seaborn ydata-profiling
   ```

## 📁 Output

- Visualizations for univariate, bivariate, and multivariate analysis
- Correlation heatmap
- An HTML profiling report (`sample.html`) summarizing dataset statistics

## 🔗 Challenge Series

This is **Project 1 of 21** in the **21 Days of AI/ML** challenge — a daily series of hands-on projects covering data analysis, machine learning, and deep learning concepts.

