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



#  Project 2:🎬 Netflix Content Analysis – Exploratory Data Analysis

## 📌 Project Overview

This project performs an **Exploratory Data Analysis (EDA) on Netflix content** to understand the characteristics and distribution of movies and TV shows available on the platform.

The analysis uses the Netflix titles dataset and explores information such as content type, release year, ratings, countries, directors, cast, genres, duration, and the year content was added to Netflix.

The project also includes **data cleaning, transformation, statistical analysis, and data visualization** to extract meaningful insights from the dataset.

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze the Netflix content catalog.
* Understand the distribution of **Movies and TV Shows**.
* Examine content by **release year**.
* Analyze **ratings and content categories**.
* Explore countries producing Netflix content.
* Identify missing values and perform data cleaning.
* Analyze the year in which content was added to Netflix.
* Create visualizations to better understand the dataset.
* Generate meaningful insights from the available data.

---

## 🛠️ Technologies & Libraries Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **WordCloud** – Text-based visualization
* **Google Colab / Jupyter Notebook**

---

## 📊 Dataset

The project uses the **Netflix Titles Dataset**, containing information about Netflix movies and TV shows.

The initial dataset contains:

* **7,787 records**
* **12 columns**

### Dataset Features

| Column         | Description                                      |
| -------------- | ------------------------------------------------ |
| `show_id`      | Unique ID of the content                         |
| `type`         | Movie or TV Show                                 |
| `title`        | Title of the content                             |
| `director`     | Director of the content                          |
| `cast`         | Cast members                                     |
| `country`      | Country or countries associated with the content |
| `date_added`   | Date the content was added to Netflix            |
| `release_year` | Original release year                            |
| `rating`       | Content rating                                   |
| `duration`     | Movie duration or number of TV show seasons      |
| `listed_in`    | Genre/category of the content                    |
| `description`  | Short description of the content                 |

---

## 🔍 Project Workflow

### 1. Importing Libraries

The required Python libraries are imported for data manipulation, analysis, visualization, and word-cloud generation.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
```

### 2. Data Loading

The Netflix dataset is loaded into a Pandas DataFrame.

```python
netflix_df = pd.read_csv(
    'netflix_titles.csv'
)
```

### 3. Initial Data Inspection

The dataset is examined using:

* `head()`
* `shape`
* `info()`
* `describe()`
* `isnull()`
* `isnull().sum()`

The initial analysis shows that several columns contain missing values, particularly `director`, `cast`, `country`, `date_added`, and `rating`.

### 4. Data Cleaning

Missing values in important text columns are handled during preprocessing.

For example:

```python
netflix_df['director'] = netflix_df['director'].fillna("Unknown")
netflix_df['cast'] = netflix_df['cast'].fillna("unknown")
```

Missing values in `date_added` and `rating` are removed from the dataset.

### 5. Date Transformation

The `date_added` column is converted into a proper datetime format.

```python
netflix_df['date_added'] = pd.to_datetime(
    netflix_df['date_added'],
    format='mixed'
)
```

A new `year_added` column is then created to identify the year in which each title was added to Netflix.

```python
netflix_df['year_added'] = netflix_df['date_added'].dt.year
```

### 6. Exploratory Data Analysis

The cleaned dataset can be analyzed to understand:

* Movie vs TV Show distribution
* Release-year trends
* Netflix content additions over time
* Content ratings
* Popular categories and genres
* Countries represented in the dataset
* Duration of movies and TV shows
* Directors and cast distribution
* Frequently occurring words and descriptions

### 7. Data Visualization

Visualization libraries such as **Matplotlib and Seaborn** are used to represent patterns and trends in the dataset.

Possible visualizations include:

* Bar charts
* Count plots
* Histograms
* Distribution plots
* Year-wise trend charts
* Rating distributions
* Country-wise analysis
* Word clouds

---

## 📈 Key Analysis Areas

### 🎥 Movies vs TV Shows

Analyze the proportion of movies and TV shows available in the Netflix dataset.

### 📅 Release Year Analysis

Study how Netflix content is distributed across different release years and identify periods with increased content production.

### ⭐ Rating Analysis

Explore the different ratings assigned to Netflix content and understand which ratings occur most frequently.

### 🌍 Country Analysis

Analyze the countries associated with Netflix content and identify countries with significant representation.

### 🎭 Genre Analysis

Use the `listed_in` column to explore different genres and categories such as dramas, comedies, documentaries, action, international movies, and TV shows.

### 📆 Netflix Addition Trends

Using the newly created `year_added` column, analyze how Netflix's content catalog changed over the years.

---

## 🧹 Data Cleaning Summary

The project handles missing values and converts date information into a useful format.

| Operation               | Purpose                     |
| ----------------------- | --------------------------- |
| Missing director values | Replaced with `Unknown`     |
| Missing cast values     | Replaced with `unknown`     |
| Missing country values  | Filled during preprocessing |
| Missing date values     | Removed                     |
| Missing rating values   | Removed                     |
| `date_added`            | Converted to datetime       |
| `year_added`            | Created from `date_added`   |

---

## 💡 Skills Demonstrated

This project demonstrates practical experience with:

* Python programming
* Data analysis
* Exploratory Data Analysis (EDA)
* Data cleaning
* Missing-value handling
* Data transformation
* Pandas DataFrames
* Statistical analysis
* Data visualization
* Working with real-world datasets
* Extracting insights from data

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn wordcloud
```

### 3. Open the notebook

Open:

```text
project 2.ipynb
```

using **Jupyter Notebook** or **Google Colab**.

### 4. Run the cells

Execute the notebook cells sequentially to perform the complete Netflix data analysis.

---

## 📁 Project Structure

```text
Netflix-Content-Analysis/
│
├── project 2.ipynb
├── netflix_titles.csv
└── README.md
```

---

## 📌 Conclusion

This project provides an in-depth exploratory analysis of Netflix's content catalog. Through **data cleaning, transformation, statistical analysis, and visualization**, the project helps identify patterns and trends in Netflix movies and TV shows.

It demonstrates how Python-based data analysis techniques can be used to transform a raw dataset into meaningful and understandable insights.

---

## 👨‍💻 Author

**Varad Takale**

Computer Engineering Graduate
Interested in **Data Analytics, Java Development, and Software Engineering**.

# Project 3:🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting house sale prices using **Machine Learning Regression techniques**. The project uses the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

The notebook covers the complete machine learning workflow, including data collection, exploratory data analysis (EDA), data preprocessing, feature preparation, model training, and evaluation.

The main objective is to understand how different features of a house can influence its **Sale Price** and build a model capable of predicting house prices accurately.

---

## 🎯 Objectives

* Analyze the factors affecting house prices.
* Perform exploratory data analysis on the dataset.
* Handle missing values and categorical features.
* Prepare the data for machine learning.
* Train regression models for house price prediction.
* Evaluate model performance using standard regression metrics.
* Compare different machine learning approaches.

---

## 📊 Dataset

The project uses the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

The dataset contains information about residential properties, including features such as:

* Lot Area
* Overall Quality
* Year Built
* Total Basement Area
* Living Area
* Garage Area
* Number of Rooms
* Neighborhood
* Sale Condition
* Sale Type
* And many other property-related features

### Dataset Size

* **Training Dataset:** 1,460 records and 80 columns
* **Testing Dataset:** 1,459 records and 79 columns
* **Target Variable:** `SalePrice`

The dataset is downloaded directly using the **Kaggle API**.

---

## 🔍 Exploratory Data Analysis

The project performs exploratory analysis of the target variable `SalePrice`.

### EDA includes:

* Distribution analysis of house prices
* SalePrice skewness analysis
* Examination of dataset structure
* Understanding numerical and categorical variables
* Visualization of important patterns in the data

EDA helps understand the distribution and characteristics of house prices before applying machine learning models.

---

## ⚙️ Data Preprocessing

The project uses several preprocessing and data preparation techniques, including:

* Loading data using **Pandas**
* Handling numerical and categorical features
* Feature encoding using `LabelEncoder`
* Feature scaling using `StandardScaler`
* Splitting data into training and validation sets
* Preparing features for regression models

---

## 🤖 Machine Learning Models

The project uses regression-based machine learning algorithms, including:

### 1. Linear Regression

Linear Regression is used as a baseline regression model to understand the relationship between input features and house prices.

### 2. XGBoost Regression

**XGBoost** is used as a more powerful gradient boosting model for predicting house prices and capturing complex relationships between different features.

---

## 📏 Model Evaluation

The models are evaluated using common regression metrics:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **R² Score**

These metrics help measure how accurately the models predict house sale prices.

---

## 🛠️ Technologies & Libraries

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* XGBoost

### Tools

* Google Colab
* Kaggle API
* Jupyter Notebook

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── project 3.ipynb
├── train.csv
├── test.csv
├── data_description.txt
├── sample_submission.csv
└── README.md
```

> **Note:** Kaggle API credentials such as `kaggle.json` should not be uploaded to GitHub.

---

## 🔄 Project Workflow

```text
Kaggle Dataset
      ↓
Data Loading
      ↓
Data Exploration
      ↓
Exploratory Data Analysis
      ↓
Data Preprocessing
      ↓
Feature Encoding & Scaling
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Linear Regression / XGBoost
      ↓
Model Evaluation
      ↓
House Price Prediction
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```

### 2. Open the notebook

Open:

```text
project 3.ipynb
```

using **Google Colab** or **Jupyter Notebook**.

### 3. Configure Kaggle API

The notebook uses the Kaggle API to download the dataset.

Upload your own `kaggle.json` file when prompted.

### 4. Run the notebook

Execute the cells sequentially to:

1. Download the dataset
2. Load the training and testing data
3. Perform EDA
4. Preprocess the data
5. Train regression models
6. Evaluate model performance
7. Generate predictions

---

## 📌 Key Learning Outcomes

Through this project, I gained practical experience in:

* Exploratory Data Analysis
* Regression problems
* Data preprocessing
* Feature encoding
* Feature scaling
* Machine learning model training
* Model evaluation
* Working with Kaggle datasets
* Using XGBoost for regression
* Visualizing data using Matplotlib and Seaborn

---

## 🔮 Future Improvements

The project can be further improved by:

* Applying advanced feature engineering
* Handling missing values using more sophisticated techniques
* Applying feature selection
* Using cross-validation
* Performing hyperparameter tuning
* Testing additional regression algorithms
* Using ensemble and stacking techniques
* Improving prediction accuracy

---

## 👨‍💻 Author

**Varad Takale**

Computer Engineering Graduate
Interested in **Java Development, Data Analytics, Machine Learning, and Software Engineering**.

---

## ⭐ Acknowledgements

* Dataset: **Kaggle – House Prices: Advanced Regression Techniques**
* Libraries: Python open-source data science and machine learning ecosystem



