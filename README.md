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

#  Project 4:🩺 Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting the presence and severity of heart disease using **Machine Learning Classification techniques**.

The project uses the **Heart Disease UCI dataset** and follows a complete machine learning workflow, including data loading, exploratory data analysis (EDA), data preprocessing, feature transformation, model training, and model evaluation.

Multiple classification algorithms are implemented and compared to understand their performance on the heart disease dataset.

---

## 🎯 Objectives

* Analyze the heart disease dataset.
* Perform Exploratory Data Analysis (EDA).
* Identify relationships between important health-related features and heart disease.
* Handle missing values and categorical variables.
* Transform numerical and categorical features.
* Train multiple classification models.
* Evaluate and compare model performance.
* Analyze predictions using a confusion matrix.
* Understand feature importance.

---

## 📊 Dataset

The project uses the **Heart Disease UCI dataset**, downloaded using `kagglehub`.

The target variable is:

```text
num
```

The target represents the heart disease classification.

The dataset contains medical and demographic features such as:

* Age
* Sex
* Chest Pain Type (`cp`)
* Resting Blood Pressure (`trestbps`)
* Cholesterol (`chol`)
* Fasting Blood Sugar (`fbs`)
* Resting ECG (`restecg`)
* Maximum Heart Rate (`thalach`)
* Exercise-Induced Angina (`exang`)
* ST Depression (`oldpeak`)
* Slope
* Number of Major Vessels (`ca`)
* Thalassemia (`thal`)

---

## 🔍 Exploratory Data Analysis

The project performs EDA to understand the dataset and identify patterns related to heart disease.

### EDA includes:

* Dataset structure and information
* Descriptive statistics
* Missing-value analysis
* Target variable distribution
* Age distribution by target
* Maximum heart rate comparison
* Chest pain type analysis
* Sex distribution by target
* Correlation analysis using a heatmap

These visualizations help identify relationships between patient characteristics and heart disease outcomes.

---

## ⚙️ Data Preprocessing

The following preprocessing techniques are implemented:

### Numerical Features

* Missing values are handled using **mean imputation**.
* Numerical features are standardized using **StandardScaler**.

### Categorical Features

* Missing categorical values are handled where required.
* Categorical variables are converted using **OneHotEncoder**.
* `drop='first'` is used to avoid redundant encoded columns.
* `handle_unknown='ig_



# project 5 : 🛍️ Customer Segmentation with Clustering

## 📌 Project Overview

This project focuses on **Customer Segmentation using Machine Learning clustering techniques**. The objective is to analyze customer characteristics and identify meaningful groups of customers based on their **age, annual income, and spending score**.

Customer segmentation can help businesses understand different customer behaviors and create more targeted marketing strategies, personalized offers, and better customer experiences.

The project uses the **Mall Customers dataset**, containing information about 200 customers, and applies Exploratory Data Analysis (EDA), data preprocessing, visualization, and clustering techniques.

---

## 🎯 Objectives

* Analyze customer demographic and spending information.
* Perform Exploratory Data Analysis (EDA).
* Understand the distribution of customer features.
* Identify relationships between income and spending behavior.
* Preprocess and scale numerical features.
* Apply clustering techniques to segment customers.
* Visualize and interpret different customer groups.

---

## 📊 Dataset

The project uses the **Mall Customers dataset**.

### Dataset Features

| Feature                  | Description                           |
| ------------------------ | ------------------------------------- |
| `CustomerID`             | Unique customer identifier            |
| `Gender`                 | Customer gender                       |
| `Age`                    | Customer age                          |
| `Annual Income (k$)`     | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Customer spending score               |

### Dataset Size

* **Rows:** 200
* **Columns:** 5
* **Age Range:** 18–70
* **Annual Income Range:** $15k–$137k
* **Spending Score Range:** 1–99

---

## 🔍 Exploratory Data Analysis

The project performs an in-depth EDA to understand the dataset.

### EDA includes:

* Dataset information and data types
* Descriptive statistical analysis
* Feature distributions
* Age distribution analysis
* Annual income distribution
* Spending score distribution
* Gender-based feature visualization
* Relationship analysis between customer attributes

`CustomerID` is removed before analysis because it is an identifier and does not provide useful information for customer clustering.

---

## 🤖 Machine Learning Techniques

The project uses clustering techniques to discover groups of customers without predefined labels.

### K-Means Clustering

**K-Means Clustering** is used to divide customers into groups based on similarities between their features.

The project uses:

* `KMeans`
* `StandardScaler`

Feature scaling is performed so that variables with different numerical ranges do not disproportionately influence the clustering algorithm.

### Hierarchical Clustering

The project also uses hierarchical clustering tools for analyzing the structure and relationships between customer groups.

---

## 🛠️ Technologies & Libraries

The project is developed in **Python** using Google Colab/Jupyter Notebook.

### Libraries Used

* 🐼 Pandas
* 🔢 NumPy
* 📊 Matplotlib
* 🎨 Seaborn
* 📈 Plotly
* 🌳 SciPy
* 🤖 Scikit-learn

---

## 📁 Project Structure

```text
Customer-Segmentation/
│
├── Project 5.ipynb
├── Mall_Customers.csv
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Open the Notebook

Open:

```text
Project 5.ipynb
```

using **Jupyter Notebook** or **Google Colab**.

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn
```

### 4. Run the Notebook

Execute the cells sequentially to perform:

```text
Data Loading
     ↓
Data Exploration
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Scaling
     ↓
Clustering
     ↓
Visualization
     ↓
Customer Segmentation
```

---

## 💡 Key Learning Outcomes

Through this project, I explored:

* Data loading and preprocessing using Pandas
* Exploratory Data Analysis
* Statistical analysis
* Data visualization
* Feature scaling
* Unsupervised Machine Learning
* K-Means clustering
* Hierarchical clustering
* Customer segmentation
* Interpretation of clustering results

---

## 📌 Applications

Customer segmentation can be useful for:

* 🎯 Targeted marketing
* 🛒 Personalized product recommendations
* 💰 Customer value analysis
* 📢 Marketing campaign optimization
* 🤝 Customer relationship management
* 📊 Business decision-making

---

## 🔮 Future Improvements

Possible improvements to this project include:

* Experimenting with different clustering algorithms.
* Comparing K-Means with hierarchical clustering results.
* Using additional customer features.
* Performing cluster profiling in greater detail.
* Developing an interactive customer segmentation dashboard.
* Deploying the model as a web application.

---




# Project 6: Time Series Analysis & Forecasting 📈

## 📌 Project Overview

This project focuses on **Time Series Analysis and Forecasting** using monthly airline passenger data from **1949 to 1960**.

The project analyzes historical passenger trends, checks the stationarity of the time series, applies transformations to make the data stationary, and builds **ARIMA and SARIMA models** to forecast future airline passenger numbers.

The project particularly demonstrates how **seasonality and trends** can be handled using time-series forecasting techniques.

---

## 🎯 Objectives

* Analyze monthly airline passenger data.
* Explore trends and seasonal patterns.
* Decompose the time series into its components.
* Test the series for stationarity using the **Augmented Dickey-Fuller (ADF) test**.
* Transform the data using logarithmic transformation and differencing.
* Analyze **ACF and PACF** plots for model identification.
* Build an **ARIMA** forecasting model.
* Build a **SARIMA** model to handle seasonality.
* Evaluate the SARIMA model using **RMSE (Root Mean Squared Error)**.
* Compare forecasted values with actual passenger numbers.

---

## 📊 Dataset

The project uses an airline passenger time-series dataset containing:

* **Time Period:** 1949–1960
* **Frequency:** Monthly
* **Rows:** 144
* **Columns:** 2

### Columns

| Column       | Description                   |
| ------------ | ----------------------------- |
| `Month`      | Month and year of observation |
| `Passengers` | Number of airline passengers  |

The dataset is loaded from the project's GitHub dataset repository.

---

## 🛠️ Technologies & Libraries

The project is implemented in **Python** using:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Statsmodels
* Scikit-learn

### Main techniques used

* Time Series Decomposition
* Augmented Dickey-Fuller (ADF) Test
* Log Transformation
* Differencing
* ACF Analysis
* PACF Analysis
* ARIMA
* SARIMA
* RMSE Evaluation

---

## 🔍 Project Workflow

### 1. Data Loading and Setup

The required Python libraries are imported and the airline passenger dataset is loaded into a Pandas DataFrame.

The dataset contains monthly passenger observations.

### 2. Exploratory Data Analysis

The passenger data is visualized to understand its overall behavior.

The analysis helps identify:

* Long-term trends
* Increasing passenger numbers
* Repeating seasonal patterns
* Changes in passenger volume over time

The time series is also decomposed using **multiplicative seasonal decomposition**.

### 3. Stationarity Testing

The **Augmented Dickey-Fuller (ADF) test** is applied to determine whether the original time series is stationary.

Stationarity is important because ARIMA-based models generally work with stationary series or require appropriate differencing.

### 4. Making the Series Stationary

A logarithmic transformation is applied:

```python
df_log = np.log(df['Passengers'])
```

The transformed series is then differenced:

```python
df_diff = df_log.diff().dropna()
```

The ADF test is performed again on the transformed and differenced series.

### 5. ACF and PACF Analysis

**Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots are generated using the stationary series.

These plots help identify appropriate parameters for the ARIMA model.

### 6. ARIMA Model

An ARIMA model with the following configuration is created:

```text
ARIMA(1, 1, 1)
```

The dataset is divided into:

* **Training data:** Up to 1958
* **Testing data:** 1959 onward

The model is then used to forecast the testing period.

### 7. SARIMA Model

Because the dataset contains clear seasonal behavior, a seasonal ARIMA model is built:

```text
SARIMA(1, 1, 1)(1, 1, 1, 12)
```

The seasonal period is **12 months**, representing yearly seasonality in monthly passenger data.

### 8. Model Evaluation

The SARIMA predictions are converted back from logarithmic scale using the exponential function.

The model is evaluated using **Root Mean Squared Error (RMSE)**:

```python
rmse = np.sqrt(mean_squared_error(original_test_data, sarima_predictions))
```

A final graph compares the actual passenger numbers with the SARIMA forecast.

---

## 📈 Model

### SARIMA Configuration

```text
Non-seasonal order: (1, 1, 1)
Seasonal order:     (1, 1, 1, 12)
```

Where:

* `p = 1` → Autoregressive component
* `d = 1` → Differencing
* `q = 1` → Moving average component
* `P = 1` → Seasonal autoregressive component
* `D = 1` → Seasonal differencing
* `Q = 1` → Seasonal moving average component
* `12` → Monthly seasonal period

---

## 📊 Evaluation Metric

### RMSE — Root Mean Squared Error

RMSE measures the difference between the actual passenger values and the forecasted values.

A lower RMSE indicates that the predictions are closer to the actual observations.

The notebook calculates the final SARIMA RMSE and displays it in the output.

---

## 📁 Project Structure

```text
Project-6/
│
├── project 6.ipynb
├── README.md
└── dataset1/
    └── airline_passenger_timeseries.csv
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Project-6
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

### 3. Open the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
project 6.ipynb
```

### 4. Run all cells

Run the notebook cells sequentially to perform the complete analysis and generate the forecasts and visualizations.

---

## 🔮 Results

The project successfully demonstrates a complete time-series forecasting workflow:

* Historical airline passenger trends are visualized.
* Seasonal patterns are identified through decomposition.
* Stationarity is tested using the ADF test.
* Log transformation and differencing are applied.
* ACF and PACF plots are used for model identification.
* ARIMA and SARIMA forecasting models are constructed.
* SARIMA is used to account for the **12-month seasonal pattern**.
* Forecasted values are compared against actual passenger numbers.
* RMSE is calculated to evaluate forecasting performance.

---

## 💡 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

1. Understanding time-series data.
2. Identifying trends and seasonality.
3. Testing stationarity.
4. Applying logarithmic transformation and differencing.
5. Understanding ACF and PACF.
6. Building ARIMA models.
7. Handling seasonality using SARIMA.
8. Splitting time-series data into training and testing sets.
9. Evaluating forecasting models using RMSE.
10. Visualizing actual vs. predicted values.

---

# project 7 : Customer Churn Prediction Using Feature Engineering

A machine learning project that analyzes customer churn and evaluates how **feature engineering** can improve customer churn prediction.

## 📌 Project Overview

Customer churn is an important business problem where companies try to identify customers who are likely to stop using their services.

In this project, the **Telco Customer Churn** dataset is used to build a baseline Logistic Regression model and then improve the feature representation through feature engineering.

The project compares model performance **before and after feature engineering** and uses a Random Forest model to identify important features.

---

## 🎯 Objectives

* Load and understand the Telco Customer Churn dataset.
* Clean and prepare the data for machine learning.
* Build a baseline churn prediction model.
* Perform feature engineering on customer information.
* Train an enhanced Logistic Regression model.
* Compare baseline and enhanced model performance.
* Identify important features using Random Forest.

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

* **Rows:** 7,043
* **Columns:** 21
* **Target variable:** `Churn`

The dataset contains customer information such as:

* Customer demographics
* Tenure
* Phone services
* Internet services
* Online security
* Technical support
* Streaming services
* Contract type
* Payment method
* Monthly charges
* Total charges
* Churn status

### Target Distribution

| Churn | Customers |
| ----- | --------: |
| No    |     5,174 |
| Yes   |     1,869 |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab / Jupyter Notebook

### Machine Learning Techniques

* Logistic Regression
* Random Forest Classifier
* StandardScaler
* OneHotEncoder
* ColumnTransformer
* Pipeline
* Train-Test Split

---

## 🧹 Data Cleaning

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Converted `TotalCharges` from object/string format to numeric.
3. Identified **11 missing values** in `TotalCharges`.
4. Replaced missing `TotalCharges` values with the median.
5. Converted the `Churn` column into a binary variable:

   * `Yes → 1`
   * `No → 0`

---

## ⚙️ Feature Engineering

Three main features were created to provide additional information to the model.

### 1. Tenure Group

The original `tenure` feature was divided into customer tenure groups:

* `0-1 Year`
* `1-2 Years`
* `2-4 Years`
* `4-5 Years`
* `5+ Years`

### 2. Number of Additional Services

A new feature called `num_add_services` counts the number of additional services used by each customer.

The services considered include:

* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### 3. Monthly Charge Ratio

A `monthly_charge_ratio` feature was created using:

```text
MonthlyCharges / (tenure + 1)
```

The `+1` prevents division by zero for customers with zero tenure.

### Additional Data Transformation

The following categories were simplified:

* `No phone service → No`
* `No internet service → No`

---

## 🤖 Machine Learning Models

### Model 1 — Baseline Logistic Regression

The first model was trained using the original dataset without the newly engineered features.

Preprocessing included:

* Standard scaling for numerical features
* One-hot encoding for categorical features
* Logistic Regression classifier

### Baseline Performance

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| 0     |      0.85 |   0.89 |     0.87 |
| 1     |      0.65 |   0.56 |     0.60 |

**Accuracy: 80%**

---

### Model 2 — Logistic Regression with Feature Engineering

The second Logistic Regression model used the engineered features.

### Enhanced Performance

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| 0     |      0.84 |   0.91 |     0.87 |
| 1     |      0.68 |   0.52 |     0.59 |

**Accuracy: 81%**

---

## 📈 Model Comparison

| Model                                  | Accuracy |
| -------------------------------------- | -------: |
| Baseline Logistic Regression           |      80% |
| Feature-Engineered Logistic Regression |      81% |

The feature-engineered model achieved a **1 percentage-point improvement in accuracy** over the baseline model.

However, the recall for the churn class decreased slightly, showing that accuracy alone is not sufficient for evaluating churn prediction.

---

## 🌲 Feature Importance

A **Random Forest Classifier** was also trained using the engineered dataset.

Feature importance was extracted from the Random Forest model to identify the **Top 15 most important features** influencing churn prediction.

The project visualizes these features using a horizontal bar chart.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Preparation
   ↓
Baseline Model
   ↓
Feature Engineering
   ↓
Enhanced Model
   ↓
Model Comparison
   ↓
Random Forest Feature Importance
   ↓
Final Conclusion
```

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── project 7.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

> The dataset is loaded in the notebook from the `dataset1` GitHub repository.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Open the notebook

Open:

```text
project 7.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

### 4. Run all cells

Execute the notebook from top to bottom to reproduce the data cleaning, feature engineering, model training, evaluation, and feature-importance analysis.

---

## 💡 Key Takeaways

* Feature engineering can improve the representation of customer behavior for machine learning.
* The feature-engineered Logistic Regression model improved overall accuracy from **80% to 81%**.
* Churn prediction should not be evaluated using accuracy alone because the churn and non-churn classes are imbalanced.
* Random Forest feature importance provides a way to investigate which features contribute most to predictions.

---

## 🚀 Future Improvements

Possible improvements to this project include:

* Hyperparameter tuning
* Cross-validation
* Testing additional classification algorithms
* Handling class imbalance using suitable techniques
* ROC-AUC and Precision-Recall analysis
* Threshold optimization
* Deploying the trained model as a web application

---

# project 8 - 👕 Fashion-MNIST Image Classification using Deep Learning

A deep learning project that classifies Fashion-MNIST images into 10 different clothing categories using **Artificial Neural Network (ANN)** and **Convolutional Neural Network (CNN)** models.

The project implements and compares three architectures:

* Basic ANN
* Basic CNN
* Deeper CNN with Batch Normalization and Dropout

## 📌 Project Overview

The **Fashion-MNIST dataset** contains grayscale images of fashion items. The objective of this project is to build deep learning models that can automatically recognize and classify these images into their respective categories.

The project covers the complete workflow:

**Dataset Loading → Data Preprocessing → Model Building → Model Training → Model Evaluation → Model Comparison**

## 📊 Dataset

The project uses the **Fashion-MNIST dataset** available through TensorFlow/Keras.

Dataset details:

* Training images: **60,000**
* Testing images: **10,000**
* Image size: **28 × 28 pixels**
* Image type: **Grayscale**
* Number of classes: **10**

The images are normalized by dividing pixel values by `255.0`.

The images are reshaped into:

```text
(28, 28, 1)
```

Labels are converted into one-hot encoded vectors containing 10 classes.

## 🏷️ Fashion-MNIST Classes

The 10 classes are:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* TensorFlow
* Keras
* Matplotlib
* Plotly
* Scikit-learn
* Seaborn

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Normalization

Pixel values are converted from the range `0–255` to `0–1`.

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

### 2. Reshaping

Images are reshaped to include a single grayscale channel:

```python
(28, 28, 1)
```

### 3. One-Hot Encoding

The class labels are converted into categorical vectors using:

```python
keras.utils.to_categorical()
```

## 🧠 Models

### 1. Basic ANN

The ANN architecture consists of:

```text
Input Image
    ↓
Flatten
    ↓
Dense(128, ReLU)
    ↓
Dense(64, ReLU)
    ↓
Dense(10, Softmax)
```

Total parameters:

**109,386**

The model uses:

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Metric: Accuracy

---

### 2. Basic CNN

The Basic CNN architecture consists of convolutional and pooling layers:

```text
Input Image
    ↓
Conv2D(32)
    ↓
MaxPooling2D
    ↓
Conv2D(64)
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense(64)
    ↓
Dense(10, Softmax)
```

Total parameters:

**121,930**

The model uses:

* ReLU activation for hidden layers
* Softmax activation for classification
* Adam optimizer
* Categorical Crossentropy loss

---

### 3. Deeper CNN

The deeper CNN adds additional convolutional layers along with **Batch Normalization** and **Dropout** to improve training and reduce overfitting.

```text
Input Image
    ↓
Conv2D(32)
    ↓
Batch Normalization
    ↓
MaxPooling
    ↓
Dropout
    ↓
Conv2D(64)
    ↓
Batch Normalization
    ↓
MaxPooling
    ↓
Dropout
    ↓
Conv2D(128)
    ↓
Batch Normalization
    ↓
MaxPooling
    ↓
Dropout
    ↓
Flatten
    ↓
Dense(128)
    ↓
Batch Normalization
    ↓
Dropout
    ↓
Dense(10, Softmax)
```

Total parameters:

**111,882**

Trainable parameters:

**111,178**

Non-trainable parameters:

**704**

## 🏋️ Model Training

Each model is trained with:

* Maximum epochs: **30**
* Batch size: **64**
* Optimizer: **Adam**
* Loss function: **Categorical Crossentropy**

The project uses:

### Early Stopping

Training monitors validation loss and stops when the validation performance no longer improves.

```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

### Model Checkpoint

The best model weights are saved based on validation loss.

Saved weight files include:

```text
best_ann_model_weights.weights.h5
best_basic_cnn_model_weights.weights.h5
best_deeper_cnn_model_weights.weights.h5
```

## 📈 Model Evaluation

The models are evaluated using the Fashion-MNIST test dataset.

The project includes:

* Test loss
* Test accuracy
* Confusion matrices
* Training vs validation accuracy
* Training vs validation loss
* Model performance comparison

Confusion matrices are generated using:

```python
from sklearn.metrics import confusion_matrix
```

Predictions are converted into class labels using `argmax()`.

## 📊 Model Comparison

The notebook compares the three models based on:

| Model      | Architecture                   | Parameters |
| ---------- | ------------------------------ | ---------: |
| ANN        | Fully Connected Neural Network |    109,386 |
| Basic CNN  | 2 Convolutional Blocks         |    121,930 |
| Deeper CNN | 3 Convolutional Blocks +       |            |









