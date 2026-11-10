# Sales Prediction using Python

## Project Overview

This project focuses on predicting product sales based on advertising expenditure across different marketing channels using Machine Learning techniques.

It was developed as part of **Task 4 – Sales Prediction using Python** under the CodeAlpha Data Science Internship.

The project uses advertising data from TV, Radio, and Newspaper channels to analyze their relationship with sales and build predictive regression models.

Two Machine Learning algorithms, Linear Regression and Random Forest Regressor, are implemented and evaluated to identify the better-performing model.

## Objectives

- Understand and analyze the advertising dataset.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Analyze relationships between advertising expenditure and sales.
- Build regression models for sales prediction.
- Compare model performance using evaluation metrics.
- Visualize actual and predicted sales values.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Dataset Description

The project uses the Advertising dataset containing 200 records and 5 columns.

| Column | Description |
|---|---|
| TV | Advertising expenditure on TV |
| Radio | Advertising expenditure on Radio |
| Newspaper | Advertising expenditure on Newspaper |
| Sales | Target variable representing sales |
| Unnamed: 0 | Index column excluded from model training |

The `Unnamed: 0` column was removed because it represents an index rather than a meaningful predictive feature.

The dataset was checked for missing values and duplicate records. No missing values or duplicate rows were identified.

## Project Structure

```text
CodeAlpha_Sales_Prediction/
│
├── Dataset/
│   ├── Advertising_Original.csv
│   └── Advertising_Cleaned.csv
│
├── Notebooks/
│   └── CodeAlpha_Task4_Sales_Prediction.ipynb
│
├── src/
│   ├── 01_Data_Preprocessing.py
│   └── 02_Sales_Prediction_Model.py
│
├── Visualizations/
│   ├── Sales_Distribution.png
│   ├── Advertising_vs_Sales.png
│   ├── Actual_vs_Predicted.png
│   └── Feature_Importance.png
│
├── Reports/
│   └── CodeAlpha_Task4_Sales_Prediction_Report.pdf
│
├── Model_Outputs/
│   ├── Model_Evaluation_Metrics.csv
│   ├── Test_Set_Predictions.csv
│   ├── Feature_Importance.csv
│   └── Model_Summary.txt
│
├── requirements.txt
│
└── README.md
```

## Project Workflow

### 1. Data Loading and Understanding

- Loaded the Advertising dataset using Pandas.
- Examined dataset dimensions and data types.
- Checked missing values and duplicate records.
- Identified the target and independent variables.

### 2. Data Preprocessing

- Removed the unnecessary index column.
- Checked the dataset for data quality issues.
- Prepared the input features and target variable.

### 3. Exploratory Data Analysis

Performed exploratory analysis to understand:

- Distribution of sales values.
- Relationship between TV advertising and sales.
- Relationship between Radio advertising and sales.
- Relationship between Newspaper advertising and sales.

### 4. Model Development

The dataset was divided into training and testing sets using an 80:20 split.

The following algorithms were implemented:

- Linear Regression
- Random Forest Regressor

### 5. Model Evaluation

The models were evaluated using three performance metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors.
- **R² Score:** Measures how much variation in the target variable is explained by the model.

## Model Performance

The following results were obtained using the test dataset:

| Evaluation Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 1.4608 | 0.6210 |
| RMSE | 1.7816 | 0.7487 |
| R² Score | 0.8994 | 0.9822 |

**Selected Model: Random Forest Regressor**

Random Forest Regressor achieved a lower MAE and RMSE and a higher R² score compared with Linear Regression on the selected test split. Therefore, it was selected as the better-performing model for this experiment.

## Visualizations

The project includes the following visualizations:

1. Sales Distribution
2. Advertising Expenditure vs Sales
3. Actual vs Predicted Sales
4. Feature Importance

These visualizations help understand the dataset, model predictions, and the relative contribution of input features to model performance.

## Installation and Execution

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd CodeAlpha_Sales_Prediction
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Preprocessing Script

```bash
python src/01_Data_Preprocessing.py
```

### Step 5: Run the Prediction Model

```bash
python src/02_Sales_Prediction_Model.py
```

Alternatively, open the Jupyter Notebook inside the `Notebooks` directory and execute the cells sequentially.

**Note:** The Python scripts should be run from the project root directory so that the relative dataset paths work correctly. Place the generated output files in the appropriate folders if organizing the project after execution.

## Key Learning Outcomes

- Practical experience with data preprocessing and analysis.
- Understanding relationships between advertising expenditure and sales.
- Hands-on implementation of regression algorithms.
- Comparing Machine Learning models using evaluation metrics.
- Creating meaningful data visualizations.
- Understanding the end-to-end Machine Learning workflow.

## Limitations

- The dataset contains only 200 observations.
- Other factors affecting sales, such as pricing, competition, seasonality, and customer behavior, are not included.
- The model demonstrates patterns in the available dataset and does not establish causal relationships.
- The model should be validated using additional data before real-world deployment.

## Internship Information

**Organization:** CodeAlpha  
**Internship Domain:** Data Science  
**Task Number:** Task 4  
**Project:** Sales Prediction using Python

## Acknowledgement

I would like to thank **CodeAlpha** for providing this opportunity to apply my Python, Data Analysis, and Machine Learning knowledge to a practical project.

---

**Developed as part of the CodeAlpha Data Science Internship.**
