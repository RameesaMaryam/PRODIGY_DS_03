# Bank Marketing Campaign Classification

This project aims to build a machine learning model to classify the success of a bank marketing campaign. The dataset used is the "Bank Marketing" dataset from the UCI Machine Learning Repository.

## Steps to Execute the Code

### Step 1: Load Libraries
The necessary libraries for this project include:
- `pandas` and `numpy` for data manipulation and analysis.
- `sklearn` for machine learning model creation and evaluation.
- `seaborn` and `matplotlib` for data visualization.

### Step 2: Load Data
The dataset is loaded from the UCI Machine Learning Repository. It contains data related to direct marketing campaigns of a Portuguese banking institution.

### Step 3: Data Preprocessing
Data preprocessing steps include:
- Inspecting the first few rows of the dataset to understand its structure.
- Checking for missing values and handling them by dropping rows with missing values.
- Converting categorical variables to dummy variables for use in machine learning models.
- Separating features and the target variable. The target variable is whether the client subscribed to a term deposit (`y_yes`).

### Step 4: Train-Test Split
The data is split into training and testing sets with a 70-30 ratio using `train_test_split` from `sklearn`.

### Step 5: Model Training
A Decision Tree Classifier is trained using the training data.

### Step 6: Model Evaluation
The trained model is evaluated on the test data. Evaluation metrics include:
- Classification report which provides precision, recall, and F1-score for each class.
- Accuracy score which gives the overall accuracy of the model.
- Confusion matrix which shows the number of true positive, true negative, false positive, and false negative predictions.

A confusion matrix is also plotted using `seaborn` for better visualization of the model performance.

## Dependencies
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

Make sure to install these libraries using pip:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Dataset
The dataset can be found at the UCI Machine Learning Repository: [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## Conclusion
This project demonstrates how to build a machine learning model to classify the success of a bank marketing campaign using a Decision Tree Classifier. It includes steps for data loading, preprocessing, model training, and evaluation.
