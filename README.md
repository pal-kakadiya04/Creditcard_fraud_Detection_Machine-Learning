# 💳 Credit Card Fraud Detection using Logistic Regression

This project focuses on building a machine learning model to detect fraudulent credit card transactions using **Logistic Regression**. The dataset used is highly imbalanced and includes transactions labeled as normal (`0`) and fraudulent (`1`). We use **under-sampling** to balance the dataset and train an effective model.

## 🔧 Technologies & Libraries
- Python
- NumPy
- Pandas
- Scikit-learn

## 📊 Dataset
The dataset used is sourced from **[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**. It contains transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with only 492 fraudulent transactions out of 284,807 total.

- **0** → Normal Transaction  
- **1** → Fraudulent Transaction

## 📊 Exploratory Data Analysis
- Checked for missing values.
- Examined class distribution: highly imbalanced.
- Separated and analyzed `legit` and `fraud` subsets.
- Compared mean values for each class to identify differences in transaction patterns.

## ⚖️ Data Balancing: Under-Sampling
- Selected 492 random normal transactions to match the 492 fraudulent ones.
- Created a balanced dataset using `pd.concat()`.

## 🧠 Model Training
- Split the balanced data using `train_test_split()` (80% training / 20% testing).
- Trained a **Logistic Regression** model using `sklearn.linear_model`.

## ✅ Model Evaluation
- Calculated **accuracy** on both training and test datasets using `accuracy_score`.

### 🔍 Results:
- Training Accuracy: `~` value (fill after running)
- Test Accuracy: `~` value (fill after running)

## 📌 Key Takeaways
- Class imbalance is a critical issue in fraud detection.
- Under-sampling can help in training balanced models but may reduce generalization.
- Logistic Regression is a simple yet effective baseline model.

## 📎 How to Run
1. Clone the repository.
2. Place the dataset at the defined path.
3. Install dependencies:  