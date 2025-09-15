
# 📈 E-commerce Session Value Prediction: A Datathon 2025 Solution

This repository contains a comprehensive machine learning project developed for the BTK Academy Datathon 2025 competition. The project's goal is to predict the commercial value of e-commerce user sessions (`session_value`). The solution employs advanced techniques in data preprocessing, feature engineering, and a sophisticated ensemble modeling architecture to achieve high prediction accuracy.

## 🎯 Project Objective

The core objective is to accurately predict the potential commercial value of each e-commerce session. These predictions can help companies optimize marketing strategies, personalize the user experience, and identify high-value user segments. The model's performance is evaluated using the **Mean Squared Error (MSE)**, a metric that measures how close the predictions are to the actual values.

## 🛠️ Technical Approach and Architecture

This project is built using a robust methodology that combines fundamental data science steps with advanced techniques.

### 1\. Exploratory Data Analysis (EDA)

A thorough EDA was performed to understand the raw data and gain critical insights for modeling. The process involved analyzing the dataset's structure, the distribution of the target variable (which was found to be right-skewed), and user behavior patterns. The findings from this stage guided the subsequent feature engineering steps.

### 2\. Advanced Feature Engineering

Meaningful and highly predictive features were created from the raw data. These features reflect the dynamics of the sessions and the behavioral tendencies of the users:

  * **Time-Based Features:** Information such as the hour, day of the week, and whether the session occurred on a weekend.
  * **Aggregation Features:** Metrics aggregated at the user and product level, including total sessions, purchases, views, and add-to-cart events.
  * **Behavioral Ratios:** Critical features like `buy_to_view_ratio` and `cart_to_view_ratio` were engineered to directly reflect a session's commercial potential.

### 3\. Stacked Ensemble Modeling

A **Stacked Ensemble** architecture was implemented to maximize prediction performance.

  * **Layer 0 (Base Models):** Powerful and fast gradient boosting models—**LightGBM**, **XGBoost**, and **CatBoost**—were used to learn from the dataset.
  * **Layer 1 (Meta-Model):** A **Ridge Regression** model was used as the meta-model, taking the predictions from the base models as input. It assigned optimal weights to each base model's prediction to produce a final, more accurate result. This approach helps to balance the potential weaknesses of individual models.

The modeling process also included **Optuna** for automated hyperparameter optimization and **Group K-Fold** cross-validation to prevent data leakage and ensure the model's ability to generalize to new users.


## ⚙️ Requirements

To run this project on your local machine, you need to install the following libraries:

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost optuna matplotlib seaborn plotly
```

## 🚀 How to Run the Project

1.  Clone this repository to your local machine:
    `git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git`
2.  Install the required libraries.
3.  Open the Jupyter Notebook in the `notebooks/` folder and follow the steps.

This project demonstrates a robust and scalable machine learning solution for predicting commercial value from user behavior, providing a valuable asset for any e-commerce company.
