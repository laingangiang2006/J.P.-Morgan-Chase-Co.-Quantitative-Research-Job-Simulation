from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import numpy as np
import pandas as pd

# Read in loan data from a CSV file
df = pd.read_csv('/Users/laingangiang/Downloads/Task 3/Task 3 and 4_Loan_Data.csv')

# Define the variable features
features = ['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']

# Calculate the payment_to_income ratio
df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']
    
# Calculate the debt_to_income ratio
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

clf = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000).fit(df[features], df['default'])
print(clf.coef_, clf.intercept_)

# Use the following code to check yourself
y_pred = clf.predict(df[features])

# Compartive analysis of the model's performance
fpr, tpr, thresholds = metrics.roc_curve(df['default'], y_pred)
print((1.0*(abs(df['default']-y_pred)).sum()) / len(df))
print(metrics.auc(fpr, tpr))

# Define all models to compare
models = {
    'Logistic Regression': clf,  # reuse the one already trained above
    'Decision Tree':        DecisionTreeClassifier(max_depth=5, random_state=0),
    'Random Forest':        RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=0),
}

print("\n--- Comparative Analysis ---")
print(f"{'Model':<22}  {'Error Rate':>10}  {'AUC':>8}")
print("-" * 45)

for name, model in models.items():
    if name != 'Logistic Regression':
        model.fit(df[features], df['default'])
    
    y_pred_m = model.predict(df[features])
    fpr_m, tpr_m, _ = metrics.roc_curve(df['default'], y_pred_m)
    
    error_rate = (1.0 * abs(df['default'] - y_pred_m)).sum() / len(df)
    auc_score  = metrics.auc(fpr_m, tpr_m)
    
    print(f"{name:<22}  {error_rate:>10.4f}  {auc_score:>8.4f}")