import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\Maharaj\OneDrive\Desktop\Codveda_Internship\churn-bigml-80.csv")

le = LabelEncoder()
df['International plan'] = le.fit_transform(df['International plan'])
df['Voice mail plan'] = le.fit_transform(df['Voice mail plan'])
df['Churn'] = le.fit_transform(df['Churn'])
df = df.drop('State', axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# hyperparameter tuning
grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
rf = RandomForestClassifier(random_state=42)

search = GridSearchCV(rf, grid, cv=3)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best params:", search.best_params_)

y_pred = best_model.predict(X_test)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

# plot feature importance
importance = best_model.feature_importances_
sort_idx = importance.argsort()

plt.figure(figsize=(8, 6))
plt.barh(X.columns[sort_idx], importance[sort_idx])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()