import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\Maharaj\OneDrive\Desktop\Codveda_Internship\1) iris.csv")

df['species'] = LabelEncoder().fit_transform(df['species'])

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# tree with max depth of 3 so it doesn't overfit
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

preds = dt.predict(X_test)
print("Acc:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds, average='weighted'))

# plot the actual tree
plt.figure(figsize=(10, 6))
plot_tree(dt, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree")
print("loading plot...")
plt.show()