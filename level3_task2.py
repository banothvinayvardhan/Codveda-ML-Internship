import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\Maharaj\OneDrive\Desktop\Codveda_Internship\1) iris.csv")

# subset to just 2 classes and 2 features so we can plot it
df = df[df['species'].isin(['setosa', 'versicolor'])]
X = df[['sepal_length', 'sepal_width']]
y = LabelEncoder().fit_transform(df['species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# linear svm
svm_lin = SVC(kernel='linear')
svm_lin.fit(X_train, y_train)
lin_preds = svm_lin.predict(X_test)
print("Linear Kernel Acc:", accuracy_score(y_test, lin_preds))

# rbf svm
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
rbf_preds = svm_rbf.predict(X_test)
print("RBF Kernel Acc:", accuracy_score(y_test, rbf_preds))

# plotting the decision boundary for the linear one
h = 0.02
x_min, x_max = X['sepal_length'].min() - 1, X['sepal_length'].max() + 1
y_min, y_max = X['sepal_width'].min() - 1, X['sepal_width'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svm_lin.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X['sepal_length'], X['sepal_width'], c=y, edgecolors='k')
plt.title("SVM Decision Boundary (Linear)")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()