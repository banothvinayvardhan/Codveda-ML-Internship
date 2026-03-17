import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load built-in digits data
digits = load_digits()
X = digits.data / 16.0  # normalize pixels
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build and train neural net
nn = MLPClassifier(hidden_layer_sizes=(128,), max_iter=40, random_state=42)
nn.fit(X_train, y_train)

preds = nn.predict(X_test)
print("NN Test Accuracy:", accuracy_score(y_test, preds))

# loss curve
plt.figure(figsize=(7, 5))
plt.plot(nn.loss_curve_)
plt.title('Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
print("loading loss graph...")
plt.show()