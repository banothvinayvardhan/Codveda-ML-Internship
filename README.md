# Codveda ML Internship Projects

Hey! This repository contains all the code and datasets for the tasks I completed during my Machine Learning Internship at Codveda. 

The internship was split into three levels, focusing on building end-to-end ML pipelines. Here is a quick breakdown of what's inside:

### Level 1 (The Basics)
- **Task 1:** Cleaned up a messy stock prices dataset (handled missing values, encoded text, and scaled the numerical features).
- **Task 2:** Built a simple Linear Regression model to predict those stock prices.
- **Task 3:** Wrote a K-Nearest Neighbors (KNN) script to classify Iris flowers (tested a few values and found that k=5 worked best).

### Level 2 (Intermediate)
- **Task 1:** Used Logistic Regression to predict customer churn in a telecom dataset and plotted the ROC curve.
- **Task 2:** Built a Decision Tree for the Iris dataset. I intentionally capped the tree depth at 3 to stop it from overfitting.
- **Task 3:** Did some unsupervised learning using K-Means clustering to segment customers based on their call minutes. 

### Level 3 (Advanced)
- **Task 1:** Trained a Random Forest model on the churn data and used GridSearchCV to find the best hyperparameters. Also plotted the feature importances.
- **Task 2:** Set up a Support Vector Machine (SVM) to classify data and mapped out the linear decision boundary.
- **Task 3:** Built a Neural Network (Multi-Layer Perceptron) to recognize handwritten digits. *(Note: I used Scikit-Learn's MLPClassifier here instead of TensorFlow/Keras to sidestep some local Python version conflicts I was running into, but it gets the job done nicely!)*

### How to run the code
Everything is written in standard Python. If you want to run the scripts yourself, you'll just need to install a few standard libraries:
`pip install pandas numpy scikit-learn matplotlib`

All the required CSV files are already here in the repo, so the scripts should run straight out of the box without needing to change any file paths.
