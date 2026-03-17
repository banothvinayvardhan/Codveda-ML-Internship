import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load and prep data
data_path = r"C:\Users\Maharaj\OneDrive\Desktop\Codveda_Internship\2) Stock Prices Data Set.csv"
df = pd.read_csv(data_path).drop('date', axis=1)

for c in ['open', 'high', 'low']:
    df[c] = df[c].fillna(df[c].mean())

df['symbol'] = LabelEncoder().fit_transform(df['symbol'])

scale_features = ['symbol', 'open', 'high', 'low', 'volume']
df[scale_features] = StandardScaler().fit_transform(df[scale_features])

X = df.drop('close', axis=1)
y = df['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("model weights:")
for f, w in zip(X.columns, lr_model.coef_):
    print(f, round(w, 4))

preds = lr_model.predict(X_test)

print("\nEvaluation:")
print("MSE:", round(mean_squared_error(y_test, preds), 4))
print("R2 score:", round(r2_score(y_test, preds), 4))