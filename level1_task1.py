import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv(r"C:\Users\Maharaj\OneDrive\Desktop\Codveda_Internship\2) Stock Prices Data Set.csv")
df = df.drop('date', axis=1)

# fix missing values
for col in ['open', 'high', 'low']:
    df[col] = df[col].fillna(df[col].mean())

# encode the stock symbols
encoder = LabelEncoder()
df['symbol'] = encoder.fit_transform(df['symbol'])

# scaling
scaler = StandardScaler()
cols_to_scale = ['symbol', 'open', 'high', 'low', 'volume']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

X = df.drop('close', axis=1)
y = df['close']

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("done preprocessing")
print("train size:", len(X_train))
print("test size:", len(X_test))