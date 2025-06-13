import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(X, columns=columns)
df['MEDV'] = y

# Train model
model = LinearRegression()
model.fit(df[columns], df['MEDV'])

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

