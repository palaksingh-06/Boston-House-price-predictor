from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load your model
model = pickle.load(open('model.pkl', 'rb'))

# Load and preprocess Boston dataset (just for column names here)
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(X, columns=columns)

@app.route('/')
def home():
    return render_template('index.html', columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[col]) for col in columns]
    final_input = np.array(input_data).reshape(1, -1)
    prediction = model.predict(final_input)[0]
    return render_template('index.html', columns=columns, prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

