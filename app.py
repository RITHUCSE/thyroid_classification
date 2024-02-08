from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    new_rf = pickle.load(file)

# Load the label encoder and scaler
with open('s_model.pkl', 'rb') as file:
    s_encoder = pickle.load(file)
new_rf.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    age = int(request.form['age'])
    sex = request.form['sex']
    t3 = float(request.form['t3'])
    t4u = float(request.form['t4u'])
    tt4 = float(request.form['tt4'])
    fti = float(request.form['fti'])
    tsh = float(request.form['tsh'])

    # Preprocess the input data
    user_data = {
        'age': age,
        'sex': sex,
        'T3': t3,
        'T4U': t4u,
        'TT4': tt4,
        'FTI': fti,
        'TSH': tsh,
        'Label': 1  # Dummy label, not used for prediction
    }
    user_df = pd.DataFrame(user_data, index=[0])
    user_df['sex'] = s_encoder.transform(user_df['sex'])
    # No need for scaling if you only have a label encoder

    # Make prediction
    prediction = new_rf.predict(user_df)

    # Determine the classification
    if prediction[0] == 1:
        result = "hyperthyroid"
    else:
        result = "hypothyroid"

    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
