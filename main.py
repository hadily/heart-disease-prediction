import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
from collections import OrderedDict

app = Flask(__name__)


def transform_chestpain(value):
    if value in [0, 1, 2]:
        return 0
    elif value == 3:
        return 1
    else:
        return None


def transform_restingrelectro(value):
    if value in [1, 2]:
        return 1
    elif value == 0:
        return 0
    else:
        return None


def data_cleaning(df):
    df['chestpain'] = df['chestpain'].apply(transform_chestpain)
    df['restingrelectro'] = df['restingrelectro'].apply(transform_restingrelectro)
    df = df.drop(columns=['exerciseangia'])
    df = df.drop(columns=['oldpeak'])
    df = df.drop(columns=['noofmajorvessels'])
    # df = df[df['age'] >= 40]
    df = df[df['slope'] != 0]
    return df


def load_data():
    df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')
    df_cleaned = data_cleaning(df)
    return df_cleaned


def train_model():
    df = load_data()
    X = df.drop(columns=['target', 'patientid'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    print(classification_report(y_test, y_pred))  # Print classification report for evaluation
    return rf_classifier


# Train the model when the server starts
rf_model = train_model()


@app.route('/patient/<int:patient_id>', methods=['GET'])
def get_patient_info(patient_id):
    df = load_data()
    patient_data = df[df['patientid'] == patient_id].to_dict(orient='records')
    if not patient_data:
        return jsonify({'error': 'Patient not found'}), 404
    return jsonify(patient_data[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()






    # Modify specific values
    gender = data.get('age', 3)  # Get gender if it exists, otherwise set to 'unknown'
    gender = data.get('gender', 0)  # Get gender if it exists, otherwise set to 'unknown'
    gender = data.get('chestpain',0)  # Get gender if it exists, otherwise set to 'unknown'
    gender = data.get('serumcholestrol', 1000)  # Get gender if it exists, otherwise set to 'unknown'
    gender = data.get('slope', 2)  # Get gender if it exists, otherwise set to 'unknown'
    gender = data.get('restingrelectro', 2)  # Get gender if it exists, otherwise set to 'unknown'
    # Modify specific values
    data['gender'] = 0  # Change gender to a constant value
    data['age'] = 30  # Change gender to a constant value
    data['chestpain'] = 0 # Change gender to a constant value
    data['serumcholestrol'] = 2000  # Change gender to a constant value
    data['restingrelectro'] = 2  # Change gender to a constant value
    data['fastingbloodsugar'] = 2  # Change gender to a constant value
    data['slope'] = 1  # Change gender to a constant value
    data.pop('device', None)
    data.pop('timestamp', None)
    data.pop('id', None)
    data.pop('spO2', None)
    post_order = [
        "age",
        "gender",
        "chestpain",
        "restingBP",
        "serumcholestrol",
        "fastingbloodsugar",
        "restingrelectro",
        "maxheartrate",
        "slope"
    ]
    # Reorder the data
    ordered_data = {attr: data[attr] for attr in post_order if attr in data}







    df = pd.DataFrame(ordered_data, index=[0])
    prediction = rf_model.predict(df)[0]
    # Map prediction to labels
    prediction_label = "Absence of Heart Disease" if prediction == 0 else "Presence of Heart Disease"
    return jsonify({'prediction': prediction_label})


# post_order = [
#     "age", 
#     "gender", 
#     "chestpain", 
#     "restingBP", 
#     "serumcholestrol", 
#     "fastingbloodsugar", 
#     "restingrelectro", 
#     "maxheartrate", 
#     "slope"
# ]

if __name__ == '__main__':
    app.run(debug=True)