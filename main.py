# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error,confusion_matrix
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from flask import Flask, request, jsonify
# app = Flask(__name__)
#
# @app.route('/predict', methods=['POST'])
# def predict():
#
#     data = request.get_json()
#     df = pd.DataFrame(data, index=[0])
#     y_pred_rf= rf_classifier.predict(df)
#     return jsonify({'prediction': y_pred_rf})
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     df = pd.read_csv('./Cardiovascular_Disease_Dataset.csv')
#     df = df.replace({'target': {0: 'Absence of Heart Disease', 1: 'Presence of Heart Disease'}})
#     df=df.drop(columns='patientid')
#     X_disease = df.drop(columns='target')
#     y = df.target
#     scaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_disease)
#     X = pd.DataFrame(scaler, columns=X_disease.columns)
#     X.describe().T
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#     rf_classifier = RandomForestClassifier(n_estimators=500, criterion="entropy", max_features='log2')
#     rf_classifier.fit(X_train, y_train)
#     print(X_test.iloc[1,1:])
#     y_pred_rf = rf_classifier.predict(X_test)
#     print('Classification Report\n\n', classification_report(y_test, y_pred_rf))
#     app.run()
#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    rf_classifier = load_model()
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    scaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(df)
    X = pd.DataFrame(scaler, columns=df.columns)
    y_pred_rf = rf_classifier.predict(X)
    return jsonify({'prediction': y_pred_rf[0]})


def load_model():
    df = pd.read_csv('./Cardiovascular_Disease_Dataset.csv')
    df = df.replace({'target': {0: 'Absence of Heart Disease', 1: 'Presence of Heart Disease'}})
    df = df.drop(columns='patientid')
    X_disease = df.drop(columns='target')
    y = df.target
    scaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_disease)
    X = pd.DataFrame(scaler, columns=X_disease.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    rf_classifier = RandomForestClassifier(n_estimators=500, criterion="entropy", max_features='log2')
    rf_classifier.fit(X_train, y_train)
    return rf_classifier


if __name__ == '__main__':
    app.run(debug=True)
