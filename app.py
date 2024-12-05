# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask("__name__")

# Load required resources
try:
    df_base = pd.read_csv("first_telc.csv")
    model = pickle.load(open("churn-model.sav", "rb"))
except FileNotFoundError as e:
    raise FileNotFoundError("Required file is missing. Ensure 'first_telc.csv' and 'churn-model.sav' exist.") from e

# Helper function for input validation
def validate_input(data):
    required_fields = [
        'query1', 'query2', 'query3', 'query4', 'query5', 'query6',
        'query7', 'query8', 'query9', 'query10', 'query11', 'query12',
        'query13', 'query14', 'query15', 'query16', 'query17', 'query18', 'query19'
    ]
    for field in required_fields:
        if not data.get(field):
            return False, f"Missing input for {field}"
    return True, ""

# Define routes
@app.route("/")
def load_page():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Collect inputs from the form
    inputs = {
        'SeniorCitizen': request.form.get('query1'),
        'MonthlyCharges': float(request.form.get('query2', 0)),
        'TotalCharges': float(request.form.get('query3', 0)),
        'gender': request.form.get('query4'),
        'Partner': request.form.get('query5'),
        'Dependents': request.form.get('query6'),
        'PhoneService': request.form.get('query7'),
        'MultipleLines': request.form.get('query8'),
        'InternetService': request.form.get('query9'),
        'OnlineSecurity': request.form.get('query10'),
        'OnlineBackup': request.form.get('query11'),
        'DeviceProtection': request.form.get('query12'),
        'TechSupport': request.form.get('query13'),
        'StreamingTV': request.form.get('query14'),
        'StreamingMovies': request.form.get('query15'),
        'Contract': request.form.get('query16'),
        'PaperlessBilling': request.form.get('query17'),
        'PaymentMethod': request.form.get('query18'),
        'tenure': int(request.form.get('query19', 0))
    }

    # Validate inputs
    is_valid, error_message = validate_input(inputs)
    if not is_valid:
        return render_template('home.html', output1="Error", output2=error_message, **inputs)

    # Prepare input data
    new_data = pd.DataFrame([inputs])

    # Process tenure for grouping
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    new_data['tenure_group'] = pd.cut(new_data['tenure'].astype(int), range(1, 80, 12), right=False, labels=labels)
    new_data.drop(columns=['tenure'], axis=1, inplace=True)

    # Encode categorical variables
    dummies = pd.get_dummies(new_data, columns=[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ])

    # Align with model's expected features
    dummies = dummies.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict churn and confidence
    try:
        prediction = model.predict(dummies)[0]
        probability = model.predict_proba(dummies)[:, 1][0]
    except Exception as e:
        return render_template('home.html', output1="Error", output2="Model failed to predict. Please try again.", **inputs)

    # Create output messages
    output1 = "This customer is likely to churn!" if prediction == 1 else "This customer is likely to continue!"
    output2 = f"Confidence: {probability * 100:.2f}%"

    return render_template('home.html', output1=output1, output2=output2, **inputs)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

@app.route("/", methods=['POST'])
def predict():
    # Collect inputs from the form
    inputs = {
        'SeniorCitizen': request.form.get('query1'),
        'MonthlyCharges': float(request.form.get('query2', 0)),
        'TotalCharges': float(request.form.get('query3', 0)),
        'gender': request.form.get('query4'),
        'Partner': request.form.get('query5'),
        'Dependents': request.form.get('query6'),
        'PhoneService': request.form.get('query7'),
        'MultipleLines': request.form.get('query8'),
        'InternetService': request.form.get('query9'),
        'OnlineSecurity': request.form.get('query10'),
        'OnlineBackup': request.form.get('query11'),
        'DeviceProtection': request.form.get('query12'),
        'TechSupport': request.form.get('query13'),
        'StreamingTV': request.form.get('query14'),
        'StreamingMovies': request.form.get('query15'),
        'Contract': request.form.get('query16'),
        'PaperlessBilling': request.form.get('query17'),
        'PaymentMethod': request.form.get('query18'),
        'tenure': int(request.form.get('query19', 0))
    }

    # Debugging: Check if inputs are being received
    print(f"Received inputs: {inputs}")

    # Validate inputs
    is_valid, error_message = validate_input(inputs)
    if not is_valid:
        return render_template('home.html', output1="Error", output2=error_message, **inputs)

    # Prepare input data
    new_data = pd.DataFrame([inputs])

    # Process tenure for grouping
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    new_data['tenure_group'] = pd.cut(new_data['tenure'].astype(int), range(1, 80, 12), right=False, labels=labels)
    new_data.drop(columns=['tenure'], axis=1, inplace=True)

    # Encode categorical variables
    dummies = pd.get_dummies(new_data, columns=[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ])

    # Align with model's expected features
    dummies = dummies.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict churn and confidence
    try:
        prediction = model.predict(dummies)[0]
        probability = model.predict_proba(dummies)[:, 1][0]
    except Exception as e:
        return render_template('home.html', output1="Error", output2="Model failed to predict. Please try again.", **inputs)

    # Create output messages
    output1 = "This customer is likely to churn!" if prediction == 1 else "This customer is likely to continue!"
    output2 = f"Confidence: {probability * 100:.2f}%"

    # Debugging: Check if output is generated
    print(f"Prediction: {output1}, Confidence: {output2}")

    return render_template('home.html', output1=output1, output2=output2, **inputs)
