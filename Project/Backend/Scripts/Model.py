import csv
import os
import pickle
import re

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report


def load_model(name):
    try:
        with open(name + ".pkl", 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        return f"Error: {e}"

def clean_data(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9-_=+!@#$%^&*();./, ]', '', text)
    url_pattern = re.compile(r'https?://\S+|www\.\S+|http?://\S+|http?//\S+|https?//\S+')
    text = url_pattern.sub('', text)
    text = text.strip()
    return text

def create_evaluation_data():
    path = os.getcwd().replace("Scripts", "")
    validation_files = os.listdir(f'{path}Data/Output/Validate')
    validation_data = list()
    for file in validation_files:
        validation_data += read_data(f'{path}Data/Output/Validate/{file}')
    return validation_data

def read_data(file):
    #Open the cleaned file
    with open(file, 'r') as f:
        #Read each line and return as list
        reader = csv.reader(f)
        data = list(reader)
    return data

class Model:
    def __init__(self, name):
        self.name = name.split("/")[-1].split(".")[0]
        self.model = load_model(name)
        self.evaluation_data = create_evaluation_data()

    def use_model(self, input_data):
        data = []
        for text in input_data:
            text = clean_data(text)
            data.append(text)
        model, vectorizer, scaler, label_encoder = self.model
        data_frame = pd.DataFrame(data, columns=['text'])

        # Use the training vectorizer and scaler
        x_test = vectorizer.transform(data_frame['text'])
        x_test_scaled = scaler.transform(x_test)

        # Predict with the model
        y_prediction = model.predict(x_test_scaled)
        # Evaluate
        print(f"{self.name} Evaluation:")
        return y_prediction

    def evaluate_model(self):
        model, vectorizer, scaler, label_encoder = self.model
        data_frame = pd.DataFrame(self.evaluation_data, columns=['text', 'label'])

        # Use the training vectorizer and scaler
        x_test = vectorizer.transform(data_frame['text'])
        x_test_scaled = scaler.transform(x_test)

        # Use the training label encoder to convert test labels
        actual_labels = label_encoder.transform(data_frame['label'])

        # Predict with the model
        predicted_labels = model.predict(x_test_scaled)
        # Evaluate
        data = {
                "MSE": mean_squared_error(actual_labels, predicted_labels),
                "R2 score": r2_score(actual_labels, predicted_labels)
        }

        if self.name != "Linear Regression" and self.name != "Ridge" and self.name != "Lasso" and self.name != "Elastic Net":
            data["Accuracy"] = f"{accuracy_score(actual_labels, predicted_labels):.2f}"
            data["Precision"] = f"{precision_score(actual_labels, predicted_labels):.2f}"
            data["Recall"] = f"{recall_score(actual_labels, predicted_labels):.2f}"
            data["F1-Score"] = f"{f1_score(actual_labels, predicted_labels):.2f}"
            data["Confusion Matrix"] = confusion_matrix(actual_labels, predicted_labels)
            data["Classification Report"] = classification_report(actual_labels, predicted_labels, zero_division=0)
        return data