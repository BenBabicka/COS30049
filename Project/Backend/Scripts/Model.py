import pickle
import re

import pandas as pd

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

class Model:
    def __init__(self, name):
        self.name = name
        self.model = load_model(name)

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