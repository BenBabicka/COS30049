import os

import datasets
import numpy
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree
from sklearn.ensemble import RandomForestClassifier  # For Random Forest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def main():
    path = os.getcwd().replace("Scripts", "")
    xlsxFile = pd.read_excel(f'{path}Data/Input/Constraint_English_Train.xlsx')
    data = []
    for index, row in xlsxFile.iterrows():
        tweet = row['tweet']
        label = row['label']
        data.append((tweet, label))

    xlsxFile = pd.read_excel(f'{path}Data/Input/Constraint_English_Test.xlsx')
    test_data = []
    for index, row in xlsxFile.iterrows():
        tweet = row['tweet']
        label = row['label']
        test_data.append((tweet, label))

    Train(data, test_data)

def Train(data, test_data):
        data_frame = pd.DataFrame(data, columns=['tweet', 'label'])
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(data_frame['tweet'])
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(data_frame['label'])
        x = csr_matrix(x)
        # Split the data into training and test sets
        data_frame_test = pd.DataFrame(test_data, columns=['tweet', 'label'])
        x_test = vectorizer.fit_transform(data_frame_test['tweet'])
        le = preprocessing.LabelEncoder()
        y_test = le.fit_transform(data_frame_test['label'])
        x_shape, _ = x_test.shape
        _, y_shape = x.shape
        newArray = np.zeros((x_shape, y_shape))
        print(newArray.shape)
        x_test = np.array(x_test.todense())
        print(x_test.shape)
        newArray[:x_test.shape[0], :x_test.shape[1]] = x_test
        x_test = csr_matrix(newArray)

        print(x.shape)
        print(x_test.shape)
        print(y.shape)
        print(y_test.shape)

        # Standardize the features
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(x)
        X_test_scaled = scaler.transform(x_test)

        # Initialize Decision Tree and Random Forest
        decision_tree = DecisionTreeClassifier(random_state=42)  # Decision Tree
        random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest

        # Train both models
        decision_tree.fit(X_train_scaled, y)  # Train Decision Tree
        random_forest.fit(X_train_scaled, y)  # Train Random Forest

        # Predict with both models
        y_pred_dt = decision_tree.predict(X_test_scaled)  # Predict with Decision Tree
        y_pred_rf = random_forest.predict(X_test_scaled)  # Predict with Random Forest

        # Function for evaluation
        # Evaluate Decision Tree and Random Forest
        evaluate_model(y_test, y_pred_dt, "Decision Tree")
        evaluate_model(y_test, y_pred_rf, "Random Forest")


def evaluate_model(y_test, y_pred, model_name):
    print(f"{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print('-' * 50)

if __name__ == "__main__":
    main()