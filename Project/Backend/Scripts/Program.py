import csv
import os
import pickle
import random
import re
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

path = os.getcwd().replace("Scripts", "")

def main():
    print("Cleaning data...")
    #Find all files in the input and clean the data
    files = os.listdir(f'{path}Data/Input')
    for file in files:
        clean_data_file(f'{path}Data/Input/{file}')
    #Initliaze the datasets
    training_data = list()
    test_data = list()

    #File all cleaned files in the output folders
    training_files = os.listdir(f'{path}Data/Output/Train')
    testing_files = os.listdir(f'{path}Data/Output/Test')

    #Read each cleaned file
    for file in training_files:
        training_data += read_data(f'{path}Data/Output/Train/{file}')
    for file in testing_files:
        test_data += read_data(f'{path}Data/Output/Test/{file}')

    #Train each model
    model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9 = train(training_data)

    #Initliaze the results
    results = []
    model_names = ["Decision Tree", "Random Forest", "Logistic Regression", "KNN", "Gradient Boosting",
                   "Linear Regression", "Ridge", "Lasso", "Elastic Net"]
    models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9]

    #Test each model and return the evaluation of the model
    for model, name in zip(models, model_names):
        filename = f'{path}Data/Output/Models/{name}.pkl'
        #Save the model to a file for later use
        """with open(filename, 'wb') as file:
            pickle.dump(model, file)"""

        y_true, y_prediction = use(model, name, test_data)
        if y_true is not None and y_prediction is not None:
            results.append((name, y_true, y_prediction))

    #Create a plot of the results
    create_metrics_comparison_plot(results)
    create_confusion_matrix_comparison(results)

def clean_data(text:str):
    # 1. Convert all text to lower.
    # 2. Keep these characters only (a-zA-Z0-9-_=+!@#$%^&*();./, ).
    # 3. Remove all links.
    # 4. Remove all white space.
    # 5. Return the cleaned data
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9-_=+!@#$%^&*();./, ]', '', text)
    url_pattern = re.compile(r'https?://\S+|www\.\S+|http?://\S+|http?//\S+|https?//\S+')
    text = url_pattern.sub('', text)
    text = text.strip()
    keywords = ['covid', 'pfizer', 'astrazeneca', 'vaccine', 'inoculation', 'immunization', 'vaccination', 'jab', 'quarantine', 'injection', 'booster', 'symptoms', 'health', 'condition']
    for keyword in keywords:
        if keyword in text:
            return text
    return None

def find_label(label):
    if type(label) == str and (label.lower() == "true"):
        label = 'real'
    elif type(label) == str and (label.lower() == "false"):
        label = 'fake'
    if type(label) == bool and label == True:
        label = 'real'
    elif type(label) == bool and label == False:
        label = 'fake'
    return label

def clean_data_file(file):
    try:
        #Create clean data list
        cleaned_data = list()
        if ".xlsx" in file:
            #Read Excel file
            data = pd.read_excel(file)
            for index, row in data.iterrows():
                #Clean the text field
                text = clean_data(str(row['tweet']))
                if text is None:
                    continue
                #Assign label
                label = find_label(row['label'])

                #If no text is present don't add to list
                if text != '' or label is not None:
                    cleaned_data.append((text, label))
        elif ".csv" in file:
            #Read csv file
            data = pd.read_csv(file)
            for index, row in data.iterrows():
                #Clean the text field
                text = clean_data(str(row[0]))
                if text is None:
                    continue
                #Assign label
                label = find_label(row[1])

                #If no text is present don't add to list
                if text != '' or label is not None:
                    cleaned_data.append((text, label))
        else:
            #Raise error if not valid file type
            raise "Invalid file type. Please use .xlsx, or .csv"
        #Create the output path
        output = file.replace('.xlsx', '.csv').replace('Input', 'Output')

        #Place in folder for what use it will have. This is dependent on the file name and if it contains (test, train, val or validate)
        if 'test' in output.lower():
            output = f"{output.split('/')[0]}/Output/Test/{output.split('/')[-1]}"
        elif 'val' in output.lower() or 'validate' in output.lower():
            output = f"{output.split('/')[0]}/Output/Validate/{output.split('/')[-1]}"
        else:
            output = f"{output.split('/')[0]}/Output/Train/{output.split('/')[-1]}"
        #Save to new csv file
        with open(output, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(cleaned_data)
    except Exception as e:
        #Print error and return nothing if error occurs
        print(f"Error: Cleaning data - {e}")
        return None

def read_data(file):
    #Open the cleaned file
    with open(file, 'r') as f:
        #Read each line and return as list
        reader = csv.reader(f)
        data = list(reader)
    return data

def train(data):
    try:
        print("Training models")
        start_time = time.time()
        #convert data to data frame
        data_frame = pd.DataFrame(data, columns=['text', 'label'])
        #Tokenize the text so it can be used to train the model
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(data_frame['text'])
        #Create a label encoder
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(data_frame['label'])
        # Create a scale so the model can expect multiple different input sizes
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(x)

        # Initialize models different models
        #Classification models
        decision_tree = DecisionTreeClassifier(random_state=42)
        random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
        knn = KNeighborsClassifier(n_neighbors=5)
        gradient_boosting = GradientBoostingClassifier(random_state=42)
        #Regression models
        linear_regression = LinearRegression()
        ridge = Ridge()
        lasso = Lasso()
        elastic_net = ElasticNet()

        # Train models
        decision_tree.fit(X_train_scaled, y)
        random_forest.fit(X_train_scaled, y)
        logistic_regression.fit(X_train_scaled, y)
        knn.fit(X_train_scaled, y)
        gradient_boosting.fit(X_train_scaled, y)
        linear_regression.fit(X_train_scaled, y)
        ridge.fit(X_train_scaled, y)
        lasso.fit(X_train_scaled, y)
        elastic_net.fit(X_train_scaled, y)
        print("Finished training models")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        print('-' * 50)
        # Return each model with the fitted preprocessing objects for reuse on test data
        return (
            (decision_tree, vectorizer, scaler, le),
            (random_forest, vectorizer, scaler, le),
            (logistic_regression, vectorizer, scaler, le),
            (knn, vectorizer, scaler, le),
            (gradient_boosting, vectorizer, scaler, le),
            (linear_regression, vectorizer, scaler, le),
            (ridge, vectorizer, scaler, le),
            (lasso, vectorizer, scaler, le),
            (elastic_net, vectorizer, scaler, le),
        )
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def use(model_pack, model_name, input_data):
    model, vectorizer, scaler, label_encoder = model_pack
    data_frame = pd.DataFrame(input_data, columns=['text', 'label'])

    # Use the training vectorizer and scaler
    x_test = vectorizer.transform(data_frame['text'])
    x_test_scaled = scaler.transform(x_test)

    # Use the training label encoder to convert test labels
    y_true = label_encoder.transform(data_frame['label'])

    # Predict with the model
    y_prediction = model.predict(x_test_scaled)
    # Evaluate
    print(f"{model_name} Evaluation:")
    random_selection_prediction(model_pack, model_name, input_data)
    if model_name == "Linear Regression" or model_name == "Ridge" or model_name == "Lasso" or model_name == "Elastic Net":
        create_regression_plot(model_pack, model_name, input_data)
    evaluate_model(y_true, y_prediction, model_name)
    print('_' * 100)
    if model_name == "Linear Regression" or model_name == "Ridge" or model_name == "Lasso" or model_name == "Elastic Net":
        return None, None
    return y_true, y_prediction

def random_selection_prediction(model_pack, model_name, input_data):
    model, vectorizer, scaler, label_encoder = model_pack
    test_group = list()
    for i in range(10):
        test_group.append(input_data[random.randint(0, len(input_data)-1)])

    data_frame = pd.DataFrame(test_group, columns=['text', 'label'])

    # Use the training vectorizer and scaler
    x_test = vectorizer.transform(data_frame['text'])
    x_test_scaled = scaler.transform(x_test)

    # Use the training label encoder to convert test labels
    y_true = label_encoder.transform(data_frame['label'])

    # Predict with the model
    y_prediction = model.predict(x_test_scaled)
    print('\n')
    for i in range(len(test_group)):
        print(f"    Text: {test_group[i][0]}")
        print(f"    Actual Labels: {test_group[i][1]}, {y_true[i]}")
        print(f"    Prediction: {y_prediction[i]}")
        print('\n')


def evaluate_model(actual_labels, predicted_labels, model_name):
    print('\n')
    print(f"MSE: {mean_squared_error(actual_labels, predicted_labels)}")
    print(f"R2 score: {r2_score(actual_labels, predicted_labels)}")

    if model_name != "Linear Regression" and model_name != "Ridge" and model_name != "Lasso" and model_name != "Elastic Net":
        print(f"Accuracy: {accuracy_score(actual_labels, predicted_labels):.2f}")
        print(f"Precision: {precision_score(actual_labels, predicted_labels):.2f}")
        print(f"Recall: {recall_score(actual_labels, predicted_labels):.2f}")
        print(f"F1-Score: {f1_score(actual_labels, predicted_labels):.2f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(actual_labels, predicted_labels))
        print("\nClassification Report:")
        print(classification_report(actual_labels, predicted_labels, zero_division=0))

def create_metrics_comparison_plot(results):
    model_names = [result[0] for result in results]
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

    # Calculate metrics for each model
    for model_name, y_true, y_pred in results:
        metrics['Accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['Precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics['Recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metrics['F1-Score'].append(f1_score(y_true, y_pred, zero_division=0))

    # Create subplots for different visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Bar chart comparing all metrics
    x = np.arange(len(model_names))
    width = 0.2

    ax1.bar(x - 1.5 * width, metrics['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax1.bar(x - 0.5 * width, metrics['Precision'], width, label='Precision', alpha=0.8)
    ax1.bar(x + 0.5 * width, metrics['Recall'], width, label='Recall', alpha=0.8)
    ax1.bar(x + 1.5 * width, metrics['F1-Score'], width, label='F1-Score', alpha=0.8)

    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison - All Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # 2. Accuracy comparison line plot
    ax2.plot(model_names, metrics['Accuracy'], marker='o', linewidth=2, markersize=8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Accuracy Score')
    ax2.set_title('Model Accuracy Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylim(0, 1.1)

    # Add value labels on points
    for i, acc in enumerate(metrics['Accuracy']):
        ax2.annotate(f'{acc:.3f}', (i, acc), textcoords="offset points", xytext=(0, 10), ha='center')

    # 3. F1-Score vs Accuracy scatter plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    for i, (acc, f1, name) in enumerate(zip(metrics['Accuracy'], metrics['F1-Score'], model_names)):
        ax3.scatter(acc, f1, s=100, c=[colors[i]], label=name, alpha=0.7)

    ax3.set_xlabel('Accuracy')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('Accuracy vs F1-Score')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.set_xlim(0, 1.1)
    ax3.set_ylim(0, 1.1)

    # 4. Radar chart for top 3 models
    top_3_indices = np.argsort(metrics['F1-Score'])[-3:]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    ax4 = plt.subplot(2, 2, 4, projection='polar')

    for idx in top_3_indices:
        values = [metrics[metric][idx] for metric in metrics.keys()]
        values += values[:1]  # Complete the circle
        ax4.plot(angles, values, 'o-', linewidth=2, label=model_names[idx])
        ax4.fill(angles, values, alpha=0.25)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics.keys())
    ax4.set_ylim(0, 1)
    ax4.set_title('Top 3 Models - Radar Chart')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.show()


def create_confusion_matrix_comparison(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (model_name, y_true, y_pred) in enumerate(results):
        cm = confusion_matrix(y_true, y_pred)

        # Create heatmap
        im = axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].figure.colorbar(im, ax=axes[i])

        # Add text annotations
        thresh = cm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                axes[i].text(k, j, format(cm[j, k], 'd'),
                             ha="center", va="center",
                             color="white" if cm[j, k] > thresh else "black")

        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_title(f'{model_name} - Confusion Matrix')

        # Get unique labels for tick labels
        unique_labels = sorted(set(np.concatenate([y_true, y_pred])))
        axes[i].set_xticks(range(len(unique_labels)))
        axes[i].set_yticks(range(len(unique_labels)))
        axes[i].set_xticklabels(unique_labels)
        axes[i].set_yticklabels(unique_labels)

    plt.tight_layout()
    plt.show()


def create_regression_plot(model_pack, model_name, input_data):
    model, vectorizer, scaler, label_encoder = model_pack
    data_frame = pd.DataFrame(input_data, columns=['text', 'label'])

    # Transform the data
    x_test = vectorizer.transform(data_frame['text'])
    x_test_scaled = scaler.transform(x_test)

    # Get true labels and predictions
    y_true = label_encoder.transform(data_frame['label'])
    y_pred = model.predict(x_test_scaled)


    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.canvas.manager.set_window_title(model_name)

    # 1. Actual vs Predicted scatter plot with regression line
    ax1.scatter(y_true, y_pred, alpha=0.6, color='blue', s=50)

    # Add perfect prediction line (y = x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add regression line for actual vs predicted
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax1.plot(y_true, p(y_true), 'g-', linewidth=2, label=f'Regression Line (y={z[0]:.3f}x+{z[1]:.3f})')

    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Linear Regression: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add R² score annotation
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Residuals plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='red', s=50)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)

    # 3. Distribution of residuals (histogram)
    ax3.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)

    # Add normal distribution curve
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))
    # Scale to match histogram
    y = y * len(residuals) * (residuals.max() - residuals.min()) / 30
    ax3.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
    ax3.legend()

    # 4. Q-Q plot for residuals normality check
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot: Residuals vs Normal Distribution')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f'{model_name} - Regression Analysis', fontsize=16, y=1.02)
    plt.show()

    # Print regression statistics
    print(f"\n{model_name} Regression Statistics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}")
    print(f"Residuals Mean: {np.mean(residuals):.4f}")
    print(f"Residuals Std: {np.std(residuals):.4f}")


if __name__ == "__main__":
    main()
