import os
import re
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

path = os.getcwd().replace("Scripts", "")

def main():
    data = clean_data(f'{path}Data/Input/Constraint_English_Train.xlsx')
    test_data = clean_data(f'{path}Data/Input/Constraint_English_Test.xlsx')

    model_1, model_2, model_3, model_4, model_5, model_6 = train(data)

    results = []
    model_names = ["Decision Tree", "Random Forest", "Logistic Regression", "KNN", "Gradient Boosting",
                   "Linear Regression"]
    models = [model_1, model_2, model_3, model_4, model_5, model_6]

    for model, name in zip(models, model_names):
        y_true, y_pred = use(model, name, test_data)
        results.append((name, y_true, y_pred))
    create_metrics_comparison_plot(results)
    create_confusion_matrix_comparison(results)


def clean_data(file):
    cleaned_data = []
    data = pd.read_excel(file)

    for index, row in data.iterrows():
        tweet = str(row['tweet'])
        tweet = re.sub(r'[^a-zA-Z0-9-_=+!@#$%^&*();./, ]', '', tweet)
        url_pattern = re.compile(r'https?://\S+|www\.\S+|http?://\S+|http?//\S+|https?//\S+')
        tweet = url_pattern.sub('', tweet)
        label = row['label']
        cleaned_data.append((tweet, label))
    return cleaned_data

def train(data):
    try:
        print("Training models")
        start_time = time.time()
        data_frame = pd.DataFrame(data, columns=['tweet', 'label'])
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(data_frame['tweet'])
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(data_frame['label'])
        # Standardize the features (keep sparse)
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(x)

        # Initialize models
        decision_tree = DecisionTreeClassifier(random_state=42)
        random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
        knn = KNeighborsClassifier(n_neighbors=5)
        gradient_boosting = GradientBoostingClassifier(random_state=42)
        linear_regression = LogisticRegression(random_state=42, max_iter=1000)

        # Train models
        decision_tree.fit(X_train_scaled, y)
        random_forest.fit(X_train_scaled, y)
        logistic_regression.fit(X_train_scaled, y)
        knn.fit(X_train_scaled, y)
        gradient_boosting.fit(X_train_scaled, y)
        linear_regression.fit(X_train_scaled, y)

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
        )
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def use(model_pack, model_name, input_data):
    model, vectorizer, scaler, label_encoder = model_pack
    data_frame = pd.DataFrame(input_data, columns=['tweet', 'label'])

    # Use the training vectorizer and scaler – do NOT refit on test data
    x_test = vectorizer.transform(data_frame['tweet'])
    x_test_scaled = scaler.transform(x_test)

    # Use the training label encoder to convert test labels
    y_true = label_encoder.transform(data_frame['label'])

    # Predict with the model
    y_prediction = model.predict(x_test_scaled)

    # Evaluate
    evaluate_model(y_true, y_prediction, model_name)

    return y_true, y_prediction

def evaluate_model(actual_labels, predicted_labels, model_name):
    print(f"{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(actual_labels, predicted_labels):.2f}")
    print(f"Precision: {precision_score(actual_labels, predicted_labels):.2f}")
    print(f"Recall: {recall_score(actual_labels, predicted_labels):.2f}")
    print(f"F1-Score: {f1_score(actual_labels, predicted_labels):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(actual_labels, predicted_labels))
    print("\nClassification Report:")
    print(classification_report(actual_labels, predicted_labels, zero_division=0))
    print('-' * 50)


def create_metrics_comparison_plot(results):
    """Create a comprehensive metrics comparison plot"""
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
    """Create confusion matrices for all models"""
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


if __name__ == "__main__":
    main()
