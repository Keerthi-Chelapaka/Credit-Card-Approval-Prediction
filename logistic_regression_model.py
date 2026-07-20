import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def logistic_reg(X_train, X_test, y_train, y_test):
    """
    Builds, trains, tests, and evaluates a Logistic Regression classification model.
    Accepts pre-split training and testing data (X_train, X_test, y_train, y_test).
    Returns the trained model and its predictions.
    """

    # ----- MODEL INITIALIZATION -----
    print("Initializing Logistic Regression classifier...")

    # Initialize Logistic Regression with a fixed random state for reproducibility
    lr_model = LogisticRegression(random_state=42,max_iter=5000,class_weight="balanced")

    # ----- MODEL TRAINING -----
    print("Training the Logistic Regression model on training data...")

    # X_train contains input features (income, education, credit history, etc.)
    # y_train contains target labels (approval status: 1=Approved, 0=Not Approved)
    lr_model.fit(X_train, y_train)

    # ----- MODEL PREDICTION -----
    print("Generating predictions on the unseen test dataset...")

    # Predict approval status for new applicants in X_test
    y_pred = lr_model.predict(X_test)

    # ----- MODEL EVALUATION -----
    print("\n----- Model Evaluation Results -----")

    # 1. Confusion Matrix: Shows counts of True Positives, True Negatives,
    # False Positives, and False Negatives.
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix")
    print(cm)

    # Optional Visual Representation (Seaborn Heatmap)
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    # 2. Classification Report
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return lr_model