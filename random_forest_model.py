from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def random_forest(X_train, X_test, y_train, y_test):
    """
    Builds, trains, and tests a Random Forest classification model,
    returning performance metrics.
    """

    # RandomForestClassifier() is initialized.
    # We use some common default hyperparameters for good initial performance.
    rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

    # Trained on the training data.
    print("\nTraining Random Forest model...")
    rf_model.fit(X_train, y_train)

    # Tested on the test set.
    print("Generating predictions...")
    y_pred = rf_model.predict(X_test)

    # Performance is assessed using the confusion matrix and classification report.
    print("\n" + "=" * 40)
    print("Random Forest Model Evaluation")
    print("=" * 40)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix\n")
    print(cm)

    plt.figure(figsize=(5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    # Classification Report
    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    return rf_model