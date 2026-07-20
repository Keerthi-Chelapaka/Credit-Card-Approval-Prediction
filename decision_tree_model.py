from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def d_tree(X_train, X_test, y_train, y_test):

    dt = DecisionTreeClassifier(random_state=42)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    print("***** Decision Tree Classifier *****")
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    plt.figure(figsize=(5,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Decision Tree")
    plt.show()

    print("Classification Report")
    print(classification_report(y_test, y_pred))

    return dt