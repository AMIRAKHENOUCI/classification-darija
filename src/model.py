from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="Model"):
    print(f"\n--- Entraînement du modèle : {model_name} ---")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return clf, report