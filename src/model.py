from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="Model"):
    print(f"\n--- Entraînement du modèle : {model_name} ---")
    
    # تعريف المودل (Option 2)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # التوقع (Prediction)
    y_pred = clf.predict(X_test)
    
    # استخراج التقارير (Metrics: Precision, Recall...)
    report = classification_report(y_test, y_pred)
    
    return clf, report