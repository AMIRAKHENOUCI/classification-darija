from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="DziriBERT Optimized"):
    # الـ RBF Kernel هو الأفضل لتمثيل BERT المعقد
    pipeline = Pipeline([
        ('norm', Normalizer()), 
        ('svm', SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced'))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return pipeline, report, acc