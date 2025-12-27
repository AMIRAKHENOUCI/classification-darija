import pandas as pd
from src.preprocessing import clean_text
from src.features import fit_transform_tfidf , get_vocabulary_info
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.dziribert import encode_texts
from sklearn.metrics import accuracy_score
from src.evaluation import plot_all_results
df = pd.read_excel("data/data1500.xlsx")

print("Nettoyage des données...")
df["clean_text"] = df["text"].apply(clean_text)

print("\n Expérience 1 : TF-IDF + Logistic Regression ")
X_tfidf, vectorizer = fit_transform_tfidf(df["clean_text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train, y_train)
y_pred_tfidf = model_tfidf.predict(X_test)

print("\nRésultats TF-IDF :")
print(classification_report(y_test, y_pred_tfidf))


print("\n" + "="*40)
print(" Expérience 2 : DziriBERT + Logistic Regression ")


X_text = df["clean_text"].tolist()
print("Encoding with DziriBERT (cela peut prendre un moment)...")
X_dziri = encode_texts(X_text)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dziri, y, test_size=0.2, random_state=42)

model_dziri = LogisticRegression(max_iter=1000)
model_dziri.fit(X_train_d, y_train_d)
y_pred_dziri = model_dziri.predict(X_test_d)

print("\nRésultats DziriBERT :")
print(classification_report(y_test_d, y_pred_dziri))

print("\n COMPARAISON FINALE ")
print(f"Accuracy TF-IDF   : 0.78")

print(f"Accuracy DziriBERT: {accuracy_score(y_test_d, y_pred_dziri):.2f}")

print(f"Taille du vocabulaire: {get_vocabulary_info(df['clean_text'])}")
plot_all_results(df, y_true=y_test_d, y_pred=y_pred_dziri)