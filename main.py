import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text
from src.dziribert import encode_texts
from src.model import train_and_evaluate


print("--- Étape A : Chargement des données ---")
df = pd.read_csv("./data/sentiment_comments.csv")
df = df[df['label'] != 1] 
df = df.sample(n=min(8000, len(df)), random_state=42)

print("--- Étape B : Nettoyage des données ---")
df["clean_text"] = df["text"].apply(clean_text)

print("--- Étape C : Encodage (DziriBERT) ---")
X_dziri = encode_texts(df["clean_text"].tolist())
y = df["label"]


print("--- Étape D : Partitionnement des données ---")
X_train, X_test, y_train, y_test = train_test_split(X_dziri, y, test_size=0.2, random_state=42)

model_final, report, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)

print("\n" + "="*40)
print(f"ACCURACY FINALE : {accuracy:.4f}")
print("="*40)
print(report)

print("\n--- Étape F : Analyse Exploratoire ---")
all_words = " ".join(df['clean_text']).split()
print(f"Taille du vocabulaire : {len(set(all_words))} mots")

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
pos_words = Counter(" ".join(df[df['label'] == 2]['clean_text']).split()).most_common(15)
sns.barplot(x=[c for w,c in pos_words], y=[w for w,c in pos_words], palette='Greens_r')
plt.title("Top 15 mots - Positifs")

# Top mots Négatifs
plt.subplot(1, 2, 2)
neg_words = Counter(" ".join(df[df['label'] == 0]['clean_text']).split()).most_common(15)
sns.barplot(x=[c for w,c in neg_words], y=[w for w,c in neg_words], palette='Reds_r')
plt.title("Top 15 mots - Négatifs")
plt.tight_layout()
plt.show()


def predict_user_text(text):
    vec = encode_texts([clean_text(text)])
    pred = model_final.predict(vec)[0]
    res = "POSITIF " if pred == 2 else "NÉGATIF"
    print(f"\nTexte : {text}\nRésultat : {res}\n")

print("\n>>> TESTEZ LE MODÈLE (écrivez une phrase en Darija) :")
while True:
    txt = input("Commentaire (ou 'exit' pour quitter) : ")
    if txt.lower() == 'exit': break
    predict_user_text(txt)