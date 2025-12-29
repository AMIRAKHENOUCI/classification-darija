import pandas as pd
import numpy as np
from src.preprocessing import clean_text
from src.dziribert import encode_texts
from src.model import train_and_evaluate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# a. Chargement des données
print("a. Chargement des données...")
df = pd.read_csv("./data/sentiment_comments.csv")

# مهم جداً: الأستاذ طلب إيجابي وسلبي فقط
# نحذف الصنف 1 (المحايد) ونبقي 0 (سلبي) و 2 (إيجابي)
df = df[df['label'] != 1]

# نأخذ عينة محترمة (مثلاً 3000 سطر) لضمان السرعة والدقة
df = df.sample(n=min(8000, len(df)), random_state=42)

# b. Nettoyage (تأكدي أن ملف preprocessing فيه Stemming كما طلب الأستاذ)
print("b. Nettoyage des données...")
df["clean_text"] = df["text"].apply(clean_text)

# c. Modélisation (DziriBERT)
print("c. Encodage avec DziriBERT...")
X_dziri = encode_texts(df["clean_text"].tolist())
y = df["label"]

# d. Partitionnement (80% Train, 20% Test)
print("d. Partitionnement 80/20...")
X_train, X_test, y_train, y_test = train_test_split(X_dziri, y, test_size=0.2, random_state=42)

# e. Test et évaluation (Option Machine Learning)
model, report, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)

print("\n" + "="*40)
print(f"RÉSULTAT FINAL ACCURACY: {accuracy:.4f}")
print("="*40)
print(report)

# --- بداية المرحلة (e) : التحليل بالصور ---
print("\n--- e. Analyse Exploratoire par Visualisation ---")

# 1. حساب حجم القاموس (Vocabulaire)
all_words = " ".join(df['clean_text']).split()
vocab_size = len(set(all_words))
print(f"Taille du vocabulaire total: {vocab_size} mots")

# 2. رسم الكلمات الأكثر تكراراً في التعليقات الإيجابية والسلبية
plt.figure(figsize=(15, 6))

# الكلمات الإيجابية (Label 2)
plt.subplot(1, 2, 1)
pos_words = " ".join(df[df['label'] == 2]['clean_text']).split()
most_common_pos = Counter(pos_words).most_common(15)
words_p, counts_p = zip(*most_common_pos)
sns.barplot(x=list(counts_p), y=list(words_p), palette='Greens_r')
plt.title("Mots les plus fréquents - Positif")

# الكلمات السلبية (Label 0)
plt.subplot(1, 2, 2)
neg_words = " ".join(df[df['label'] == 0]['clean_text']).split()
most_common_neg = Counter(neg_words).most_common(15)
words_n, counts_n = zip(*most_common_neg)
sns.barplot(x=list(counts_n), y=list(words_n), palette='Reds_r')
plt.title("Mots les plus fréquents - Négatif")

plt.tight_layout()
plt.show() # هادي راح تفتحلك نافذة فيها الرسم البياني