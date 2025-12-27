import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_all_results(df, y_true=None, y_pred=None):
    # --- الجزء الأول: Histogrammes (الكلمات - Option 3) ---
    available_labels = df['label'].unique()
    
    fig, axes = plt.subplots(1, len(available_labels), figsize=(15, 6))
    if len(available_labels) == 1: axes = [axes]

    for i, label in enumerate(available_labels):
        subset = df[df['label'] == label]
        words = " ".join(subset["clean_text"].astype(str)).split()
        if words:
            most_common = Counter(words).most_common(15)
            w, c = zip(*most_common)
            axes[i].barh(w, c, color=['red' if label==0 else 'green' if label==2 else 'blue'][0])
            axes[i].set_title(f'Top 15 Mots - Label {label}')
            axes[i].invert_yaxis()

    plt.tight_layout()
    plt.show()

    # --- الجزء الثاني: Confusion Matrix (التعلم الآلي - Option 2) ---
    # نزيدو هاد الجزء باش نحققوا شرط الـ Machine Learning
    if y_true is not None and y_pred is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        # هنا نعرضو المصفوفة اللي توري الصح والخطأ في التوقعات
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Neg (0)', 'Pos (2)'])
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title("Matrice de Confusion (Évaluation Machine Learning)")
        plt.show()