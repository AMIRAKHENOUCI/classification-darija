import re
import string
from nltk.corpus import stopwords

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^\w\sء-ي]', ' ', text)

    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)

    replacements = {
        r'\bالل\b': 'الله',
        r'\bولل\b': 'الله',
        r'\bنتم\b': 'انتم',
        r'\bكمش\b': 'ماكمش',
        r'\bاو\b': 'و',
        r'\bهادي\b': 'هذه',
        r'\bهادا\b': 'هذا',
        r'\bهاذ\b': 'هذا'
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    words = text.split()

    stop_words = set(stopwords.words('arabic'))
    custom_stops = {'اللى','اللي','باش','تاع','راه','حنا','انت','انا',
                    'ههه','هههه','ههههه','واش','نتم','كمش','او','يا','ياا',
                    'في','على','مع','من','الى','ل','عن'}
    all_stops = stop_words.union(custom_stops)
    words = [w for w in words if w not in all_stops]

    return " ".join(words)
