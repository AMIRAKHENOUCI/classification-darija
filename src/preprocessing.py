import re

def clean_text(text):
    if not isinstance(text, str): 
        return ""
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\sء-ي]', ' ', text)
    text = re.sub("[إأآ]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    
    text = re.sub(r'(.)\1+', r'\1', text)
    
    return " ".join(text.split())