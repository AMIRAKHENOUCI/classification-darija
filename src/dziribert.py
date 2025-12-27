from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

MODEL_NAME = "alger-ia/dziribert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def encode_texts(texts, max_len=64, batch_size=16):
    model.eval() 
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(embeddings)
        
        print(f"Batch {i//batch_size + 1} done...")

    return np.vstack(all_embeddings)