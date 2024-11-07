import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_data(file_path):
    data = pd.read_json(file_path)
    data["combined"] = data['title'] + " " + data["summary"]
    return data

def save_embeddings(data, output_file):
    data['embedding'] = data['combined'].apply(lambda text: model.encode(text, convert_to_tensor=True))
    data['embedding'] = data['embedding'].apply(lambda x: x.cpu().detach().numpy())
    with open(output_file, 'wb') as f:
        pickle.dump(data['embedding'].tolist(), f)

data = load_data("articles.json")
save_embeddings(data, "embeddings.pkl")
