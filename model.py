import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load the articles data
def load_data(file_path):
    data = pd.read_json(file_path)
    data["combined"] = data['title'] + " " + data["summary"]
    return data

# Generate and save embeddings
def save_embeddings(data, output_file):
    data['embedding'] = data['combined'].apply(lambda text: model.encode(text, convert_to_tensor=True))
    # Convert tensors to numpy arrays before saving to avoid issues with pickling
    data['embedding'] = data['embedding'].apply(lambda x: x.cpu().detach().numpy())
    with open(output_file, 'wb') as f:
        pickle.dump(data['embedding'].tolist(), f)

data = load_data("articles.json")
save_embeddings(data, "embeddings.pkl")
