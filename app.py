import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pickle
import torch

# Load the model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load the articles data
def load_data(file_path):
    data = pd.read_json(file_path)
    data["combined"] = data['title'] + " " + data["summary"]
    return data

# Load the embeddings from a pickle file
def load_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        return pickle.load(f)

# Get similar articles based on user query
def get_similar_articles(data, user_query, top_k=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check if CUDA is available
    query_embedding = model.encode(user_query, convert_to_tensor=True).to(device)  # Move query to device

    # Ensure all embeddings are on the same device
    data['embedding'] = data['embedding'].apply(lambda x: x.to(device))  # Move embeddings to the same device

    data['similarity'] = data['embedding'].apply(lambda x: util.pytorch_cos_sim(query_embedding, x).item())
    top_articles = data.nlargest(top_k, 'similarity')[['title', 'summary', 'url', 'similarity']]
    return top_articles

# Main Streamlit app
def main():
    st.set_page_config(page_title="Système de Recommandation d'Articles", layout="wide")
    st.title("Système de Recommandation d'Articles")
    st.write("Entrez une requête pour trouver des articles similaires.")

    # Load data and embeddings
    data = load_data("articles.json")
    embeddings = load_embeddings("embeddings.pkl")
    data['embedding'] = [torch.tensor(e) for e in embeddings]  # Ensure embeddings are in tensor format

    user_query = st.chat_input("Quels articles recherchez-vous ?")

    if user_query:
        top_articles = get_similar_articles(data, user_query)
        st.write("Articles similaires :")
        for index, row in top_articles.iterrows():
            st.subheader(row['title'])
            st.write(row['summary'])
            st.write(f"[Lire l'article]({row['url']})")
            st.write(f"Similarité : {row['similarity']:.4f}")
            st.write("\n")

if __name__ == "__main__":
    main()
