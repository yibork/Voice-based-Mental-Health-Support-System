# generate_embeddings.py

import os
import json
import pickle
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# File path to save the embeddings
EMBEDDINGS_FILE = "embeddings/embeddings.pkl"

# Load the JSON file with the documents
with open("Psych8k/Alexander_Street_shareGPT_2.0.json", "r") as file:
    json_data = json.load(file)

# Initialize an empty list to hold the documents
documents = []

# Iterate over each item in the JSON data to create Documents
for item in json_data:
    if 'input' in item:
        documents.append(Document(page_content=item['input'], metadata={"source": "input"}))
    if 'output' in item:
        documents.append(Document(page_content=item['output'], metadata={"source": "output"}))

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Generate embeddings using FAISS and save them to a file
def generate_and_save_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        print("Embeddings file already exists.")
    else:
        print("Generating new embeddings...")
        # Create the vector store
        vector = FAISS.from_documents(documents, embedder)

        # Save the vector store (embeddings) to a file
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(vector, f)
        print("Embeddings saved to:", EMBEDDINGS_FILE)

if __name__ == "__main__":
    generate_and_save_embeddings()
