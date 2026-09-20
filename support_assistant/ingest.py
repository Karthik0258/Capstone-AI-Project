import os, glob
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("zepto_policies")

# Add all docs
for f in glob.glob("docs/*.txt"):
    with open(f, "r") as fh:
        text = fh.read()
    emb = model.encode(text).tolist()
    collection.add(documents=[text], embeddings=[emb], ids=[os.path.basename(f)])

print("Ingestion complete: corpus embedded into ChromaDB")
