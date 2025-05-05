import pandas as pd
import uuid
from sqlalchemy import create_engine
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from config import *

def extract_data():
    engine = create_engine(DB_URI)
    query = """
    SELECT a.id, a.price, a.description, s.name AS style, u.name AS artist, u.country
    FROM art_works a
    JOIN styles s ON a.style = s.id
    JOIN users u ON a.artist_id = u.id
    WHERE a.description IS NOT NULL;
    """
    df = pd.read_sql(query, engine)
    print(f"✅ Extracted {len(df)} rows.")
    return df

def build_documents(df):
    return [
        Document(
            text=(
                f"Œuvre : {row['description']}\n"
                f"Prix : {row['price']} EUR\n"
                f"Style : {row['style']}\n"
                f"Artiste : {row['artist']} ({row['country']})"
            ),
            metadata={
                "id": row["id"],
                "artist": row["artist"],
                "style": row["style"],
                "country": row["country"],
                "price": row["price"]
            }
        )
        for _, row in df.iterrows()
    ]

def index_documents():
    df = extract_data()
    if df.empty:
        return

    documents = build_documents(df)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    # Création de la collection Qdrant
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_size = len(embed_model.get_text_embedding("test"))
    vector_params = VectorParams(size=vector_size, distance=Distance.COSINE)

    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ Deleted collection: {COLLECTION_NAME}")

    qdrant_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vector_params)
    print(f"📦 Created collection: {COLLECTION_NAME}")

    # Création de l'index et persistance
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    index.storage_context.persist()
    print("✅ Index persisted to ./storage")

if __name__ == "__main__":
    index_documents()
