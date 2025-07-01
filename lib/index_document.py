import pandas as pd
import uuid
from sqlalchemy import create_engine
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.embeddings.openai import OpenAIEmbedding
from config import *

def extract_data():
    engine = create_engine("postgresql+psycopg2://admin:admin%402025@147.79.114.72:31432/sellart")
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
    #embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

    # Création de la collection Qdrant
    qdrant_client = QdrantClient(host="localhost", port=6333)
    vector_size = len(embed_model.get_text_embedding("test"))
    vector_params = VectorParams(size=vector_size, distance=Distance.COSINE)

    if qdrant_client.collection_exists("sellart_artworks"):
        qdrant_client.delete_collection("sellart_artworks")
        print(f"🗑️ Deleted collection: sellart_artworks")

    qdrant_client.create_collection(collection_name="sellart_artworks", vectors_config=vector_params)
    print(f"📦 Created collection: sellart_artworks")

    # Création de l'index et persistance
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="sellart_artworks")
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
