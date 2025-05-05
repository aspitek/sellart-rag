# config.py
import os

DB_URI = os.getenv("DB_URI")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB = os.getenv("REDIS_DB")

