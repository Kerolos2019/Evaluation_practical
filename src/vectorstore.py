
import chromadb 
from chromadb.utils import embedding_functions

from functools import lru_cache

from src import config 

@lru_cache(maxsize=1)
def get_collection():

    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.OPENAI_EMBEDDING_MODEL,
    )

    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

