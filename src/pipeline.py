from src import config 
from src.chunking import fixed_size_chunk
from src.loaders import load_documents
from src.vectorstore import get_collection 


def needs_ingestion() -> bool: 

    collection = get_collection()
    return collection.count() == 0

def run_ingestion() -> None: 

    if not needs_ingestion(): 
        print("Ingestion skipped: knowledge base already exists.")
        return

    print("Starting ingestion of documents into the knowledge base...")

    documents = load_documents(config.DATASETS_DIR)

    ids: list[str] = [] 
    texts: list[str] = []
    metadatas: list[dict] = []

    for doc in documents:

        chunks = fixed_size_chunk(doc.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP) 

        for i, chunk in enumerate(chunks):
            ids.append(f"{doc.source}::{i}") 
            texts.append(chunk)

            metadatas.append({"source": doc.source, "chunk_index": i})

    if not texts: 
        print("No documents found for ingestion.")
        return

    collection = get_collection()  

    batch_size = 100 

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end]
        )



if __name__ == "__main__":
    run_ingestion()