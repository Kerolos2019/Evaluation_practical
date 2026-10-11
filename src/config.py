import os 

from pathlib import Path 

from dotenv import load_dotenv 

load_dotenv() 

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT_DIR / "datasets"

CHROMA_PERSIST_DIR = ROOT_DIR / ".chroma"

CHROMA_COLLECTION_NAME = "chatbot_knowledge_base"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = int(CHUNK_SIZE * 0.2)  # 20% overlap 


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


OPENAI_CHAT_MODEL = "gpt-5" 


OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

