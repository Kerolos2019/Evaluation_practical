import subprocess 
import sys 
from pathlib import Path 

ROOT_DIR = Path(__file__).resolve().parent 

sys.path.insert(0, str(ROOT_DIR)) 

from src.pipeline import run_ingestion 

if __name__ == "__main__":

    run_ingestion()

    streamlit_app = ROOT_DIR / "src" / "app.py"

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(streamlit_app)],
        check=True 
    )

