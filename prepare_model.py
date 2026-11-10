"""Explicit, optional one-time model download. Never called by the app."""
from pathlib import Path
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download an optional multilingual clustering model.')
    parser.add_argument('--output', default='models/paraphrase-multilingual-MiniLM-L12-v2')
    args = parser.parse_args()
    os.environ.setdefault('HF_HOME', str(Path('.model-cache').resolve()))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', trust_remote_code=False)
    model.save(str(Path(args.output)))
    print(Path(args.output).resolve())
