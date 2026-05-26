import argparse, os
import anthropic
from dotenv import load_dotenv
from rag.embedder import embed
from rag.store import query

load_dotenv()

def build_prompt(query: str, chunks: list[dict]) -> str:
    pass

def main():
    pass

if __name__ == "__main__":
    main()