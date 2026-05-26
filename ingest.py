import argparse
from pathlib import Path
from rag.chunker import load_and_chunk
from rag.embedder import embed
from rag.store import upsert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="./docs")
    args = parser.parse_args()
    
    all_chunks = []
    for md_file in Path(args.docs).glob("**/*.md"):
        chunks = load_and_chunk(md_file)
        print(f"{md_file.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    embeddings = embed([c["text"] for c in all_chunks])
    upsert(all_chunks, embeddings)
    print("Done. Upserted to ChromaDB")
    
    
if __name__ == "__main__":
    main()