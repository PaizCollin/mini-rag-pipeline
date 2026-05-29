import argparse, os
from openai import OpenAI
from dotenv import load_dotenv
from rag.embedder import embed
from rag.store import query

load_dotenv()

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c["source"]}]\n{c["text"]}" for c in chunks
    )
    return f"""Answer the question belos using ONLY the context provided. 
If the answer isn't in the context, say so.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:""" 

def main():
    parser = argparse.ArgumentParser
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    
    print(f"Querying for: {args.query}")
    q_embedding = embed([args.query])[0]
    chunks = query(q_embedding, top_k=args.top_k)
    
    print(f"Retrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"[{c['distance']:.3f}] {c['source']}")
    
    prompt = build_prompt(args.query, chunks)
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    
    print(response.choices[0].message.content)    

if __name__ == "__main__":
    main()