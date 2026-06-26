from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader


reader = PdfReader(r"C:\Users\91951\OneDrive\Desktop\pythonProject\Rag\data\ai.pdf")
text = "\n".join(page.extract_text() for page in reader.pages)

def fixed_text(text):
    
    chunks=[]
    t=''
    temp=0
    for i in text:
        if temp<500:
            t+=i
            temp+=1
        else:
            chunks.append(t+i)
            temp=0
            t=''
    
    return chunks

def recursive_chunk(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=51,length_function=len)
    
    chunk=text_splitter.split_text(text)
    return chunk
def Sentence_splitter(text):
    documents = [Document(text=text)]
    splitter = SentenceSplitter(chunk_size=502, chunk_overlap=68)
    nodes = splitter.get_nodes_from_documents(documents)
    return [node.text for node in nodes]   # ← extract .text from each node

bge=SentenceTransformer("BAAI/bge-base-en-v1.5")   
class retrival:
    
    def __init__(self):
        pass
    
    def encode_bge(self,chunk):
    
        emb_bge = bge.encode(chunk)  
        return {"chunks": chunk, "embeddings": emb_bge}
    
    
    def retrieve(self,query, chunks, chunk_embeddings, embedder, k=3):
        # 1. Embed the query
        q_emb = embedder.encode([query])[0]   # shape (768,)
        
        # 2. Cosine similarity against every chunk
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
        q_norm = np.linalg.norm(q_emb)
        sims = (chunk_embeddings @ q_emb) / (chunk_norms * q_norm)
        
        # 3. Top k indices, highest score first
        top_k_idx = np.argsort(-sims)[:k]
        
        return [(chunks[i], float(sims[i])) for i in top_k_idx]
  
  
# Build chunks with all three strategies
chunks_fixed     = fixed_text(text)
chunks_recursive = recursive_chunk(text)
chunks_sentence  = Sentence_splitter(text)

# Build all three indexes
recu = retrival()
indexes = {
    "fixed":     recu.encode_bge(chunks_fixed),
    "recursive": recu.encode_bge(chunks_recursive),
    "sentence":  recu.encode_bge(chunks_sentence),
}

# Sanity check — see how each strategy chunked the doc
for name, idx in indexes.items():
    print(f"{name:10s} → {len(idx['chunks'])} chunks")

# The query
query = "Which large language model serves as the base model for the language digital twin?"

# Run against all three
print(f"\n{'='*70}")
print(f"QUERY: {query}")
print('='*70)

for strategy, idx in indexes.items():
    results = recu.retrieve(query, idx["chunks"], idx["embeddings"], bge, k=3)
    print(f"\n[{strategy.upper()}]")
    for rank, (chunk, score) in enumerate(results, 1):
        preview = chunk[:300].replace("\n", " ")
        print(f"\n  #{rank} (sim={score:.3f})")
        print(f"  {preview}...")