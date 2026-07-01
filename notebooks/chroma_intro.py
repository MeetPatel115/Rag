import chromadb
from pypdf import PdfReader
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader(r"C:\Users\91951\OneDrive\Desktop\pythonProject\Rag\data\ai.pdf")
text = "\n".join(page.extract_text() for page in reader.pages)

def recursive_chunk(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=51,length_function=len)
    
    chunk=text_splitter.split_text(text)
    return chunk
recu_chunk=recursive_chunk(text)
client=chromadb.PersistentClient(path=r"C:\Users\91951\OneDrive\Desktop\pythonProject\Rag\data")
collection =client.get_or_create_collection(name="Aimeet")


collection.add(
    ids=[str(uuid.uuid4()) for _ in recu_chunk],
    documents=recu_chunk,
    metadatas=[{"chunk": i} for i in range(len(recu_chunk))]
)
result=collection.query(query_texts=["What is the primary objective of the proposed language-based digital twin framework?",
    "Which dataset was used to train and evaluate the proposed digital twin model?",    "Why does the paper use a conditional variational autoencoder (cVAE) instead of relying only on text similarity metrics?",
    "How does incorporating participant metadata contribute to generating personalized responses?"],n_results=4)

for i,x in enumerate(result['documents']):
    
    print(f"Query:  {i}")
    print("\n result -".join(x))