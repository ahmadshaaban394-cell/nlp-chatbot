# build_cyber_vectorstore.py - FIXED VERSION
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from cyber_data_loader import load_cyber_files, split_documents

print("=" * 60)
print("BUILDING CYBERSECURITY VECTORSTORE")
print("=" * 60)

# Load cybersecurity files
print("\n📁 Loading cybersecurity files...")
documents = load_cyber_files(".")

if not documents:
    print("❌ No documents found! Make sure your .txt files are in the current directory.")
    exit(1)

# Split into chunks
print("\n✂️  Splitting documents into chunks...")
chunks = split_documents(documents)

# Create embeddings
print("\n🔧 Creating embeddings (downloading model if needed)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vectorstore
print("\n🏗️ Building FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save vectorstore
vectorstore_dir = "vectorstore/faiss_index"
os.makedirs(vectorstore_dir, exist_ok=True)
vectorstore.save_local(vectorstore_dir)

print("\n" + "=" * 60)
print("✅ SUCCESS! Cybersecurity vectorstore created!")
print(f"📁 Location: {vectorstore_dir}")
print("=" * 60)