# cyber_data_loader.py - FIXED VERSION
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Fixed import

def load_cyber_files(folder_path: str = ".") -> list:
    """
    Load all cybersecurity text files from a folder.
    
    Args:
        folder_path (str): Path to folder containing .txt files
        
    Returns:
        list: List of Document objects
    """
    documents = []
    
    # Define which files to load (or load all .txt files)
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    print(f"Found {len(txt_files)} text files")
    
    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc = Document(
            page_content=content,
            metadata={
                "source": file,
                "topic": file.replace('.txt', '').replace('_', ' ').title()
            }
        )
        documents.append(doc)
        print(f"✓ Loaded: {file}")
    
    return documents

def split_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split documents into smaller chunks.
    
    Args:
        documents (list): List of Document objects
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def get_topic_summary(documents: list) -> dict:
    """
    Get summary of topics in the documents.
    
    Args:
        documents (list): List of Document objects
        
    Returns:
        dict: Dictionary with topic counts
    """
    topics = {}
    for doc in documents:
        topic = doc.metadata.get("topic", "Unknown")
        topics[topic] = topics.get(topic, 0) + 1
    
    return topics

if __name__ == "__main__":
    print("=" * 50)
    print("Cybersecurity Data Loader Test")
    print("=" * 50)
    
    # Load files
    docs = load_cyber_files(".")
    
    # Split into chunks
    chunks = split_documents(docs)
    
    # Show summary
    topics = get_topic_summary(docs)
    print("\nTopics loaded:")
    for topic, count in topics.items():
        print(f"  - {topic}: {count} files")
    
    print(f"\n✓ Total chunks ready for vectorstore: {len(chunks)}")