# Cybersecurity NLP Chatbot

An AI-powered cybersecurity question-answering application built with Retrieval-Augmented Generation (RAG), FAISS vector search, Sentence Transformers, Hugging Face Transformers, and Streamlit.

The chatbot retrieves relevant information from a local cybersecurity knowledge base and can optionally use a generative transformer model to produce a clear response.

## Project Interface

![Cybersecurity Chatbot Interface](screenshots/cybersecurity-chatbot-interface.png)

## Features

- Answers cybersecurity questions from a local knowledge base
- Uses semantic search with Sentence Transformer embeddings
- Stores and retrieves document vectors with FAISS
- Supports optional Hugging Face text generation
- Includes a modern Streamlit chat interface
- Provides suggested cybersecurity questions
- Supports rebuilding the knowledge base from the sidebar
- Handles selected general conversational questions
- Runs locally without requiring a paid API

## Covered Cybersecurity Topics

The local knowledge base includes topics such as:

- Authentication and authorization
- JSON Web Tokens
- SQL injection and NoSQL injection
- Cross-site scripting
- Cross-site request forgery
- Denial-of-service and distributed denial-of-service attacks
- Malware and phishing
- Insecure direct object references
- Server-side request forgery
- XML external entity attacks
- Secure coding and web security

## How It Works

1. Cybersecurity text files are loaded as documents.
2. `sentence-transformers/all-MiniLM-L6-v2` converts the documents into embeddings.
3. FAISS stores the embeddings and performs semantic retrieval.
4. The application selects relevant cybersecurity context for the question.
5. `google/flan-t5-base` can generate a natural-language response from the retrieved context.
6. Streamlit displays the conversation in an interactive web interface.

## Technology Stack

- Python
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- Hugging Face Transformers
- PyTorch

## Project Structure

```text
nlp-chatbot/
├── screenshots/
│   └── cybersecurity-chatbot-interface.png
├── app.py
├── build_cyber_vectorstore.py
├── update_vectorstore.py
├── cyber_data_loader.py
├── Authentication.txt
├── authorization.txt
├── common_attacks.txt
├── csrf.txt
├── ddos.txt
├── dos.txt
├── idor.txt
├── JWT
├── malware.txt
├── nosql.txt
├── phishing.txt
├── secure_coding.txt
├── sql_injection.txt
├── ssrf.txt
├── web_security.txt
├── xss.txt
├── xxe.txt
├── requirements.txt
└── README.md
```

The generated FAISS index is stored locally under `vectorstore/faiss_index/` and is excluded from version control.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ahmadshaaban394-cell/nlp-chatbot.git
cd nlp-chatbot
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The first run can take longer because the embedding and transformer models may need to be downloaded and the FAISS knowledge base may need to be built.

## Example Questions

- What is SQL injection?
- Explain cross-site scripting.
- What is the difference between authentication and authorization?
- How does a DDoS attack work?
- What are secure coding best practices?
- What is an IDOR vulnerability?
- Explain SSRF and XXE attacks.

## Future Improvements

- Add automated tests
- Add conversation export
- Add document upload support
- Add source citations to generated answers
- Add evaluation metrics for retrieval quality
- Deploy the application online
- Refactor the application into smaller modules

## Author

**Ahmad Shaaban**

[LinkedIn](https://www.linkedin.com/in/ahmad-shaaban-17b675259/)  
[GitHub](https://github.com/ahmadshaaban394-cell)

## Disclaimer

This project is intended for educational purposes. Cybersecurity information should be used responsibly and only in authorized environments.
