import os
import re
import shutil
import streamlit as st

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =============================
# PAGE SETTINGS
# =============================
st.set_page_config(
    page_title="Cybersecurity NLP Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================
# PATHS AND MODELS
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_GENERATION_MODEL = "google/flan-t5-base"


# =============================
# CUSTOM CSS DESIGN
# =============================
def apply_custom_css():
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background:
                radial-gradient(circle at 20% 20%, rgba(20, 120, 255, 0.18), transparent 28%),
                radial-gradient(circle at 80% 10%, rgba(0, 255, 200, 0.08), transparent 24%),
                repeating-linear-gradient(
                    90deg,
                    rgba(255,255,255,0.00) 0px,
                    rgba(255,255,255,0.00) 14px,
                    rgba(255,255,255,0.035) 15px,
                    rgba(255,255,255,0.00) 16px
                ),
                linear-gradient(90deg, #02112a 0%, #06265a 45%, #0a3e88 100%);
            color: #f4f7ff;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        /* Important: do NOT hide Streamlit toolbar.
           The toolbar/sidebar button must stay visible. */
        div[data-testid="stToolbar"] {
            visibility: visible !important;
        }

        button[kind="header"] {
            visibility: visible !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(33,37,48,0.96), rgba(26,30,40,0.96));
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {
            color: #f6f8ff;
        }

        .kb-box {
            background: rgba(53, 122, 73, 0.55);
            border: 1px solid rgba(120, 255, 170, 0.18);
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 16px;
            color: #9fffbd;
            font-weight: 600;
        }

        .kb-file {
            display: inline-block;
            padding: 7px 11px;
            border-radius: 10px;
            margin-bottom: 10px;
            background: rgba(8, 16, 28, 0.60);
            border: 1px solid rgba(255,255,255,0.06);
            color: #57ff97;
            font-family: monospace;
            font-size: 0.94rem;
        }

        .model-note {
            background: rgba(0, 180, 140, 0.14);
            border: 1px solid rgba(0, 255, 180, 0.18);
            border-radius: 14px;
            padding: 10px 12px;
            color: #bfffe9;
            margin-top: 10px;
            margin-bottom: 10px;
            font-size: 0.92rem;
        }

        section.main > div.block-container {
            max-width: 760px;
            margin-top: 2rem;
            margin-bottom: 7rem;
            padding: 1.3rem 1.2rem 5.8rem 1.2rem;
            border-radius: 34px;
            border: 1px solid rgba(255,255,255,0.16);
            background:
                linear-gradient(180deg, rgba(26,52,92,0.88), rgba(24,45,79,0.84));
            box-shadow:
                0 18px 55px rgba(0, 0, 0, 0.45),
                inset 0 1px 0 rgba(255,255,255,0.10);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }

        .hero-wrap {
            text-align: center;
            padding-top: 0.2rem;
            padding-bottom: 0.85rem;
        }

        .hero-chip {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            color: #d7e8ff;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.65rem;
        }

        .hero-title {
            font-size: 2.45rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            color: #d4def0;
            font-size: 0.98rem;
            line-height: 1.55;
            margin-bottom: 0.3rem;
        }

        h1, h2, h3, h4, h5, h6,
        p, span, li, label, div {
            color: #f4f7ff;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: #f1f5ff;
            line-height: 1.65;
        }

        [data-testid="stMarkdownContainer"] strong {
            color: #ffffff;
        }

        div[data-testid="stChatMessage"] {
            background: rgba(8, 20, 39, 0.28);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            padding: 8px 8px 4px 8px;
            margin-bottom: 0.65rem;
            backdrop-filter: blur(6px);
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(180deg, rgba(37,50,69,0.95), rgba(24,32,46,0.95));
            color: white;
            font-weight: 600;
            transition: 0.2s ease;
        }

        .stButton > button:hover {
            border-color: rgba(0,255,180,0.40);
            color: #8affcf;
            transform: translateY(-1px);
        }

        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 18px;
            left: 50%;
            transform: translateX(-50%);
            width: min(760px, calc(100% - 2rem));
            z-index: 9999;
        }

        div[data-testid="stChatInput"] > div {
            border-radius: 18px !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            background: rgba(26, 32, 43, 0.95) !important;
            box-shadow: 0 8px 35px rgba(0,0,0,0.32);
        }

        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] input {
            color: #ffffff !important;
        }

        div[data-testid="stChatInput"] textarea::placeholder,
        div[data-testid="stChatInput"] input::placeholder {
            color: #b3bfd4 !important;
        }

        hr {
            border-color: rgba(255,255,255,0.10);
        }

        .footer-note {
            color: #bfc9dc;
            font-size: 0.92rem;
            margin-top: 0.4rem;
        }

        @media (max-width: 768px) {
            section.main > div.block-container {
                margin-top: 1rem;
                margin-bottom: 7rem;
                border-radius: 24px;
                padding-bottom: 5.8rem;
            }

            .hero-title {
                font-size: 1.85rem;
            }

            div[data-testid="stChatInput"] {
                width: calc(100% - 1rem);
                bottom: 10px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_css()


# =============================
# TOPIC ALIASES
# =============================
TOPIC_ALIASES = {
    "sql injection": ["sql injection", "sqli", "sql"],
    "nosql injection": ["nosql injection", "nosql"],
    "xss": ["xss", "cross site scripting", "cross-site scripting"],
    "csrf": ["csrf", "cross site request forgery", "cross-site request forgery"],
    "ddos": ["ddos", "distributed denial of service"],
    "dos": ["dos", "denial of service"],
    "phishing": ["phishing", "fake email", "social engineering"],
    "malware": ["malware", "virus", "worm", "trojan", "ransomware", "spyware", "keylogger"],
    "idor": ["idor", "insecure direct object reference"],
    "ssrf": ["ssrf", "server side request forgery", "server-side request forgery"],
    "xxe": ["xxe", "xml external entity", "xml external entity injection"],
    "jwt": ["jwt", "json web token", "token"],
    "authentication": ["authentication", "login", "brute force", "credential stuffing", "password spraying"],
    "authorization": ["authorization", "access control", "permissions", "roles"],
    "common attacks": ["common attacks", "cybersecurity attacks", "common cybersecurity attacks"],
    "secure coding": ["secure coding", "secure coding practices", "input validation", "output encoding", "prepared statements"],
    "web security": ["web security", "web application security", "application security", "api security"],
}


# =============================
# EXACT FILE MAP
# =============================
TOPIC_FILE_MAP = {
    "sql injection": ["sql_injection.txt"],
    "nosql injection": ["nosql.txt"],
    "xss": ["xss.txt"],
    "csrf": ["csrf.txt"],
    "ddos": ["ddos.txt"],
    "dos": ["dos.txt"],
    "phishing": ["phishing.txt"],
    "malware": ["malware.txt"],
    "idor": ["idor.txt"],
    "ssrf": ["ssrf.txt"],
    "xxe": ["xxe.txt"],
    "jwt": ["JWT", "jwt.txt"],
    "authentication": ["Authentication.txt", "authentication.txt"],
    "authorization": ["authorization.txt", "Authorization.txt"],
    "common attacks": ["common_attacks.txt"],
    "secure coding": ["secure_coding.txt"],
    "web security": ["web_security.txt"],
}


# =============================
# SECURITY KEYWORDS
# =============================
SECURITY_KEYWORDS = [
    "sql", "nosql", "xss", "csrf", "ddos", "dos", "phishing", "malware",
    "virus", "worm", "trojan", "ransomware", "spyware", "keylogger",
    "idor", "ssrf", "xxe", "jwt", "token", "authentication", "authorization",
    "login", "password", "brute force", "credential stuffing", "access control",
    "secure coding", "web security", "application security", "api security",
    "vulnerability", "attack", "cyber", "cybersecurity", "hacker", "exploit",
    "session", "cookie", "encryption", "hash", "firewall", "botnet",
    "confidentiality", "integrity", "availability", "security plan",
]


# =============================
# NORMALIZATION
# =============================
def normalize_query(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_security_question(query):
    q = normalize_query(query)

    for word in SECURITY_KEYWORDS:
        if word in q:
            return True

    return False


# =============================
# TEXT HELPERS
# =============================
def clean_text(text):
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def get_txt_files():
    ignored = {"requirements.txt"}
    files = []

    for file in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, file)

        if os.path.isfile(path):
            lower = file.lower()

            if lower.endswith(".txt") and lower not in ignored:
                files.append(file)

            if lower == "jwt":
                files.append(file)

    return sorted(files)


def file_to_topic(filename):
    name = filename.replace(".txt", "")
    name = name.replace("_", " ")

    if name.lower() == "jwt":
        return "JWT"

    return name.title()


def normalize_topic_title(topic):
    topic = topic.strip()

    replacements = {
        "Sql Injection": "SQL Injection",
        "Nosql": "NoSQL Injection",
        "Nosql Injection": "NoSQL Injection",
        "Xss": "XSS",
        "Csrf": "CSRF",
        "Ddos": "DDoS",
        "Dos": "DoS",
        "Idor": "IDOR",
        "Ssrf": "SSRF",
        "Xxe": "XXE",
        "Jwt": "JWT",
    }

    return replacements.get(topic, topic)


def detect_topic(query):
    q = normalize_query(query)

    if not is_security_question(query):
        return "general"

    if "nosql injection" in q or re.search(r"\bnosql\b", q):
        return "nosql injection"

    if "sql injection" in q or re.search(r"\bsqli\b", q) or re.search(r"\bsql\b", q):
        return "sql injection"

    for topic, aliases in TOPIC_ALIASES.items():
        if topic in ["sql injection", "nosql injection"]:
            continue

        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return topic

    return "security_general"


def load_documents():
    files = get_txt_files()
    documents = []

    for file in files:
        path = os.path.join(BASE_DIR, file)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = clean_text(f.read())
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                content = clean_text(f.read())

        if not content:
            continue

        topic = normalize_topic_title(file_to_topic(file))

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": file,
                    "topic": topic,
                    "filename": file,
                },
            )
        )

    return documents


# =============================
# HUGGING FACE EMBEDDINGS
# =============================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# =============================
# HUGGING FACE TRANSFORMER MODEL
# =============================
@st.cache_resource
def get_hf_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_GENERATION_MODEL)
        return tokenizer, model
    except Exception as e:
        st.warning(f"Hugging Face Transformer could not be loaded: {e}")
        return None, None


def is_bad_generation(answer):
    if not answer:
        return True

    cleaned = answer.strip()

    if len(cleaned) < 8:
        return True

    bad_outputs = [
        "idx",
        "jimmy taylor",
        "none",
        "n/a",
        "unknown",
        "answer:",
        "question:",
    ]

    if cleaned.lower() in bad_outputs:
        return True

    if len(cleaned.split()) <= 2:
        return True

    return False


def hf_generate(prompt, max_new_tokens=180):
    tokenizer, model = get_hf_model()

    if tokenizer is None or model is None:
        return None

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if is_bad_generation(answer):
            return None

        return answer

    except Exception:
        return None


def generate_general_answer(query):
    prompt = f"""
You are a friendly assistant inside a student NLP chatbot project.
The user is asking a general conversation question, not a cybersecurity question.

Answer naturally in English.
Be friendly, clear, and short.
Do not invent a personal human name.
Do not answer with random names.
Do not answer with one word.
If asked your name, say you are a Cybersecurity NLP Chatbot.
If asked how you are, say you are doing well and ready to help.
If the question is casual, answer casually.

User question:
{query}

Final answer:
"""

    answer = hf_generate(prompt, max_new_tokens=120)

    if answer:
        return answer

    second_prompt = f"""
Answer this general chatbot question in a friendly complete sentence.

Question: {query}

Answer:
"""

    answer = hf_generate(second_prompt, max_new_tokens=80)

    if answer:
        return answer

    return (
        "I am doing well and ready to help you. "
        "You can ask me a general question or a cybersecurity question."
    )


def build_vectorstore(force_rebuild=False):
    embeddings = get_embeddings()

    if force_rebuild and os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)

    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    documents = load_documents()

    if not documents:
        return None

    vectorstore = FAISS.from_documents(documents, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    return vectorstore


def keyword_score(query, doc):
    q = normalize_query(query)
    text = doc.page_content.lower()
    topic = doc.metadata.get("topic", "").lower()
    source = doc.metadata.get("source", "").lower()

    score = 0
    words = re.findall(r"[a-zA-Z0-9]+", q)

    for word in words:
        if len(word) > 2 and word in text:
            score += 1

    detected = detect_topic(query)

    if detected and detected not in ["general", "security_general"]:
        for word in detected.replace("-", " ").split():
            if word in topic or word in source or word in text:
                score += 5

    return score


def retrieve_documents(query, vectorstore):
    detected_topic = detect_topic(query)

    if detected_topic == "general":
        return []

    all_docs = load_documents()

    if detected_topic in TOPIC_FILE_MAP:
        wanted_files = TOPIC_FILE_MAP[detected_topic]
        matched_docs = []

        for doc in all_docs:
            source = doc.metadata.get("source", "")
            if source in wanted_files:
                matched_docs.append(doc)

        if matched_docs:
            return matched_docs

    if detected_topic == "security_general":
        wanted_files = ["common_attacks.txt", "secure_coding.txt", "web_security.txt"]
        matched_docs = []

        for doc in all_docs:
            source = doc.metadata.get("source", "")
            if source in wanted_files:
                matched_docs.append(doc)

        if matched_docs:
            return matched_docs

    try:
        vector_docs = vectorstore.similarity_search(query, k=4)
    except Exception:
        vector_docs = []

    if not vector_docs:
        return []

    ranked = sorted(
        vector_docs,
        key=lambda d: keyword_score(query, d),
        reverse=True,
    )

    return ranked[:3]


def docs_to_context(docs, max_chars=3000):
    context_parts = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        topic = doc.metadata.get("topic", "unknown")
        text = clean_text(doc.page_content)

        context_parts.append(f"Source: {source}\nTopic: {topic}\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    if len(context) > max_chars:
        context = context[:max_chars]

    return context


def generate_transformer_answer(query, docs):
    if not docs:
        return None

    context = docs_to_context(docs, max_chars=3000)

    prompt = f"""
You are a cybersecurity teaching assistant.
Answer the cybersecurity question using only the context below.
Write a clear structured answer with headings and bullet points.
Do not invent information outside the context.
Use simple student-friendly English.

Question:
{query}

Context:
{context}

Final answer:
"""

    answer = hf_generate(prompt, max_new_tokens=320)

    if answer and len(answer) >= 25:
        return answer

    return None


def extract_field(text, field):
    pattern = rf"{field}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+?:|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        return ""

    return clean_text(match.group(1))


def bulletize(text):
    lines = []

    for line in text.splitlines():
        line = line.strip()
        line = line.strip("-•").strip()

        if line:
            lines.append(line)

    return lines


def remove_duplicate_title_from_definition(definition, topic):
    definition = clean_text(definition)
    topic_clean = topic.strip().lower()

    lines = definition.splitlines()

    while lines and lines[0].strip().lower() == topic_clean:
        lines.pop(0)

    definition = "\n".join(lines).strip()

    definition = re.sub(
        rf"^Definition:\s*{re.escape(topic)}\s*",
        "",
        definition,
        flags=re.IGNORECASE,
    )

    definition = re.sub(r"^Definition:\s*", "", definition, flags=re.IGNORECASE)

    return clean_text(definition)


def make_classic_answer(query, docs):
    if not docs:
        if detect_topic(query) == "general":
            return generate_general_answer(query)

        return (
            "I could not find a clear cybersecurity topic for your question.\n\n"
            "Try asking questions like:\n"
            "- What is SQL Injection?\n"
            "- Explain XSS.\n"
            "- What is CSRF?\n"
            "- What is phishing?\n"
            "- What is the difference between authentication and authorization?"
        )

    main_doc = docs[0]
    text = clean_text(main_doc.page_content)

    source = main_doc.metadata.get("source", "unknown")
    topic = normalize_topic_title(main_doc.metadata.get("topic", "Cybersecurity Topic"))

    section_names = [
        "How it works",
        "Common attack locations",
        "Common locations",
        "Common vulnerable features",
        "Types",
        "Structure",
        "Header",
        "Payload",
        "Signature",
        "Common claims",
        "Security risks",
        "Risks",
        "Targets",
        "Common targets",
        "Impact",
        "Example",
        "Why it is dangerous",
        "Warning signs",
        "Prevention",
        "Best practices",
        "Secure coding practices",
        "Difference from NoSQL Injection",
        "Difference from SQL Injection",
        "Difference between XSS and CSRF",
        "Difference between CSRF and XSS",
        "Difference between DoS and DDoS",
        "Difference between authentication and authorization",
        "Related terms",
    ]

    section_regex = "|".join(re.escape(s) for s in section_names)

    definition = re.split(
        rf"\n(?:{section_regex}):",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    definition = remove_duplicate_title_from_definition(definition, topic)

    fields = {
        "How it works": extract_field(text, "How it works"),
        "Types": extract_field(text, "Types"),
        "Structure": extract_field(text, "Structure"),
        "Security risks": extract_field(text, "Security risks"),
        "Risks": extract_field(text, "Risks"),
        "Targets": extract_field(text, "Targets"),
        "Common targets": extract_field(text, "Common targets"),
        "Impact": extract_field(text, "Impact"),
        "Example": extract_field(text, "Example"),
        "Why it is dangerous": extract_field(text, "Why it is dangerous"),
        "Warning signs": extract_field(text, "Warning signs"),
        "Prevention": extract_field(text, "Prevention"),
        "Best practices": extract_field(text, "Best practices"),
    }

    answer = f"### {topic}\n\n"

    if definition:
        answer += f"**Definition:**\n{definition}\n\n"

    for label, content in fields.items():
        if not content:
            continue

        answer += f"**{label}:**\n"

        items = bulletize(content)

        if len(items) > 1:
            for item in items:
                answer += f"- {item}\n"
            answer += "\n"
        else:
            answer += f"{content}\n\n"

    answer += f"**Source:** `{source}`"

    if len(docs) > 1:
        extra_sources = []

        for doc in docs[1:]:
            s = doc.metadata.get("source", "unknown")

            if s != source and s not in extra_sources:
                extra_sources.append(s)

        if extra_sources:
            answer += "\n\n**Related sources:** "
            answer += ", ".join(f"`{s}`" for s in extra_sources)

    return answer


def make_answer(query, docs, use_transformer=True):
    detected_topic = detect_topic(query)

    if detected_topic == "general":
        return generate_general_answer(query)

    if use_transformer and docs:
        transformer_answer = generate_transformer_answer(query, docs)

        if transformer_answer:
            sources = []

            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                if source not in sources:
                    sources.append(source)

            transformer_answer += "\n\n**Sources:** "
            transformer_answer += ", ".join(f"`{source}`" for source in sources)
            transformer_answer += f"\n\n_Model used: Hugging Face Transformer `{HF_GENERATION_MODEL}`_"

            return transformer_answer

    return make_classic_answer(query, docs)


# =============================
# COMPLEX / COMPARISON ANSWERS
# =============================
def answer_comparison(query):
    q = normalize_query(query)

    if detect_topic(query) == "general":
        return None

    if "security plan" in q or ("protect" in q and "web application" in q):
        return """
### Security Plan to Protect a Small Web Application

To protect a small web application, the system should use several security layers.

---

### 1. Protect against SQL Injection

**Risk:**  
Attackers may inject malicious SQL code into input fields or URL parameters to read, change, or delete database data.

**Prevention:**
- Use prepared statements
- Use parameterized queries
- Validate all user inputs
- Use least privilege database accounts
- Do not show detailed database errors to users
- Keep database software updated

---

### 2. Protect against XSS

**Risk:**  
Attackers may inject malicious scripts into web pages viewed by users.

**Prevention:**
- Escape output data
- Validate user input
- Sanitize HTML input
- Use Content Security Policy
- Avoid unsafe JavaScript functions like `innerHTML`
- Use HttpOnly and Secure cookie flags

---

### 3. Protect against CSRF

**Risk:**  
Attackers may trick a logged-in user into sending unwanted requests to the website.

**Prevention:**
- Use CSRF tokens
- Use SameSite cookies
- Verify request origin
- Require confirmation for sensitive actions
- Do not use GET requests for state-changing actions

---

### 4. Protect against Phishing

**Risk:**  
Attackers may trick users into entering passwords or sensitive information on fake pages.

**Prevention:**
- Train users to recognize suspicious emails and links
- Use multi-factor authentication
- Verify sender identity
- Avoid clicking unknown links
- Report suspicious messages
- Use email filtering

---

### 5. Protect against DDoS

**Risk:**  
Attackers may flood the website with traffic from many devices until it becomes slow or unavailable.

**Prevention:**
- Use rate limiting
- Use traffic filtering
- Use firewalls
- Use CDN or DDoS protection services
- Use load balancing
- Monitor abnormal traffic

---

### Conclusion

A secure web application needs input validation, output encoding, prepared statements, CSRF protection, strong authentication, DDoS protection, and user awareness against phishing.

**Sources:** `secure_coding.txt`, `web_security.txt`, `sql_injection.txt`, `xss.txt`, `csrf.txt`, `phishing.txt`, `ddos.txt`
"""

    if "login" in q and "comments" in q and "file upload" in q and "user profile" in q:
        return """
### Cybersecurity Risks for a Website with Login, Comments, File Upload, and User Profile Pages

A website with login, comments, file upload, and user profile pages has several important cybersecurity risks because each feature accepts user input or handles sensitive data.

---

### 1. Login Page Risks

**Possible risks:**
- Brute force attacks
- Credential stuffing
- Weak passwords
- Phishing
- Session hijacking
- Account enumeration

**Prevention:**
- Use strong password policies
- Hash and salt passwords
- Use multi-factor authentication
- Rate limit login attempts
- Lock accounts after many failed attempts
- Use HTTPS
- Protect session cookies with HttpOnly, Secure, and SameSite flags

---

### 2. Comments Page Risks

**Possible risks:**
- Cross-Site Scripting XSS
- Stored XSS
- Spam content
- Malicious links

**Prevention:**
- Validate user input
- Escape output data
- Sanitize HTML content
- Use Content Security Policy
- Avoid unsafe JavaScript functions like innerHTML

---

### 3. File Upload Page Risks

**Possible risks:**
- Malware upload
- Uploading executable files
- Large file abuse
- File overwrite attacks
- Path traversal
- Server compromise

**Prevention:**
- Validate file type
- Limit file size
- Rename uploaded files
- Store files outside executable folders
- Scan files for malware
- Allow only safe extensions like PDF, JPG, PNG
- Do not trust only the file extension

---

### 4. User Profile Page Risks

**Possible risks:**
- IDOR vulnerabilities
- Broken access control
- Unauthorized access to other profiles
- Data leakage
- XSS in profile fields

**Prevention:**
- Verify object ownership
- Check authorization on the server side
- Do not rely only on frontend restrictions
- Validate and escape profile inputs
- Apply least privilege

---

### Conclusion

This website needs strong authentication, secure input handling, safe file upload validation, proper authorization checks, and protection against XSS, CSRF, SQL Injection, IDOR, and malware uploads.

**Sources:** `Authentication.txt`, `xss.txt`, `idor.txt`, `secure_coding.txt`, `web_security.txt`
"""

    if "authentication" in q and "authorization" in q:
        return """
### Difference between Authentication and Authorization

**Authentication:**  
Authentication verifies the identity of the user.  
It answers the question: **Who are you?**

Examples:
- Login using email and password
- Fingerprint authentication
- One-time password
- Multi-factor authentication

**Authorization:**  
Authorization decides what the authenticated user is allowed to access.  
It answers the question: **What are you allowed to do?**

Examples:
- A normal user cannot access the admin dashboard
- A student can view only their own grades
- A parent can view only their child's data
- An admin can manage users

**Main difference:**  
Authentication happens first to verify identity. Authorization happens after login to control permissions and access.

**Sources:** `Authentication.txt`, `authorization.txt`
"""

    if "stored xss" in q and "reflected xss" in q and "dom" in q:
        return """
### Difference between Stored XSS, Reflected XSS, and DOM-based XSS

**Stored XSS:**  
The malicious script is saved on the server, for example inside comments, messages, or user profiles.

**Reflected XSS:**  
The malicious script is not stored. It is sent through a link or request and reflected immediately in the page response.

**DOM-based XSS:**  
The vulnerability happens in client-side JavaScript using unsafe data from the URL or browser source.

**Main difference:**  
Stored XSS is saved on the server, Reflected XSS comes from the request, and DOM-based XSS happens inside the browser's JavaScript.

**Prevention:**
- Validate input
- Escape output
- Sanitize HTML
- Avoid unsafe JavaScript functions like `innerHTML`
- Use Content Security Policy

**Source:** `xss.txt`
"""

    if "xss" in q and ("session hijacking" in q or "account takeover" in q):
        return """
### How XSS Can Lead to Session Hijacking and Account Takeover

XSS allows attackers to inject malicious scripts into a trusted website.  
When a victim opens the infected page, the script runs inside the victim's browser.

**How it can lead to session hijacking:**
- The script may steal cookies or session tokens.
- If the attacker gets the session token, they may use it to impersonate the victim.
- This can allow the attacker to access the account without knowing the password.

**How it can lead to account takeover:**
- The attacker can perform actions as the victim.
- The attacker may change account settings.
- The attacker may steal sensitive information.

**Prevention:**
- Escape output data
- Use HttpOnly cookies
- Use Secure and SameSite cookie flags
- Validate user input
- Use Content Security Policy

**Source:** `xss.txt`
"""

    if "jwt" in q and "encoded" in q and "encrypted" in q:
        return """
### Why JWT Payload Is Encoded but Not Encrypted

A JWT payload is usually **Base64URL encoded**, not encrypted.

**Encoded means:**  
The data is converted into another format so it can be safely transmitted.  
Anyone who has the token can decode and read the payload.

**Encrypted means:**  
The data is protected so only someone with the correct key can read it.

**Why this is important:**  
Developers should not store sensitive information inside a JWT payload, such as passwords, private data, or secret keys.

**Best practices:**
- Do not put sensitive data in the payload
- Use HTTPS
- Validate the signature
- Set expiration time
- Use strong signing secrets
- Reject unsigned tokens

**Source:** `JWT`
"""

    if "phishing" in q and ("spear" in q or "whaling" in q or "smishing" in q or "vishing" in q):
        return """
### Difference Between Phishing, Spear Phishing, Whaling, Smishing, and Vishing

**Phishing:**  
A general social engineering attack that tricks users into revealing sensitive information.

**Spear phishing:**  
A targeted phishing attack aimed at a specific person or organization.

**Whaling:**  
A phishing attack targeting high-level people such as managers or administrators.

**Smishing:**  
Phishing through SMS or text messages.

**Vishing:**  
Phishing through voice calls.

**Prevention:**
- Verify sender identity
- Avoid suspicious links
- Use multi-factor authentication
- Report suspicious messages
- Train users to recognize phishing signs

**Source:** `phishing.txt`
"""

    if ("dos" in q and "ddos" in q) or ("difference" in q and "denial" in q):
        return """
### Difference between DoS and DDoS

**DoS:**  
Denial of Service uses one system or one source to overwhelm a server or network.

**DDoS:**  
Distributed Denial of Service uses many systems, often a botnet, to flood the target with traffic.

**Main difference:**  
DDoS is stronger and harder to stop because the attack comes from many devices, while DoS usually comes from one source.

**Sources:** `dos.txt`, `ddos.txt`
"""

    return None


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## Knowledge Base")

    txt_files = get_txt_files()

    if txt_files:
        st.markdown(
            f"<div class='kb-box'>{len(txt_files)} files loaded</div>",
            unsafe_allow_html=True,
        )
        for file in txt_files:
            st.markdown(
                f"<div class='kb-file'>{file}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.error("No text files found in the app folder.")

    st.divider()

    st.markdown("### AI Model")
    st.markdown(
        f"""
        <div class="model-note">
        Embeddings: <b>{EMBEDDING_MODEL}</b><br>
        Transformer: <b>{HF_GENERATION_MODEL}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    use_transformer = st.toggle(
        "Use Hugging Face Transformer",
        value=True,
        help="If enabled, the app uses the Hugging Face model to generate answers.",
    )

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Rebuilding vectorstore..."):
            st.session_state.vectorstore = build_vectorstore(force_rebuild=True)
        st.success("Knowledge base rebuilt successfully.")
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### Try these questions")
    st.code("How are you?")
    st.code("What is your name?")
    st.code("What is SQL Injection?")
    st.code("Explain XSS.")
    st.code("What is the difference between authentication and authorization?")
    st.code("Create a security plan to protect a small web application from SQL Injection, XSS, CSRF, phishing, and DDoS.")


# =============================
# LOAD VECTORSTORE
# =============================
if "vectorstore" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.vectorstore = build_vectorstore(force_rebuild=False)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am a cybersecurity NLP chatbot using FAISS, "
                "Sentence Transformers, and Hugging Face Transformers."
            ),
        }
    ]


# =============================
# HERO SECTION
# =============================
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-chip">AI POWERED KNOWLEDGE BASE</div>
        <div class="hero-title">Cybersecurity Chat GPT</div>
        <div class="hero-subtitle">
            Ask general questions or cybersecurity questions using RAG, FAISS, Sentence Transformers, and Hugging Face Transformers.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================
# DISPLAY CHAT HISTORY
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =============================
# USER INPUT
# =============================
user_input = st.chat_input("Send a message")

if user_input:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            topic_type = detect_topic(user_input)

            if topic_type == "general":
                answer = generate_general_answer(user_input)
            else:
                comparison = answer_comparison(user_input)

                if comparison:
                    answer = comparison
                elif st.session_state.vectorstore is None:
                    answer = (
                        "No knowledge base found. "
                        "Make sure the `.txt` files are inside the `app` folder."
                    )
                else:
                    docs = retrieve_documents(user_input, st.session_state.vectorstore)
                    answer = make_answer(user_input, docs, use_transformer=use_transformer)

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


st.divider()
st.markdown(
    "<div class='footer-note'>Cybersecurity NLP Chatbot | RAG using FAISS + Sentence Transformers + Hugging Face Transformers</div>",
    unsafe_allow_html=True,
)