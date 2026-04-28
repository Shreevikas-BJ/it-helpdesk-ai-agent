🛠️ AI Helpdesk Agent

This project is an AI-powered IT helpdesk assistant that can classify support tickets, retrieve knowledge base articles, suggest troubleshooting steps, and safely execute diagnostic commands.

🚀 What it does

Classifies tickets (VPN, Wi-Fi, Printer, Password Reset, etc.)

Retrieves and reranks knowledge base articles (RAG)

Suggests fixes (templates or LLM via Ollama)

Executes only safe commands (ping, ipconfig, whoami, echo)

Blocks unsafe commands with a strict safety judge

🏗️ Tech Stack

PyTorch (DistilBERT, CUDA for GPU acceleration)

LangChain + LangGraph

FAISS + Cross-Encoder

Ollama (Mistral LLM)

Streamlit UI

⚡ Quickstart
# Clone repo
git clone https://github.com/your-username/ai-helpdesk-agent.git
cd ai-helpdesk-agent

# Install dependencies
pip install -r requirements.txt

# Train classifier
python scripts/train_classifier.py --data data/helpdesk_train_big.csv --out_dir checkpoints/helpdesk-classifier --epochs 3

# Run Streamlit app
streamlit run ui/app_streamlit.py

📊 Example

Input:
"My VPN disconnects every 10 minutes."

Output:
Category → vpn_issue
Verdict → PASS
Commands → ping 8.8.8.8 -n 3, whoami

Powered by RAG + CUDA acceleration + Ollama to bring AI into IT support.