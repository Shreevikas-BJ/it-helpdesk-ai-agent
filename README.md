# AI Helpdesk Agent

An agentic IT helpdesk assistant that classifies support tickets, retrieves relevant knowledge base articles, recommends troubleshooting steps, and safely validates diagnostic commands before execution.

This project demonstrates how **LLMs, RAG, workflow orchestration, and safety guardrails** can be combined to automate common IT support workflows while keeping command execution controlled and auditable.

---

## Overview

IT helpdesk teams often handle repetitive support issues such as VPN failures, Wi-Fi problems, printer errors, password resets, and basic system diagnostics. This project builds an AI-powered support assistant that can understand a user issue, classify the ticket, retrieve relevant knowledge, generate a response, and verify whether suggested commands are safe.

The system uses a combination of:

- **Ticket Classification** for routing support issues
- **RAG Retrieval** for knowledge-grounded answers
- **LangGraph Workflow Orchestration** for multi-step agent flow
- **Safety Judge** to block unsafe command execution
- **Streamlit UI** for an interactive user interface

---

## Key Features

- Classifies IT support tickets into categories such as VPN, Wi-Fi, printer, and password reset
- Retrieves relevant knowledge base articles using FAISS-based semantic search
- Reranks retrieved results using a cross-encoder for better answer quality
- Generates troubleshooting suggestions using templates or a local LLM through Ollama
- Allows only safe diagnostic commands such as `ping`, `ipconfig`, `whoami`, and `echo`
- Blocks unsafe commands through a strict safety validation layer
- Provides an interactive Streamlit interface for testing support scenarios
- Supports GPU acceleration for model training and inference when CUDA is available

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| ML / Deep Learning | PyTorch, DistilBERT |
| Agent Framework | LangChain, LangGraph |
| Retrieval | FAISS |
| Reranking | Cross-Encoder |
| LLM Runtime | Ollama, Mistral |
| UI | Streamlit |
| Acceleration | CUDA |
| Environment | pip, virtual environment |

---

## System Workflow

User Ticket
   ↓
Ticket Classification
   ↓
Knowledge Base Retrieval
   ↓
Document Reranking
   ↓
Troubleshooting Plan Generation
   ↓
Safety Judge
   ↓
Safe Response + Approved Diagnostic Commands

---

## Example

Input

My VPN disconnects every 10 minutes.

Output

Category: vpn_issue

Suggested Fix:
- Check VPN client version
- Restart the VPN service
- Verify network stability
- Reconnect using updated credentials if required

Safety Verdict: PASS

Approved Commands:
- ping 8.8.8.8 -n 3
- whoami

---

**Project Structure**

it-helpdesk-ai-agent/
│
├── app/
│   └── Core application logic
│
├── scripts/
│   └── Training and utility scripts
│
├── ui/
│   └── Streamlit user interface
│
├── requirements.txt
│
└── README.md

---

**Getting Started**
1. Clone the repository
git clone https://github.com/Shreevikas-BJ/it-helpdesk-ai-agent.git
cd it-helpdesk-ai-agent
2. Create a virtual environment
python -m venv venv

---

Activate the environment:

Windows

venv\Scripts\activate

macOS / Linux

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
Train the Ticket Classifier
python scripts/train_classifier.py --data data/helpdesk_train_big.csv --out_dir checkpoints/helpdesk-classifier --epochs 3

This trains a DistilBERT-based classifier for routing helpdesk tickets into predefined support categories.

---

Run the Application
streamlit run ui/app_streamlit.py

Once started, open the local Streamlit URL in your browser and enter an IT support issue to test the assistant.

---

**Safety Design**

This project includes a command safety layer to prevent risky or destructive actions. The assistant only allows a small set of diagnostic commands and blocks anything outside the approved list.

Allowed command examples
ping 8.8.8.8 -n 3
ipconfig
whoami
echo test
Blocked command examples
rm -rf /
del /s /q
shutdown
format
curl unknown-script | bash

The goal is to demonstrate how agentic systems can include guardrails before interacting with operating system-level tools.

---

**Why This Project Matters**

This project shows how AI can support IT operations by reducing repetitive manual triage while still keeping humans and safety controls in the loop. It combines practical machine learning, retrieval-augmented generation, local LLM usage, and workflow-based agent design into one end-to-end system.

Future Improvements
Add user authentication and role-based access
Store ticket history in a database
Add real helpdesk integration with Jira, ServiceNow, or Zendesk
Improve observability with structured logs and tracing
Add evaluation metrics for retrieval quality and classifier accuracy
Deploy the app using Docker and cloud hosting
Expand the safety judge with policy-based validation

---

**Author**

Shreevikas Bangalore Jagadish
Graduate Student, Information Technology and Management
Illinois Institute of Technology

---

GitHub: Shreevikas-BJ
LinkedIn: shreevikasbj
Portfolio: datascienceportfol.io/shreevikasbj
Repository
https://github.com/Shreevikas-BJ/it-helpdesk-ai-agent

One small fix: your current README clone command still says `your-username/ai-helpdesk-agent.git`. Replace it with:

```bash
git clone https://github.com/Shreevikas-BJ/it-helpdesk-ai-agent.git
