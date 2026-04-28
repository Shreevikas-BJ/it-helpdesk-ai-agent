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

```text
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
