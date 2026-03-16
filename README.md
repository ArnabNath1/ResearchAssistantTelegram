# 🔬 AI Research Assistant Telegram Bot

A production-grade, AI-powered research assistant that transforms how you interact with scientific literature. Powered by **Groq (Llama 3.3 70B)**, **Astra DB (Vector Memory)**, and **RAG architecture**, this bot provides deep scientific reasoning and paper analysis directly in Telegram.

## 🚀 Key Features

- **📄 Full Paper Analysis**: PDF parsing with PyMuPDF to extract methodology, math, and experiments.
- **🕳️ Research Gap Detection**: Identify unexplored areas in state-of-the-art papers.
- **🧪 Experiment Design**: Generate concrete, testable hypotheses and experimental plans.
- **💡 Novelty Engine**: Suggest original research contributions based on current findings.
- **📚 Automated Lit Reviews**: Pull data from arXiv & Semantic Scholar to synthesize literature.
- **📐 Math Clarifier**: Deep explanation of complex equations and technical notations.
- **❓ RAG-Powered Q&A**: Ask any technical question grounded in your uploaded research papers.
- **🆔 arXiv Integration**: Fetch and analyze papers instantly using just an arXiv ID.

## 🛠️ Technology Stack

- **Backend**: Python (FastAPI)
- **Interface**: Telegram Bot API (`python-telegram-bot`)
- **LLM**: Groq API (Llama-3.3-70b-versatile)
- **Vector DB**: DataStax Astra DB (Vector Store)
- **Embeddings**: `sentence-transformers` (Local)
- **Retrieval**: arXiv API & Semantic Scholar API

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd ResearchAssistant
   ```

2. **Run the setup script** (Windows):
   ```bash
   ./setup.bat
   ```
   *This will create a virtual environment and install all dependencies.*

3. **Configure Environment**:
   Edit the `.env` file with your credentials:
   ```env
   TELEGRAM_BOT_TOKEN=your_token
   GROQ_API_KEY=your_key
   ASTRA_DB_APPLICATION_TOKEN=your_token
   ASTRA_DB_API_ENDPOINT=your_endpoint
   ```

## 🏃 Running the System

Start both the FastAPI server and the Telegram Bot:
```bash
./run.bat
```

## 📖 Bot Commands

- `/start` — Launch the main research menu.
- `/arxiv <id>` — Fetch, download, and analyze an arXiv paper.
- `/search <query>` — Search for papers across multiple sources.
- `/litreview <topic>` — Generate a structured literature review.
- `/qa <question>` — Ask scientific questions (context-aware).
- `/clear` — Wipe conversation history for a fresh start.

---

## 🛡️ License
MIT License. Created for researchers, by AI.
