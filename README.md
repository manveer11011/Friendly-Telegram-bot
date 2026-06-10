# Dual-Mode Friendly Telegram Bot (Local GPU or Gemini Cloud API)

Welcome! This is a modern, high-performance, asynchronous Telegram bot that supports two execution modes, controlled dynamically via environment variables:

1. **Local Mode (`MODE=local`)**: Runs entirely offline on your local GPU/CPU. It is powered by **Qwen3-4B** for chat generation and backed by a local **FAISS Vector Database** + **SQLite metadata store** using a local `all-MiniLM-L6-v2` sentence-transformer model (384 dimensions).
2. **API Mode (`MODE=api`)**: Runs online using the **Gemini Developer API** for fast, resource-efficient chat generation, and uses Gemini's embeddings (3072 dimensions) for FAISS + SQLite semantic memory without requiring heavy local GPU resources.

---

## Key Features

*   **Real-Time Streaming Responses**: Displays the bot's response chunk-by-chunk in real-time with a dynamic typing cursor (`▌`) to reduce perceived latency.
*   **Dual Mode Configuration**: Instantly switches between local offline GPU inference and online cloud API text/embeddings using the `MODE` setting in `.env`.
*   **Unified SQLite Database & Isolated Indexes**: Prevents FAISS dimension conflicts by storing embeddings in mode-specific FAISS files while logging conversation metadata in a single shared SQLite database:
    *   **Shared SQLite Metadata**: Saves all metadata to `bot_history.db` with a `mode` column (`api` or `local`) and mapping `faiss_id`.
    *   **Local Index**: Saves local 384-dimensional vector embeddings to `bot_history_local.index`.
    *   **API Index**: Saves API 3072-dimensional vector embeddings to `bot_history_api.index`.
*   **Personalized Greetings**: Supports `/adddetail` to collect the user's full name, saving it to SQLite and injecting it into chatbot system instructions.
*   **Asynchronous Background Logging**: Computes embeddings and saves transaction logs asynchronously on a background worker thread (`asyncio.create_task` + `asyncio.to_thread`) for lag-free performance.
*   **Warm Chatbot Personality**: Configured via custom system instructions to behave like a close, warm, and supportive friend who includes multiple expressive emojis in every response.

---

## Setup & Installation

### Prerequisites
Make sure you have **Python 3.12** installed on your Windows system (recommended for stable CUDA PyTorch wheels).

### 1. Initialize Environment
In the project directory, initialize a Python virtual environment:
```powershell
# Create virtual environment using Python 3.12
py -3.12 -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
Install PyTorch with CUDA 12.1 support, Hugging Face transformers, and Gemini integration packages:
```powershell
pip install -r requirements.txt
```

### 3. Configure `.env`
Create or update a `.env` file in the root directory:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
MODE=local
```

#### Settings Details:
*   `TELEGRAM_BOT_TOKEN`: Token obtained from **@BotFather** on Telegram.
*   `GEMINI_API_KEY`: API key from Google AI Studio (only required if `MODE=api`).
*   `MODE`: Either `local` (offline GPU/CPU execution) or `api` (online cloud API execution).
*   **Model Path**: The model is loaded directly from your local disk path: `D:\models\Qwen3-4B`

---

## Running the Bot

Start the bot locally:
```powershell
.venv\Scripts\python bot.py
```

*   **Local Mode Startup**: Takes about 15–30 seconds during startup to load Qwen checkpoint shards and SentenceTransformers into GPU memory.
*   **API Mode Startup**: Launches instantly since no heavy local models are loaded.

---

## Bot Commands

*   `/start` - Initializes the chat, greets you (personally, if your name is saved), and lists available commands.
*   `/help` - Shows general instruction guidelines.
*   `/adddetail` - Prompts you to enter your full name so the bot can personalize its replies.
*   `/cancel` - Cancels the active `/adddetail` input prompt.
*   `/history` - Retrieves your last 10 logged conversation entries from the FAISS database for the active mode.

---

## File Structure

```text
├── .env                  # Private credentials, active mode selection, and API keys
├── .gitignore            # Excludes venv, cached files, and database files
├── requirements.txt      # Pinned dependency packages
├── bot.py                # Main Telegram application, streaming response logic, and message routing
├── db_manager.py         # Dynamic VectorDBManager managing isolated database indexes and schemas
├── bot_history.db        # Unified SQLite database for both modes (contains a "mode" column)
├── bot_history_local.index # FAISS index file (384 dimensions) for local mode
└── bot_history_api.index # FAISS index file (3072 dimensions) for API mode
```

