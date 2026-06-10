# Friendly Gemini-Powered Telegram Bot (with FAISS Vector DB)

Welcome! This is a modern, high-performance, asynchronous Telegram bot powered by **Google's Gemini models** and backed by a local **FAISS Vector Database** + **SQLite metadata store** for semantic memory and history tracking.

---

## 🌟 Key Features

*   **⚡ Real-Time Streaming Responses**: Utilizes `generate_content_stream` to display Gemini's response chunk-by-chunk in real-time, reducing perceived latency to near zero. Includes a dynamic typing cursor (`▌`).
*   **👥 Personalized Greetings**: Supports `/adddetail` to collect the user's full name. The bot stores this and feeds it into the system instructions, allowing Gemini to chat with the user by name.
*   **🧠 FAISS Vector Memory**: Generates semantic embeddings of user prompts using `gemini-embedding-001` (3072 dimensions) and indexes them on-disk with FAISS.
*   **📁 SQLite Metadata Store**: Persists chat timestamps, usernames, message logs, and responses, cleanly linked 1:1 to the FAISS index keys.
*   **💬 Warm Chatbot Personality**: Configured via system instructions to behave like a close, supportive friend who always includes expressive emojis in its responses.
*   **⚙️ Throttle Optimization**: Throttles message edits to once every `0.8s` to speed up display rendering while staying safely within Telegram's rate-limiting rules.
*   **🛡️ Robust Error Fallbacks**: Features automatic markdown-to-plain-text formatting fallback if Telegram fails to parse unescaped AI markdown characters.

---

## 🛠️ Architecture Overview

When a user chats with the bot:
1. **Gemini Streaming**: The bot queries Gemini, streaming the response text back to the Telegram chat.
2. **Asynchronous Vectorization**: Once the conversation finishes, the bot launches an asynchronous background task.
3. **Embedding Generation**: The prompt is embedded using `gemini-embedding-001`.
4. **Local DB Save**: The embedding vector is appended to the local FAISS index (`bot_history.index`), and message metadata is saved to SQLite (`bot_history.db`) on a background worker thread (`asyncio.to_thread`) to ensure zero lag.

---

## ⚙️ Setup & Installation

### Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### 1. Clone & Initialize Environment
In the project directory, initialize a Python virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (Mac/Linux)
source .venv/bin/activate
```

### 2. Install Dependencies
Install the required dependencies inside your activated virtual environment:
```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a `.env` file in the root directory (based on the template provided) and add your keys:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

*   **Telegram Bot Token**: Get this by messaging **@BotFather** on Telegram.
*   **Gemini API Key**: Create a free key at [Google AI Studio](https://aistudio.google.com/).

---

## 🚀 Running the Bot

Start the bot locally:
```bash
python bot.py
```

---

## ⌨️ Bot Commands

*   `/start` - Initializes the chat, greets you (personally, if your name is saved), and lists available commands.
*   `/help` - Shows general instruction guidelines.
*   `/adddetail` - Prompts you to enter your full name so the bot can personalize its replies.
*   `/cancel` - Cancels the active `/adddetail` input prompt.
*   `/history` - Retrieves your last 10 logged conversation entries from the FAISS database.

---

## 📁 File Structure

```text
├── .env                  # Private credentials and model settings (ignored by git)
├── .gitignore            # Excludes venv, cached files, and database files
├── requirements.txt      # Project packages (python-telegram-bot, google-genai, faiss-cpu, numpy)
├── bot.py                # Main Telegram application, command/message handlers, background tasks
├── db_manager.py         # VectorDBManager for FAISS and SQLite operations
├── bot_history.db        # Persisted SQLite database (created on startup)
└── bot_history.index     # Persisted FAISS vector index (created on first message)
```
