import logging
import os
import asyncio
import threading
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, 
    ContextTypes, 
    CommandHandler, 
    MessageHandler, 
    filters,
    ConversationHandler
)
from google import genai
from google.genai import types
from db_manager import VectorDBManager

# Load environment variables from .env file
load_dotenv()

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global placeholders
tokenizer = None
model = None
embedding_model = None
client = None  # Gemini API Client
db = None

# Mode Configuration (api / local)
MODE = os.getenv("MODE", "local").lower()

# Local Model Configuration
MODEL_PATH = r"D:\models\Qwen3-4B"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Conversation Handler States
AWAITING_FULL_NAME = 1

# Command handler for /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_name = update.effective_user.first_name if update.effective_user else "there"
    
    # Check if we have the user's saved full name
    saved_name = await asyncio.to_thread(db.get_user_personal_info, update.effective_user.id)
    greeting_name = saved_name if saved_name else user_name
    
    welcome_message = (
        f"Hi {greeting_name}! 👋\n\n"
        "Hi, I am your assistant bot\n"
        "Send me any message, 😏\n\n"
        "Available commands:\n"
        "/start      - Welcome message\n"
        "/help       - Show instructions\n"
        "/adddetail  - Save or update your full name\n"
        "/history    - Show your last 10 logged conversation entries"
    )
    await update.message.reply_text(welcome_message)

# Command handler for /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Here are the available commands:\n"
        "/start      - Start interacting with the bot\n"
        "/help       - Get this help message\n"
        "/adddetail  - Tell the bot your full name to personalize interactions\n"
        "/history    - Retrieve your recent chat history from the FAISS database\n\n"
        "Every question and answer is embedded and logged into a local vector database for long-term memory!"
    )
    await update.message.reply_text(help_text)

# --- Add Detail Conversation Flow ---

async def add_detail_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point for /adddetail command."""
    await update.message.reply_text(
        "Please enter your full name (or type /cancel to abort):"
    )
    return AWAITING_FULL_NAME

async def save_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Saves the user's full name to SQLite database."""
    full_name = update.message.text.strip()
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name or "Unknown"
    
    if not full_name:
        await update.message.reply_text("Invalid name. Please type a valid name (or /cancel to abort):")
        return AWAITING_FULL_NAME
        
    try:
        await asyncio.to_thread(db.save_user_personal_info, user_id, username, full_name)
        await update.message.reply_text(
            f"Successfully saved your name as *{full_name}*! 🎉 Now we know who you are.",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error saving name to DB: {e}")
        await update.message.reply_text("Sorry, I had trouble saving your detail. Please try again later.")
        
    return ConversationHandler.END

async def cancel_add_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the /adddetail conversation."""
    await update.message.reply_text("Action cancelled.")
    return ConversationHandler.END

# --- Message history retrieving ---

# Command handler for retrieving history
async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        # Fetch recent history asynchronously using thread executor
        history = await asyncio.to_thread(db.get_recent_history, user_id, 10)
        
        if not history:
            await update.message.reply_text("You don't have any logged conversation history yet!")
            return
            
        reply_lines = ["*Your Recent Chat History (Last 10 messages) - Unified (FREE & PREMIUM):*"]
        # history is ordered by timestamp DESC, so we reverse it to show chronological order
        for idx, (msg, reply, timestamp, row_mode) in enumerate(reversed(history), 1):
            mode_label = "PREMIUM" if row_mode == "api" else "FREE"
            reply_lines.append(f"\n{idx}. *[{timestamp}]* (Mode: {mode_label})")
            reply_lines.append(f"❓ *You:* {msg}")
            reply_lines.append(f"🤖 *Bot:* {reply}")
            
        history_text = "\n".join(reply_lines)
        
        # Split history text if it exceeds Telegram's 4096 character limit
        if len(history_text) > 4000:
            history_text = history_text[:3990] + "\n...[truncated]"
            
        await update.message.reply_text(history_text, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        await update.message.reply_text("Sorry, I encountered an error while trying to fetch your history.")

# Background task to embed and log the conversation
async def log_conversation_bg(user_id: int, username: str, user_message: str, bot_reply: str):
    try:
        logger.info(f"Generating embedding for user message in {MODE} mode...")
        
        if MODE == "api":
            # API Mode: Call Gemini Embedding API
            embed_response = await client.aio.models.embed_content(
                model="gemini-embedding-001",
                contents=user_message,
            )
            embedding_vector = embed_response.embeddings[0].values
        else:
            # Local Mode: Call local sentence-transformers
            embedding_np = await asyncio.to_thread(embedding_model.encode, user_message)
            embedding_vector = embedding_np.tolist()
        
        # Save embedding and metadata using a background thread
        await asyncio.to_thread(
            db.insert_entry, 
            embedding_vector, 
            user_id, 
            username, 
            user_message, 
            bot_reply
        )
        logger.info(f"Successfully saved conversation to FAISS and SQLite ({MODE} mode).")
        
    except Exception as db_err:
        logger.error(f"Failed to log conversation to database: {db_err}")

# Async generator to yield streamed chunks from local model
async def generate_local_stream(user_text: str, instruction: str):
    from transformers import TextIteratorStreamer
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_text}
    ]
    
    # Format the prompt using Qwen's template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512)
    
    # Run the generate method in a separate thread so it doesn't block the loop
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens as they arrive
    while True:
        chunk = await asyncio.to_thread(lambda: next(iter(streamer), None))
        if chunk is None:
            break
        yield chunk

def remove_thinking_tags(text: str) -> str:
    """Removes `<think>...</think>` tags and any content inside them.
    Also handles open-ended `<think>` blocks during streaming."""
    import re
    # Remove complete <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # If there is an unclosed <think> tag left, strip everything from it to the end
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned

# Message handler for chatting with Gemini or local model
async def chat_with_gemini(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name or "Unknown"
    logger.info(f"Received message: '{user_text}' from user: {user_id} in {MODE} mode")
    
    # Send a placeholder reply message that we will edit dynamically
    placeholder_msg = await update.message.reply_text("Thinking...")
    
    try:
        accumulated_text = ""
        last_update_time = asyncio.get_event_loop().time()
        
        # Check if user has personal info registered
        saved_name = await asyncio.to_thread(db.get_user_personal_info, user_id)
        name_part = f" The user's name is {saved_name}." if saved_name else ""
        
        # Configure friendly system instructions with strict emoji requirements
        instruction = (
            f"You are a close, warm, and supportive friend to the user.{name_part} "
            "Adopt a friendly, casual, and enthusiastic tone. Keep your responses concise, clear, and direct. "
            "CRITICAL: You MUST include multiple emojis in every single response you send. "
            "Every response should be expressive, engaging, and loaded with friendly emojis."
        )
        
        if MODE == "api":
            # API Mode Streaming
            config = types.GenerateContentConfig(
                system_instruction=instruction
            )
            response_stream = await client.aio.models.generate_content_stream(
                model=gemini_model,
                contents=user_text,
                config=config,
            )
        else:
            # Local Mode Streaming
            response_stream = generate_local_stream(user_text, instruction)
        
        async for chunk in response_stream:
            if chunk:
                # Safely extract text chunk dynamically (local uses str, API uses object chunk.text)
                text_chunk = chunk if isinstance(chunk, str) else chunk.text
                if text_chunk:
                    accumulated_text += text_chunk
                    
                    # Update the message in Telegram, throttling to once every 0.8s for faster rendering
                    cleaned_display = remove_thinking_tags(accumulated_text)
                    if cleaned_display.strip():
                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_update_time > 0.8:
                            try:
                                await context.bot.edit_message_text(
                                    chat_id=update.effective_chat.id,
                                    message_id=placeholder_msg.message_id,
                                    text=cleaned_display + " ▌"  # cursor character
                                )
                                last_update_time = current_time
                            except Exception as edit_err:
                                logger.warning(f"Error during stream update: {edit_err}")
                                pass
        
        # Process the final clean content
        accumulated_text = remove_thinking_tags(accumulated_text)
        if not accumulated_text.strip():
            accumulated_text = "I couldn't generate a response."

            
        # Final update to send the complete formatted response
        try:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=placeholder_msg.message_id,
                text=accumulated_text,
                parse_mode="Markdown"
            )
        except Exception as parse_error:
            logger.warning(f"Markdown formatting failed for final stream update: {parse_error}")
            try:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=placeholder_msg.message_id,
                    text=accumulated_text
                )
            except Exception as final_err:
                logger.error(f"Failed to edit final fallback message: {final_err}")
        
        # Run DB logging asynchronously in the background
        asyncio.create_task(log_conversation_bg(user_id, username, user_text, accumulated_text))
                
    except Exception as e:
        logger.error(f"Error calling model stream ({MODE} mode): {e}")
        try:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=placeholder_msg.message_id,
                text="Sorry, I encountered an error trying to process that request."
            )
        except Exception:
            pass

def main():
    # Retrieve the bot token from environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token or bot_token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        logger.error("TELEGRAM_BOT_TOKEN is missing or not set in the .env file!")
        print("\n[ERROR] Bot token missing! Please check your .env file and set TELEGRAM_BOT_TOKEN.\n")
        return

    # Initialize the Database based on current MODE
    global db
    db = VectorDBManager(mode=MODE)

    # Initialize the model / client based on current MODE
    global client, tokenizer, model, embedding_model, gemini_model
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    if MODE == "api":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
            logger.error("GEMINI_API_KEY is missing or not set in the .env file!")
            print("\n[ERROR] Gemini API Key missing! Please check your .env file and set GEMINI_API_KEY.\n")
            return
            
        logger.info("Initializing Gemini Cloud Client (API Mode)...")
        client = genai.Client(api_key=gemini_key)
        logger.info("Gemini Cloud Client successfully initialized.")
    else:
        logger.info("Initializing Local GPU Model loader (Local Mode)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from sentence_transformers import SentenceTransformer
        import torch
        
        logger.info(f"Loading local tokenizer from: {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        logger.info(f"Loading local causal LM model from: {MODEL_PATH} in 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        logger.info(f"Loading sentence-transformer embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("All local models successfully initialized and ready!")

    # Build the application with the bot's token
    application = ApplicationBuilder().token(bot_token).build()

    # Set up ConversationHandler for /adddetail command
    add_detail_conv = ConversationHandler(
        entry_points=[CommandHandler("adddetail", add_detail_start)],
        states={
            AWAITING_FULL_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_name)],
        },
        fallbacks=[CommandHandler("cancel", cancel_add_detail)],
    )

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(add_detail_conv)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_gemini))

    # Run the bot (polls updates from Telegram)
    display_mode = "PREMIUM" if MODE == "api" else "FREE"
    logger.info(f"Bot starting up in {display_mode} mode... Press Ctrl+C to stop.")
    application.run_polling()

if __name__ == '__main__':
    main()
