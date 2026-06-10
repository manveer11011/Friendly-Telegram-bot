import logging
import os
import asyncio
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
client = None
gemini_model = "gemini-2.5-flash"
db = None

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
            
        reply_lines = ["*Your Recent Chat History (Last 10 messages):*"]
        # history is ordered by timestamp DESC, so we reverse it to show chronological order
        for idx, (msg, reply, timestamp) in enumerate(reversed(history), 1):
            reply_lines.append(f"\n{idx}. *[{timestamp}]*")
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
        logger.info(f"Generating embedding for user message: '{user_message[:30]}...'")
        
        # Get embedding vector using Gemini API
        embed_response = await client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=user_message,
        )
        
        embedding_vector = embed_response.embeddings[0].values
        
        # Save embedding and metadata using a background thread (to avoid blocking async loop)
        await asyncio.to_thread(
            db.insert_entry, 
            embedding_vector, 
            user_id, 
            username, 
            user_message, 
            bot_reply
        )
        logger.info("Successfully saved conversation to FAISS and SQLite.")
        
    except Exception as db_err:
        logger.error(f"Failed to log conversation to database: {db_err}")

# Message handler for chatting with Gemini
async def chat_with_gemini(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name or "Unknown"
    logger.info(f"Received message: '{user_text}' from user: {user_id}")
    
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
            
        # Configure Gemini system instructions to request concise responses for faster generation
        config = types.GenerateContentConfig(
            system_instruction=instruction
        )
        
        # Request a stream from the Gemini API
        response_stream = await client.aio.models.generate_content_stream(
            model=gemini_model,
            contents=user_text,
            config=config,
        )
        
        async for chunk in response_stream:
            if chunk.text:
                accumulated_text += chunk.text
                
                # Update the message in Telegram, throttling to once every 0.8s for faster rendering
                if accumulated_text.strip():
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_update_time > 0.8:
                        try:
                            await context.bot.edit_message_text(
                                chat_id=update.effective_chat.id,
                                message_id=placeholder_msg.message_id,
                                text=accumulated_text + " ▌"  # cursor character
                            )
                            last_update_time = current_time
                        except Exception as edit_err:
                            logger.warning(f"Error during stream update: {edit_err}")
                            pass
        
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
        
        # Run DB logging asynchronously in the background so we don't slow down the chat interface
        asyncio.create_task(log_conversation_bg(user_id, username, user_text, accumulated_text))
                
    except Exception as e:
        logger.error(f"Error calling Gemini API stream: {e}")
        try:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=placeholder_msg.message_id,
                text="Sorry, I encountered an error trying to process that request."
            )
        except Exception:
            pass

def main():
    # Retrieve the tokens from environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    has_errors = False
    if not bot_token or bot_token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        logger.error("TELEGRAM_BOT_TOKEN is missing or not set in the .env file!")
        has_errors = True
        
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("GEMINI_API_KEY is missing or not set in the .env file!")
        has_errors = True
        
    if has_errors:
        print("\n[ERROR] Configuration missing! Please check your .env file and set both TELEGRAM_BOT_TOKEN and GEMINI_API_KEY.\n")
        return

    # Initialize the Database
    global db
    db = VectorDBManager()

    # Initialize the Gemini Client
    global client, gemini_model
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=gemini_key)

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

    # Register handlers (ConversationHandler must be registered before the general chat handler)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(add_detail_conv)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_gemini))

    # Run the bot (polls updates from Telegram)
    logger.info("Bot starting up... Press Ctrl+C to stop.")
    application.run_polling()

if __name__ == '__main__':
    main()
