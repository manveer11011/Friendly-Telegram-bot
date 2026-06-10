import faiss
import numpy as np
import sqlite3
import datetime
import os
import logging

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, mode="local"):
        self.mode = mode
        self.db_path = "bot_history.db"  # Unified SQLite database for both modes
        if mode == "api":
            self.index_path = "bot_history_api.index"
            self.dim = 3072  # Gemini gemini-embedding-001 outputs 3072 dimensions
        else:
            self.index_path = "bot_history_local.index"
            self.dim = 384   # Local sentence-transformers outputs 384 dimensions
        
        # Initialize SQLite database
        self._init_sqlite()
        
        # Initialize FAISS index
        self._init_faiss()
 
    def _init_sqlite(self):
        """Initializes the SQLite database and metadata tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if conversation_history exists and needs recreation for the new schema
        cursor.execute("PRAGMA table_info(conversation_history)")
        columns = [row[1] for row in cursor.fetchall()]
        if columns and "mode" not in columns:
            logger.info("Outdated schema detected in SQLite. Recreating conversation_history table...")
            cursor.execute("DROP TABLE IF EXISTS conversation_history")
            
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                faiss_id INTEGER,
                mode TEXT,
                user_id INTEGER,
                username TEXT,
                user_message TEXT,
                bot_reply TEXT,
                timestamp TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_personal_information (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                full_name TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("SQLite database initialized successfully.")
 
    def _init_faiss(self):
        """Loads FAISS index from disk or creates a new one."""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index from {self.index_path} with {self.index.ntotal} items.")
            except Exception as e:
                logger.error(f"Failed to read FAISS index from {self.index_path}: {e}. Creating new index.")
                self.index = faiss.IndexFlatL2(self.dim)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            logger.info("Created new FAISS index.")
 
    def save_index(self):
        """Saves the FAISS index to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"FAISS index saved to {self.index_path}. Total vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error saving FAISS index to disk: {e}")
 
    def insert_entry(self, embedding_vector: list, user_id: int, username: str, user_message: str, bot_reply: str):
        """
        Inserts an embedding into FAISS and logs corresponding metadata in SQLite.
        
        Args:
            embedding_vector (list): List of floats of length EMBEDDING_DIM
            user_id (int): Telegram user ID
            username (str): Telegram username
            user_message (str): The prompt sent by user
            bot_reply (str): The response returned by Gemini/Qwen
        """
        if len(embedding_vector) != self.dim:
            raise ValueError(f"Embedding vector must be of dimension {self.dim}, got {len(embedding_vector)}")
            
        # Format the vector as float32 numpy array
        vector_np = np.array([embedding_vector], dtype='float32')
        
        # Add to FAISS index. The index ID will be the position inside the index (0-based)
        faiss_id = self.index.ntotal
        self.index.add(vector_np)
        self.save_index()
        
        # Insert metadata into SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            INSERT INTO conversation_history (faiss_id, mode, user_id, username, user_message, bot_reply, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (faiss_id, self.mode, user_id, username, user_message, bot_reply, timestamp))
        
        conn.commit()
        conn.close()
        logger.info(f"Logged conversation to DB with FAISS ID {faiss_id} in {self.mode} mode.")
 
    def get_recent_history(self, user_id: int, limit: int = 10) -> list:
        """Retrieves the most recent conversation records for a given user from SQLite (both modes)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_message, bot_reply, timestamp, mode 
            FROM conversation_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return rows

 
    def save_user_personal_info(self, user_id: int, username: str, full_name: str):
        """Inserts or updates a user's full name in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            INSERT OR REPLACE INTO user_personal_information (user_id, username, full_name, timestamp)
            VALUES (?, ?, ?, ?)
        """, (user_id, username, full_name, timestamp))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved personal info for user {user_id}: {full_name}")
 
    def get_user_personal_info(self, user_id: int) -> str:
        """Retrieves a user's saved full name, or None if not set."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT full_name 
            FROM user_personal_information 
            WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
