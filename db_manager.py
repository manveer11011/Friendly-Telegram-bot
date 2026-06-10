import faiss
import numpy as np
import sqlite3
import datetime
import os
import logging

logger = logging.getLogger(__name__)

DB_FILE = "bot_history.db"
INDEX_FILE = "bot_history.index"
EMBEDDING_DIM = 3072 # Gemini gemini-embedding-001 outputs 3072-dimensional vectors

class VectorDBManager:
    def __init__(self):
        self.db_path = DB_FILE
        self.index_path = INDEX_FILE
        
        # Initialize SQLite database
        self._init_sqlite()
        
        # Initialize FAISS index
        self._init_faiss()

    def _init_sqlite(self):
        """Initializes the SQLite database and metadata tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY, -- matches FAISS index position
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
                self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        else:
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
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
            bot_reply (str): The response returned by Gemini
        """
        if len(embedding_vector) != EMBEDDING_DIM:
            raise ValueError(f"Embedding vector must be of dimension {EMBEDDING_DIM}, got {len(embedding_vector)}")
            
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
            INSERT INTO conversation_history (id, user_id, username, user_message, bot_reply, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (faiss_id, user_id, username, user_message, bot_reply, timestamp))
        
        conn.commit()
        conn.close()
        logger.info(f"Logged conversation to DB with ID: {faiss_id}")

    def get_recent_history(self, user_id: int, limit: int = 10) -> list:
        """Retrieves the most recent conversation records for a given user from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_message, bot_reply, timestamp 
            FROM conversation_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def search_similar(self, query_embedding: list, k: int = 3) -> list:
        """
        Performs semantic vector search on FAISS and retrieves details from SQLite.
        
        Args:
            query_embedding (list): Embedding of the query sentence.
            k (int): Number of nearest neighbors to retrieve.
        """
        if self.index.ntotal == 0:
            return []
            
        vector_np = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(vector_np, k)
        
        # Parse output indices (filter out -1 which represents no match found)
        valid_indices = [int(idx) for idx in indices[0] if idx != -1]
        
        if not valid_indices:
            return []
            
        # Retrieve records from SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # SQL parameter placeholders
        placeholders = ','.join('?' for _ in valid_indices)
        query = f"""
            SELECT id, user_id, username, user_message, bot_reply, timestamp 
            FROM conversation_history 
            WHERE id IN ({placeholders})
        """
        
        cursor.execute(query, valid_indices)
        rows = cursor.fetchall()
        conn.close()
        
        # Reorder rows to match the distance order
        row_dict = {row[0]: row for row in rows}
        sorted_rows = [row_dict[idx] for idx in valid_indices if idx in row_dict]
        
        return sorted_rows

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
