"""
SQLite storage backend implementation.

This module provides a SQLite-based storage backend for the memory system,
suitable for development and small-scale deployments.
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from contextlib import contextmanager
from ..storage.base_storage import BaseStorage
from ..core.models import (
    Message, Conversation, MemoryFragment,
    ConversationSummary, UserProfile,
    MessageRole, MemoryType, Framework
)
from ..core.exceptions import StorageError


class SQLiteStorage(BaseStorage):
    """
    SQLite storage backend implementation.
    
    This class provides persistent storage using SQLite database,
    which is file-based and requires no server setup.
    """
    
    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self.initialize()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
        try:
            yield self.connection
        except sqlite3.Error as e:
            self.connection.rollback()
            raise StorageError(f"Database error: {str(e)}")
        else:
            self.connection.commit()
    
    def initialize(self) -> None:
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    framework TEXT,
                    title TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    summary TEXT,
                    metadata TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    token_count INTEGER,
                    metadata TEXT,
                    embedding BLOB,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            # Create memory_fragments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_fragments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT,
                    fragment_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    expiry_date TIMESTAMP,
                    embedding BLOB,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            # Create conversation_summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_topics TEXT,
                    created_at TIMESTAMP NOT NULL,
                    message_count_at_summary INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            # Create user_profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT,
                    known_entities TEXT,
                    interaction_style TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user 
                ON memory_fragments(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance 
                ON memory_fragments(importance_score DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_accessed 
                ON memory_fragments(last_accessed DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user 
                ON conversations(user_id)
            """)
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    # Message operations
    
    def store_message(self, message: Message) -> int:
        """Store a message."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            embedding_blob = json.dumps(message.embedding) if message.embedding else None
            metadata_json = json.dumps(message.metadata) if message.metadata else None
            
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, timestamp,
                                    token_count, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                message.conversation_id,
                message.role.value if isinstance(message.role, MessageRole) else message.role,
                message.content,
                message.timestamp.isoformat(),
                message.token_count,
                metadata_json,
                embedding_blob
            ))
            
            return cursor.lastrowid
    
    def get_message(self, message_id: int) -> Optional[Message]:
        """Retrieve a message by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM messages WHERE id = ?
            """, (message_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_message(row)
            return None
    
    def get_messages(self, conversation_id: Optional[str] = None,
                    limit: Optional[int] = None,
                    offset: int = 0) -> List[Message]:
        """Retrieve messages."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM messages"
            params = []
            
            if conversation_id:
                query += " WHERE conversation_id = ?"
                params.append(conversation_id)
            
            query += " ORDER BY timestamp ASC"
            
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(query, params)
            
            return [self._row_to_message(row) for row in cursor.fetchall()]
    
    def search_messages(self, query: str, limit: int = 10) -> List[Tuple[Message, float]]:
        """Search messages."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Simple text search
            cursor.execute("""
                SELECT *, 
                       LENGTH(content) - LENGTH(REPLACE(LOWER(content), LOWER(?), '')) 
                       AS match_count
                FROM messages
                WHERE content LIKE ?
                ORDER BY match_count DESC
                LIMIT ?
            """, (query, f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                message = self._row_to_message(row)
                # Calculate simple relevance score
                score = min(row['match_count'] / len(query), 1.0) if row['match_count'] else 0
                results.append((message, score))
            
            return results
    
    # Conversation operations
    
    def store_conversation(self, conversation: Conversation) -> str:
        """Store a conversation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(conversation.metadata) if conversation.metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO conversations 
                (id, user_id, framework, title, created_at, updated_at, summary, metadata, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation.id,
                conversation.user_id,
                conversation.framework.value if conversation.framework else None,
                conversation.title,
                conversation.created_at.isoformat(),
                conversation.updated_at.isoformat(),
                conversation.summary,
                metadata_json,
                conversation.is_active
            ))
            
            # Store messages
            for message in conversation.messages:
                message.conversation_id = conversation.id
                self.store_message(message)
            
            return conversation.id
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM conversations WHERE id = ?
            """, (conversation_id,))
            
            row = cursor.fetchone()
            if row:
                conversation = self._row_to_conversation(row)
                # Load messages
                conversation.messages = self.get_messages(conversation_id)
                return conversation
            return None
    
    def get_user_conversations(self, user_id: str,
                              limit: Optional[int] = None,
                              offset: int = 0) -> List[Conversation]:
        """Get conversations for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM conversations 
                WHERE user_id = ? 
                ORDER BY updated_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(query, (user_id,))
            
            conversations = []
            for row in cursor.fetchall():
                conv = self._row_to_conversation(row)
                # Load messages for each conversation
                conv.messages = self.get_messages(conv.id)
                conversations.append(conv)
            
            return conversations
    
    def update_conversation(self, conversation: Conversation) -> None:
        """Update a conversation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(conversation.metadata) if conversation.metadata else None
            
            cursor.execute("""
                UPDATE conversations
                SET title = ?, updated_at = ?, summary = ?, metadata = ?, is_active = ?
                WHERE id = ?
            """, (
                conversation.title,
                conversation.updated_at.isoformat(),
                conversation.summary,
                metadata_json,
                conversation.is_active,
                conversation.id
            ))
    
    # Memory fragment operations
    
    def store_memory(self, memory: MemoryFragment) -> int:
        """Store a memory fragment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            embedding_blob = json.dumps(memory.embedding) if memory.embedding else None
            metadata_json = json.dumps(memory.metadata) if memory.metadata else None
            
            cursor.execute("""
                INSERT INTO memory_fragments
                (user_id, conversation_id, fragment_type, content, importance_score,
                 created_at, last_accessed, access_count, expiry_date, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.user_id,
                memory.conversation_id,
                memory.fragment_type.value if isinstance(memory.fragment_type, MemoryType) else memory.fragment_type,
                memory.content,
                memory.importance_score,
                memory.created_at.isoformat(),
                memory.last_accessed.isoformat(),
                memory.access_count,
                memory.expiry_date.isoformat() if memory.expiry_date else None,
                embedding_blob,
                metadata_json
            ))
            
            return cursor.lastrowid
    
    def get_memory(self, memory_id: int) -> Optional[MemoryFragment]:
        """Retrieve a memory fragment by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM memory_fragments WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_memory(row)
            return None
    
    def search_memories(self, query: str, limit: int = 10) -> List[Tuple[MemoryFragment, float]]:
        """Search memory fragments."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT *, 
                       LENGTH(content) - LENGTH(REPLACE(LOWER(content), LOWER(?), '')) 
                       AS match_count
                FROM memory_fragments
                WHERE content LIKE ?
                ORDER BY match_count DESC, importance_score DESC
                LIMIT ?
            """, (query, f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                memory = self._row_to_memory(row)
                score = min(row['match_count'] / len(query), 1.0) if row['match_count'] else 0
                results.append((memory, score))
            
            return results
    
    def search_user_memories(self, user_id: str, query: str,
                           limit: int = 10,
                           search_type: str = 'semantic') -> List[MemoryFragment]:
        """Search memories for a specific user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # For now, implement keyword search
            # Semantic search would require embedding comparison
            cursor.execute("""
                SELECT * FROM memory_fragments
                WHERE user_id = ? AND content LIKE ?
                ORDER BY importance_score DESC, last_accessed DESC
                LIMIT ?
            """, (user_id, f'%{query}%', limit))
            
            return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def get_recent_memories(self, user_id: str, limit: int = 10) -> List[MemoryFragment]:
        """Get recent memories for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM memory_fragments
                WHERE user_id = ?
                ORDER BY last_accessed DESC
                LIMIT ?
            """, (user_id, limit))
            
            return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def get_important_memories(self, user_id: str, limit: int = 10) -> List[MemoryFragment]:
        """Get most important memories for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM memory_fragments
                WHERE user_id = ?
                ORDER BY importance_score DESC, access_count DESC
                LIMIT ?
            """, (user_id, limit))
            
            return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def get_all_user_memories(self, user_id: str) -> List[MemoryFragment]:
        """Get all memories for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM memory_fragments
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
            
            return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def get_user_memory_count(self, user_id: str) -> int:
        """Get count of memories for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM memory_fragments
                WHERE user_id = ?
            """, (user_id,))
            
            return cursor.fetchone()['count']
    
    def update_memory_access(self, memory_id: int) -> None:
        """Update memory access timestamp and count."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE memory_fragments
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """, (datetime.now().isoformat(), memory_id))
    
    # Memory deletion operations
    
    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory fragment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM memory_fragments WHERE id = ?
            """, (memory_id,))
            
            return cursor.rowcount > 0
    
    def delete_expired_memories(self, user_id: str) -> int:
        """Delete expired memories for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM memory_fragments
                WHERE user_id = ? AND expiry_date < ?
            """, (user_id, datetime.now().isoformat()))
            
            return cursor.rowcount
    
    def delete_low_importance_memories(self, user_id: str, threshold: float) -> int:
        """Delete memories below importance threshold."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM memory_fragments
                WHERE user_id = ? AND importance_score < ?
            """, (user_id, threshold))
            
            return cursor.rowcount
    
    def delete_old_memories(self, user_id: str, cutoff_date: datetime) -> int:
        """Delete memories older than cutoff date."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM memory_fragments
                WHERE user_id = ? AND created_at < ?
            """, (user_id, cutoff_date.isoformat()))
            
            return cursor.rowcount
    
    def delete_all_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM memory_fragments WHERE user_id = ?
            """, (user_id,))
            
            return cursor.rowcount
    
    # Summary operations
    
    def store_summary(self, summary: ConversationSummary) -> int:
        """Store a conversation summary."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            key_topics_json = json.dumps(summary.key_topics)
            metadata_json = json.dumps(summary.metadata) if summary.metadata else None
            
            cursor.execute("""
                INSERT INTO conversation_summaries
                (conversation_id, summary, key_topics, created_at,
                 message_count_at_summary, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                summary.conversation_id,
                summary.summary,
                key_topics_json,
                summary.created_at.isoformat(),
                summary.message_count_at_summary,
                metadata_json
            ))
            
            return cursor.lastrowid
    
    def get_conversation_summaries(self, conversation_id: str) -> List[ConversationSummary]:
        """Get summaries for a conversation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM conversation_summaries
                WHERE conversation_id = ?
                ORDER BY created_at DESC
            """, (conversation_id,))
            
            return [self._row_to_summary(row) for row in cursor.fetchall()]
    
    # User profile operations
    
    def store_user_profile(self, profile: UserProfile) -> None:
        """Store or update a user profile."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            preferences_json = json.dumps(profile.preferences)
            entities_json = json.dumps(profile.known_entities)
            metadata_json = json.dumps(profile.metadata) if profile.metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles
                (user_id, preferences, known_entities, interaction_style,
                 created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                preferences_json,
                entities_json,
                profile.interaction_style,
                profile.created_at.isoformat(),
                profile.updated_at.isoformat(),
                metadata_json
            ))
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM user_profiles WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_profile(row)
            return None
    
    def update_user_profile(self, profile: UserProfile) -> None:
        """Update a user profile."""
        self.store_user_profile(profile)  # INSERT OR REPLACE handles updates
    
    # Utility operations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count conversations
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            stats['total_conversations'] = cursor.fetchone()['count']
            
            # Count messages
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            stats['total_messages'] = cursor.fetchone()['count']
            
            # Count memories
            cursor.execute("SELECT COUNT(*) as count FROM memory_fragments")
            stats['total_memories'] = cursor.fetchone()['count']
            
            # Count users
            cursor.execute("SELECT COUNT(*) as count FROM user_profiles")
            stats['total_users'] = cursor.fetchone()['count']
            
            # Database size
            if os.path.exists(self.db_path):
                stats['database_size_bytes'] = os.path.getsize(self.db_path)
            
            return stats
    
    def clear_all(self) -> None:
        """Clear all data from storage."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete all data (order matters due to foreign keys)
            cursor.execute("DELETE FROM conversation_summaries")
            cursor.execute("DELETE FROM memory_fragments")
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM conversations")
            cursor.execute("DELETE FROM user_profiles")
    
    def backup(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            with self.get_connection() as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            return True
        except Exception as e:
            raise StorageError(f"Backup failed: {str(e)}")
    
    def restore(self, backup_path: str) -> bool:
        """Restore from a backup."""
        try:
            self.close()
            backup_conn = sqlite3.connect(backup_path)
            new_conn = sqlite3.connect(self.db_path)
            backup_conn.backup(new_conn)
            backup_conn.close()
            new_conn.close()
            self.connection = None
            return True
        except Exception as e:
            raise StorageError(f"Restore failed: {str(e)}")
    
    # Helper methods to convert database rows to model objects
    
    def _row_to_message(self, row) -> Message:
        """Convert a database row to a Message object."""
        return Message(
            id=row['id'],
            conversation_id=row['conversation_id'],
            role=MessageRole(row['role']),
            content=row['content'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            token_count=row['token_count'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            embedding=json.loads(row['embedding']) if row['embedding'] else None
        )
    
    def _row_to_conversation(self, row) -> Conversation:
        """Convert a database row to a Conversation object."""
        return Conversation(
            id=row['id'],
            user_id=row['user_id'],
            framework=Framework(row['framework']) if row['framework'] else None,
            title=row['title'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            messages=[],  # Messages loaded separately
            summary=row['summary'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            is_active=bool(row['is_active'])
        )
    
    def _row_to_memory(self, row) -> MemoryFragment:
        """Convert a database row to a MemoryFragment object."""
        return MemoryFragment(
            id=row['id'],
            user_id=row['user_id'],
            conversation_id=row['conversation_id'],
            fragment_type=MemoryType(row['fragment_type']),
            content=row['content'],
            importance_score=row['importance_score'],
            created_at=datetime.fromisoformat(row['created_at']),
            last_accessed=datetime.fromisoformat(row['last_accessed']),
            access_count=row['access_count'],
            expiry_date=datetime.fromisoformat(row['expiry_date']) if row['expiry_date'] else None,
            embedding=json.loads(row['embedding']) if row['embedding'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def _row_to_summary(self, row) -> ConversationSummary:
        """Convert a database row to a ConversationSummary object."""
        return ConversationSummary(
            id=row['id'],
            conversation_id=row['conversation_id'],
            summary=row['summary'],
            key_topics=json.loads(row['key_topics']) if row['key_topics'] else [],
            created_at=datetime.fromisoformat(row['created_at']),
            message_count_at_summary=row['message_count_at_summary'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def _row_to_profile(self, row) -> UserProfile:
        """Convert a database row to a UserProfile object."""
        return UserProfile(
            user_id=row['user_id'],
            preferences=json.loads(row['preferences']) if row['preferences'] else {},
            known_entities=json.loads(row['known_entities']) if row['known_entities'] else {},
            interaction_style=row['interaction_style'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
