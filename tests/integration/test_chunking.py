"""Test suite for AST-based chunking functionality."""

import tempfile
import os

import pytest

from chunking.multi_language_chunker import MultiLanguageChunker


@pytest.mark.integration
class TestChunking:
    """Test suite for AST-based chunking functionality."""

    def test_chunking(self):
        """Test AST-based chunking of Python code."""
        # Create a more complex test Python file
        test_code = '''
import os
import json
from typing import Dict, List, Optional

# Configuration constants
API_VERSION = "v1"
DEFAULT_TIMEOUT = 30

class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, connection_string: str):
        """Initialize database manager."""
        self.connection_string = connection_string
        self.connection = None

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            # Connection logic here
            self.connection = create_connection(self.connection_string)
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute SQL query with parameters."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or {})
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user with username and password."""
    if not username or not password:
        raise ValueError("Username and password required")

    # Hash password for comparison
    password_hash = hash_password(password)

    # Database lookup
    db = DatabaseManager(DATABASE_URL)
    if not db.connect():
        raise ConnectionError("Database unavailable")

    query = "SELECT * FROM users WHERE username = ? AND password_hash = ?"
    results = db.execute_query(query, {"username": username, "password_hash": password_hash})

    if results:
        return results[0]
    return None

@login_required
def get_user_profile(user_id: int) -> Dict:
    """Get user profile data."""
    db = DatabaseManager(DATABASE_URL)
    db.connect()

    query = "SELECT * FROM user_profiles WHERE user_id = ?"
    profiles = db.execute_query(query, {"user_id": user_id})

    if not profiles:
        raise ValueError(f"Profile not found for user {user_id}")

    return profiles[0]
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()

            try:
                # Test chunking
                chunker = MultiLanguageChunker(os.path.dirname(f.name))
                chunks = chunker.chunk_file(f.name)

                # Assertions
                assert len(chunks) > 0, "Should generate at least one chunk"
                # Note: Minimum chunks may vary based on chunking strategy

                # Check chunk structure
                for chunk in chunks:
                    # Validate chunk attributes
                    assert hasattr(chunk, 'chunk_type'), "Chunk should have chunk_type"
                    assert hasattr(chunk, 'start_line'), "Chunk should have start_line"
                    assert hasattr(chunk, 'end_line'), "Chunk should have end_line"
                    assert hasattr(chunk, 'content'), "Chunk should have content"
                    assert hasattr(chunk, 'tags'), "Chunk should have tags"

                    assert chunk.start_line > 0, "Start line should be positive"
                    assert chunk.end_line >= chunk.start_line, "End line should be >= start line"
                    assert isinstance(chunk.tags, (list, set)), "Tags should be a list or set"
                    assert isinstance(chunk.content, str), "Content should be a string"
                    assert len(chunk.content) > 0, "Content should not be empty"

            finally:
                os.unlink(f.name)

    def test_chunking_with_decorators(self):
        """Test that decorators are captured in chunks."""
        test_code = '''
@login_required
def protected_function():
    """A protected function."""
    pass

@property
@cache
def expensive_property(self):
    """An expensive property."""
    return compute_something()
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()

            try:
                chunker = MultiLanguageChunker(os.path.dirname(f.name))
                chunks = chunker.chunk_file(f.name)

                assert len(chunks) > 0, "Should generate chunks with decorators"

                # Check that decorators are captured
                has_decorators = any(chunk.decorators for chunk in chunks)
                assert has_decorators, "At least one chunk should have decorators"

            finally:
                os.unlink(f.name)
