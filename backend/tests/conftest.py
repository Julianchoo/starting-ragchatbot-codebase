import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add the backend directory to the path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@dataclass
class TestConfig:
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5  # Fixed from 0
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def test_config():
    return TestConfig()


@pytest.fixture
def temp_chroma_path():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_course():
    return Course(
        title="Test Course",
        instructor="Test Instructor", 
        course_link="https://example.com/course",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Test Lesson 1",
                lesson_link="https://example.com/lesson1"
            ),
            Lesson(
                lesson_number=2,
                title="Test Lesson 2", 
                lesson_link="https://example.com/lesson2"
            )
        ]
    )


@pytest.fixture
def mock_course_chunks():
    return [
        CourseChunk(
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0,
            content="This is the first chunk of content about testing."
        ),
        CourseChunk(
            course_title="Test Course", 
            lesson_number=1,
            chunk_index=1,
            content="This is the second chunk with more testing information."
        ),
        CourseChunk(
            course_title="Test Course",
            lesson_number=2, 
            chunk_index=2,
            content="This is content from lesson 2 about advanced topics."
        )
    ]


@pytest.fixture
def mock_search_results():
    return SearchResults(
        documents=[
            "This is the first chunk of content about testing.",
            "This is the second chunk with more testing information."
        ],
        metadata=[
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 1}
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    return SearchResults(
        documents=[],
        metadata=[], 
        distances=[],
        error="Database connection failed"
    )


@pytest.fixture
def mock_vector_store():
    mock = Mock()
    mock.search = Mock()
    mock.get_all_courses_metadata = Mock()
    mock.get_lesson_link = Mock()
    return mock


@pytest.fixture
def mock_anthropic_client():
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Test response"
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_tool_manager():
    mock = Mock()
    mock.get_tool_definitions = Mock(return_value=[])
    mock.execute_tool = Mock(return_value="Tool executed successfully")
    mock.get_last_sources = Mock(return_value=[])
    mock.reset_sources = Mock()
    return mock