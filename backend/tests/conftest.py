import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
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
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Test Lesson 2",
                lesson_link="https://example.com/lesson2",
            ),
        ],
    )


@pytest.fixture
def mock_course_chunks():
    return [
        CourseChunk(
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0,
            content="This is the first chunk of content about testing.",
        ),
        CourseChunk(
            course_title="Test Course",
            lesson_number=1,
            chunk_index=1,
            content="This is the second chunk with more testing information.",
        ),
        CourseChunk(
            course_title="Test Course",
            lesson_number=2,
            chunk_index=2,
            content="This is content from lesson 2 about advanced topics.",
        ),
    ]


@pytest.fixture
def mock_search_results():
    return SearchResults(
        documents=[
            "This is the first chunk of content about testing.",
            "This is the second chunk with more testing information.",
        ],
        metadata=[
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 1},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    return SearchResults(
        documents=[], metadata=[], distances=[], error="Database connection failed"
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


@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Create test app
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Mock RAG system for testing
    mock_rag = Mock()
    mock_rag.query.return_value = ("Test answer", ["test_source.md"])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course 1", "Course 2"]
    }
    mock_rag.session_manager.create_session.return_value = "test_session_123"
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for testing"""
    mock = Mock()
    mock.query.return_value = ("Mock answer", ["mock_source.md"])
    mock.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Python Basics", "Advanced Python", "Web Development"]
    }
    mock.session_manager.create_session.return_value = "mock_session_456"
    return mock


@pytest.fixture
def sample_query_request():
    """Sample query request data for testing"""
    return {
        "query": "What is Python?",
        "session_id": "test_session"
    }


@pytest.fixture
def sample_query_request_no_session():
    """Sample query request without session ID"""
    return {
        "query": "How do I use FastAPI?"
    }
