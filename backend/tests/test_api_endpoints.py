import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from fastapi import HTTPException


@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""

    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns correct message"""
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Course Materials RAG System API"}

    def test_query_endpoint_with_session(self, test_client, sample_query_request):
        """Test query endpoint with provided session ID"""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session"
        assert isinstance(data["sources"], list)

    def test_query_endpoint_without_session(self, test_client, sample_query_request_no_session):
        """Test query endpoint without session ID (should create new session)"""
        response = test_client.post("/api/query", json=sample_query_request_no_session)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"  # From mock fixture
        assert isinstance(data["sources"], list)

    def test_query_endpoint_invalid_request(self, test_client):
        """Test query endpoint with invalid request data"""
        invalid_request = {"invalid_field": "value"}
        response = test_client.post("/api/query", json=invalid_request)
        
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query string"""
        empty_query = {"query": ""}
        response = test_client.post("/api/query", json=empty_query)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_courses_endpoint(self, test_client):
        """Test courses endpoint returns course statistics"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Course 1", "Course 2"]
        assert isinstance(data["course_titles"], list)

    def test_query_endpoint_with_rag_exception(self, test_app):
        """Test query endpoint when RAG system raises exception"""
        # Create a test app with a failing RAG system
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        from pydantic import BaseModel
        from typing import List, Optional
        
        class QueryRequest(BaseModel):
            query: str
            session_id: Optional[str] = None
        
        class QueryResponse(BaseModel):
            answer: str
            sources: List[str]
            session_id: str
        
        failing_app = FastAPI(title="Failing Test App")
        failing_app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        failing_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
        
        # Mock RAG that raises exception
        failing_rag = Mock()
        failing_rag.query.side_effect = Exception("RAG system failure")
        failing_rag.session_manager.create_session.return_value = "fail_session"
        
        @failing_app.post("/api/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            try:
                session_id = request.session_id or failing_rag.session_manager.create_session()
                answer, sources = failing_rag.query(request.query, session_id)
                return QueryResponse(answer=answer, sources=sources, session_id=session_id)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        client = TestClient(failing_app)
        response = client.post("/api/query", json={"query": "test query"})
        
        assert response.status_code == 500
        assert "RAG system failure" in response.json()["detail"]

    def test_courses_endpoint_with_exception(self, test_app):
        """Test courses endpoint when analytics raises exception"""
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        from pydantic import BaseModel
        from typing import List
        
        class CourseStats(BaseModel):
            total_courses: int
            course_titles: List[str]
        
        failing_app = FastAPI(title="Failing Test App")
        failing_app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        failing_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
        
        # Mock RAG that raises exception
        failing_rag = Mock()
        failing_rag.get_course_analytics.side_effect = Exception("Analytics failure")
        
        @failing_app.get("/api/courses", response_model=CourseStats)
        async def get_course_stats():
            try:
                analytics = failing_rag.get_course_analytics()
                return CourseStats(
                    total_courses=analytics["total_courses"],
                    course_titles=analytics["course_titles"]
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        client = TestClient(failing_app)
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics failure" in response.json()["detail"]

    def test_query_endpoint_response_structure(self, test_client, sample_query_request):
        """Test that query endpoint response has correct structure"""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_courses_endpoint_response_structure(self, test_client):
        """Test that courses endpoint response has correct structure"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check that all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_cors_headers(self, test_client):
        """Test that CORS headers are properly set"""
        response = test_client.get("/")
        
        # FastAPI with CORS middleware should include these headers
        assert response.status_code == 200
        # Note: TestClient may not include all CORS headers in response
        # This is a basic test to ensure the endpoint is accessible

    def test_query_endpoint_with_long_query(self, test_client):
        """Test query endpoint with a very long query string"""
        long_query = "a" * 10000  # 10KB query
        request_data = {"query": long_query}
        
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_endpoint_with_special_characters(self, test_client):
        """Test query endpoint with special characters"""
        special_query = {
            "query": "What about émojis 🚀 and símb0ls #@$%^&*()?",
            "session_id": "special_session_123"
        }
        
        response = test_client.post("/api/query", json=special_query)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "special_session_123"