import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk
from rag_system import RAGSystem
from tests.fixtures import TEST_QUERIES, TestDataFixtures


class TestRAGSystem:

    @pytest.fixture
    def mock_config(self, test_config):
        """Use the test configuration"""
        return test_config

    @pytest.fixture
    def rag_system(self, mock_config):
        """Create RAGSystem with mocked dependencies"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
        ):

            # Setup mocks
            mock_doc_proc_instance = Mock()
            mock_vector_store_instance = Mock()
            mock_ai_gen_instance = Mock()
            mock_session_mgr_instance = Mock()

            mock_doc_proc.return_value = mock_doc_proc_instance
            mock_vector_store.return_value = mock_vector_store_instance
            mock_ai_gen.return_value = mock_ai_gen_instance
            mock_session_mgr.return_value = mock_session_mgr_instance

            system = RAGSystem(mock_config)

            # Store mocked components for easy access in tests
            system._mock_document_processor = mock_doc_proc_instance
            system._mock_vector_store = mock_vector_store_instance
            system._mock_ai_generator = mock_ai_gen_instance
            system._mock_session_manager = mock_session_mgr_instance

            return system

    def test_initialization(self, rag_system, mock_config):
        """Test RAGSystem initialization"""
        assert rag_system.config == mock_config
        assert hasattr(rag_system, "document_processor")
        assert hasattr(rag_system, "vector_store")
        assert hasattr(rag_system, "ai_generator")
        assert hasattr(rag_system, "session_manager")
        assert hasattr(rag_system, "tool_manager")
        assert hasattr(rag_system, "search_tool")
        assert hasattr(rag_system, "outline_tool")

        # Verify tools are registered
        assert len(rag_system.tool_manager.tools) == 2
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools

    def test_add_course_document_success(self, rag_system):
        """Test successful course document addition"""
        # Setup mocks
        mock_course = TestDataFixtures.create_sample_course()
        mock_chunks = TestDataFixtures.create_course_chunks()

        rag_system._mock_document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )

        # Execute
        result_course, chunk_count = rag_system.add_course_document(
            "/path/to/course.pdf"
        )

        # Verify
        assert result_course == mock_course
        assert chunk_count == len(mock_chunks)

        rag_system._mock_document_processor.process_course_document.assert_called_once_with(
            "/path/to/course.pdf"
        )
        rag_system._mock_vector_store.add_course_metadata.assert_called_once_with(
            mock_course
        )
        rag_system._mock_vector_store.add_course_content.assert_called_once_with(
            mock_chunks
        )

    def test_add_course_document_error(self, rag_system):
        """Test course document addition with error"""
        rag_system._mock_document_processor.process_course_document.side_effect = (
            Exception("Processing error")
        )

        result_course, chunk_count = rag_system.add_course_document("/invalid/path.pdf")

        assert result_course is None
        assert chunk_count == 0

        # Should not call vector store methods on error
        rag_system._mock_vector_store.add_course_metadata.assert_not_called()
        rag_system._mock_vector_store.add_course_content.assert_not_called()

    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    def test_add_course_folder_success(self, mock_listdir, mock_exists, rag_system):
        """Test successful course folder processing"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.txt", "invalid.jpg"]

        mock_course = TestDataFixtures.create_sample_course()
        mock_chunks = TestDataFixtures.create_course_chunks()

        rag_system._mock_document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )
        rag_system._mock_vector_store.get_existing_course_titles.return_value = []

        with patch("rag_system.os.path.isfile", return_value=True):
            courses, chunks = rag_system.add_course_folder("/docs/folder")

        assert courses == 2  # Should process 2 valid files
        assert chunks == len(mock_chunks) * 2

        # Should have been called twice (once for each valid file)
        assert (
            rag_system._mock_document_processor.process_course_document.call_count == 2
        )

    @patch("rag_system.os.path.exists")
    def test_add_course_folder_not_exists(self, mock_exists, rag_system):
        """Test course folder processing when folder doesn't exist"""
        mock_exists.return_value = False

        courses, chunks = rag_system.add_course_folder("/nonexistent/folder")

        assert courses == 0
        assert chunks == 0

    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    def test_add_course_folder_with_clear_existing(
        self, mock_listdir, mock_exists, rag_system
    ):
        """Test course folder processing with clear_existing=True"""
        mock_exists.return_value = True
        mock_listdir.return_value = []

        rag_system.add_course_folder("/docs/folder", clear_existing=True)

        rag_system._mock_vector_store.clear_all_data.assert_called_once()

    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    def test_add_course_folder_skip_existing(
        self, mock_listdir, mock_exists, rag_system
    ):
        """Test that existing courses are skipped"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf"]

        mock_course = TestDataFixtures.create_sample_course("Existing Course")
        mock_chunks = TestDataFixtures.create_course_chunks()

        rag_system._mock_document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )
        rag_system._mock_vector_store.get_existing_course_titles.return_value = [
            "Existing Course"
        ]

        with patch("rag_system.os.path.isfile", return_value=True):
            courses, chunks = rag_system.add_course_folder("/docs/folder")

        assert courses == 0  # Should skip existing course
        assert chunks == 0

        # Should not add to vector store
        rag_system._mock_vector_store.add_course_metadata.assert_not_called()
        rag_system._mock_vector_store.add_course_content.assert_not_called()

    def test_query_without_session(self, rag_system):
        """Test query processing without session ID"""
        rag_system._mock_ai_generator.generate_response.return_value = "AI response"
        rag_system.tool_manager.get_last_sources.return_value = ["Source 1", "Source 2"]

        response, sources = rag_system.query("What is machine learning?")

        assert response == "AI response"
        assert sources == ["Source 1", "Source 2"]

        # Verify AI generator was called correctly
        rag_system._mock_ai_generator.generate_response.assert_called_once()
        call_args = rag_system._mock_ai_generator.generate_response.call_args

        assert "What is machine learning?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] == rag_system.tool_manager.get_tool_definitions()
        assert call_args[1]["tool_manager"] == rag_system.tool_manager

    def test_query_with_session(self, rag_system):
        """Test query processing with session ID"""
        session_id = "test-session-123"
        conversation_history = "Previous conversation"

        rag_system._mock_session_manager.get_conversation_history.return_value = (
            conversation_history
        )
        rag_system._mock_ai_generator.generate_response.return_value = (
            "AI response with context"
        )
        rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = rag_system.query(
            "Follow up question", session_id=session_id
        )

        assert response == "AI response with context"

        # Verify session manager interactions
        rag_system._mock_session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )
        rag_system._mock_session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow up question", "AI response with context"
        )

        # Verify AI generator received conversation history
        call_args = rag_system._mock_ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == conversation_history

    def test_query_prompt_formatting(self, rag_system):
        """Test that query prompt is formatted correctly"""
        rag_system._mock_ai_generator.generate_response.return_value = "Response"

        rag_system.query("What are the main topics?")

        call_args = rag_system._mock_ai_generator.generate_response.call_args
        query_arg = call_args[1]["query"]

        assert "Answer this question about course materials:" in query_arg
        assert "What are the main topics?" in query_arg

    def test_sources_management(self, rag_system):
        """Test that sources are properly managed"""
        rag_system._mock_ai_generator.generate_response.return_value = "Response"
        rag_system.tool_manager.get_last_sources.return_value = ["Source 1"]

        response, sources = rag_system.query("Test query")

        # Verify sources were retrieved and reset
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()
        assert sources == ["Source 1"]

    def test_get_course_analytics(self, rag_system):
        """Test getting course analytics"""
        rag_system._mock_vector_store.get_course_count.return_value = 5
        rag_system._mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1",
            "Course 2",
            "Course 3",
            "Course 4",
            "Course 5",
        ]

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]

    def test_tool_manager_setup(self, rag_system):
        """Test that tool manager is set up correctly"""
        # Verify both tools are registered
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_definitions) == 2

        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_search_tool_integration(self, rag_system):
        """Test that search tool is properly integrated"""
        # The search tool should use the same vector store as the RAG system
        assert rag_system.search_tool.store == rag_system.vector_store

        # Tool should be registered in tool manager
        assert (
            rag_system.tool_manager.tools["search_course_content"]
            == rag_system.search_tool
        )

    def test_outline_tool_integration(self, rag_system):
        """Test that outline tool is properly integrated"""
        # The outline tool should use the same vector store as the RAG system
        assert rag_system.outline_tool.store == rag_system.vector_store

        # Tool should be registered in tool manager
        assert (
            rag_system.tool_manager.tools["get_course_outline"]
            == rag_system.outline_tool
        )

    @pytest.mark.integration
    def test_end_to_end_query_flow(self, rag_system):
        """Integration test for complete query flow"""
        session_id = "integration-test"
        query = "What is covered in the introduction?"

        # Setup expected flow
        rag_system._mock_session_manager.get_conversation_history.return_value = None
        rag_system._mock_ai_generator.generate_response.return_value = (
            "The introduction covers basic concepts."
        )
        rag_system.tool_manager.get_last_sources.return_value = ["Course - Lesson 1"]

        # Execute
        response, sources = rag_system.query(query, session_id)

        # Verify complete flow
        assert response == "The introduction covers basic concepts."
        assert sources == ["Course - Lesson 1"]

        # Verify all components were called
        rag_system._mock_session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )
        rag_system._mock_ai_generator.generate_response.assert_called_once()
        rag_system._mock_session_manager.add_exchange.assert_called_once_with(
            session_id, query, response
        )
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_with_different_types(self, rag_system):
        """Test different types of queries are handled appropriately"""
        rag_system._mock_ai_generator.generate_response.return_value = "Response"
        rag_system.tool_manager.get_last_sources.return_value = []

        # Test content query
        response, sources = rag_system.query(TEST_QUERIES["content_query"])
        assert response == "Response"

        # Test structure query
        response, sources = rag_system.query(TEST_QUERIES["course_structure_query"])
        assert response == "Response"

        # Test specific lesson query
        response, sources = rag_system.query(TEST_QUERIES["specific_lesson_query"])
        assert response == "Response"

        # All should have used the AI generator
        assert rag_system._mock_ai_generator.generate_response.call_count == 3

    def test_error_handling_in_query(self, rag_system):
        """Test error handling during query processing"""
        # Mock AI generator to raise exception
        rag_system._mock_ai_generator.generate_response.side_effect = Exception(
            "AI service error"
        )

        # Should not crash - let the exception propagate or handle gracefully
        # depending on the implementation choice
        with pytest.raises(Exception):
            rag_system.query("Test query")
