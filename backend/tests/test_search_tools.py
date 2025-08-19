import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults
from tests.fixtures import TestDataFixtures, TEST_QUERIES, EXPECTED_RESPONSES


class TestCourseSearchTool:
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        mock_store = Mock()
        mock_store.search = Mock()
        mock_store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
        return mock_store
    
    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create a CourseSearchTool with mocked vector store"""
        return CourseSearchTool(mock_vector_store)
    
    def test_get_tool_definition(self, search_tool):
        """Test that tool definition is correctly structured"""
        definition = search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]
    
    def test_execute_successful_search(self, search_tool, mock_vector_store):
        """Test successful search with results"""
        # Setup mock to return successful results
        mock_results = TestDataFixtures.create_search_results_with_content()
        mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = search_tool.execute("test query")
        
        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result format
        assert isinstance(result, str)
        assert "[Test Course - Lesson 1]" in result
        assert "introduction to the course" in result
        assert len(search_tool.last_sources) > 0
    
    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test search with course name filter"""
        mock_results = TestDataFixtures.create_search_results_with_content()
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("test query", course_name="Test Course")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=None
        )
        assert isinstance(result, str)
    
    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test search with lesson number filter"""
        mock_results = TestDataFixtures.create_search_results_with_content()
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("test query", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=1
        )
        assert isinstance(result, str)
    
    def test_execute_with_both_filters(self, search_tool, mock_vector_store):
        """Test search with both course name and lesson number filters"""
        mock_results = TestDataFixtures.create_search_results_with_content()
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("test query", course_name="Test Course", lesson_number=2)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query", 
            course_name="Test Course",
            lesson_number=2
        )
        assert isinstance(result, str)
    
    def test_execute_empty_results(self, search_tool, mock_vector_store):
        """Test search that returns no results"""
        # Setup mock to return empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("nonexistent query")
        
        assert result == "No relevant content found."
        assert len(search_tool.last_sources) == 0
    
    def test_execute_empty_results_with_filters(self, search_tool, mock_vector_store):
        """Test empty results with filter information in response"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("test", course_name="Nonexistent Course", lesson_number=5)
        
        expected = "No relevant content found in course 'Nonexistent Course' in lesson 5."
        assert result == expected
    
    def test_execute_error_handling(self, search_tool, mock_vector_store):
        """Test handling of search errors"""
        # Setup mock to return error results
        error_results = SearchResults(
            documents=[], 
            metadata=[], 
            distances=[], 
            error="Database connection failed"
        )
        mock_vector_store.search.return_value = error_results
        
        result = search_tool.execute("test query")
        
        assert result == "Database connection failed"
        assert len(search_tool.last_sources) == 0
    
    def test_format_results_with_lesson_links(self, search_tool, mock_vector_store):
        """Test result formatting includes lesson links"""
        mock_results = TestDataFixtures.create_search_results_with_content()
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        result = search_tool.execute("test query")
        
        # Verify lesson link was requested
        mock_vector_store.get_lesson_link.assert_called()
        
        # Verify sources contain lesson links
        assert len(search_tool.last_sources) > 0
        # Some sources should contain the pipe separator for links
        has_link = any("|" in source for source in search_tool.last_sources)
        assert has_link or not mock_vector_store.get_lesson_link.return_value
    
    def test_sources_tracking(self, search_tool, mock_vector_store):
        """Test that sources are properly tracked"""
        mock_results = TestDataFixtures.create_search_results_with_content()
        mock_vector_store.search.return_value = mock_results
        
        # Execute search
        search_tool.execute("test query")
        
        # Verify sources are tracked
        assert len(search_tool.last_sources) == len(mock_results.documents)
        
        # Sources should contain course and lesson info
        for source in search_tool.last_sources:
            assert "Test Course" in source
            assert "Lesson" in source


class TestCourseOutlineTool:
    
    @pytest.fixture
    def mock_vector_store(self):
        mock_store = Mock()
        mock_store.get_all_courses_metadata = Mock()
        return mock_store
    
    @pytest.fixture 
    def outline_tool(self, mock_vector_store):
        return CourseOutlineTool(mock_vector_store)
    
    def test_get_tool_definition(self, outline_tool):
        """Test outline tool definition structure"""
        definition = outline_tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "course_title" in definition["input_schema"]["properties"]
        assert "course_title" in definition["input_schema"]["required"]
    
    def test_execute_successful_outline(self, outline_tool, mock_vector_store):
        """Test successful course outline retrieval"""
        mock_metadata = TestDataFixtures.create_course_metadata()
        mock_vector_store.get_all_courses_metadata.return_value = mock_metadata
        
        result = outline_tool.execute("Building Towards Computer")
        
        assert "Building Towards Computer Use with Anthropic" in result
        assert "Colt Steele" in result
        assert "Course URL" in result
        assert "Course Lessons" in result
        assert "1. Introduction" in result
    
    def test_execute_no_courses(self, outline_tool, mock_vector_store):
        """Test when no courses exist"""
        mock_vector_store.get_all_courses_metadata.return_value = []
        
        result = outline_tool.execute("Any Course")
        
        assert result == "No courses found in the database."
    
    def test_execute_no_match(self, outline_tool, mock_vector_store):
        """Test when no course matches the query"""
        mock_metadata = TestDataFixtures.create_course_metadata()
        mock_vector_store.get_all_courses_metadata.return_value = mock_metadata
        
        result = outline_tool.execute("Nonexistent Course")
        
        assert "No course found matching 'Nonexistent Course'" in result
        assert "Available courses:" in result
    
    def test_execute_error_handling(self, outline_tool, mock_vector_store):
        """Test error handling in outline tool"""
        mock_vector_store.get_all_courses_metadata.side_effect = Exception("Database error")
        
        result = outline_tool.execute("Any Course")
        
        assert "Error retrieving course outline: Database error" in result


class TestToolManager:
    
    @pytest.fixture
    def tool_manager(self):
        return ToolManager()
    
    @pytest.fixture
    def mock_tool(self):
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "A test tool"
        }
        mock_tool.execute.return_value = "Tool result"
        return mock_tool
    
    def test_register_tool(self, tool_manager, mock_tool):
        """Test tool registration"""
        tool_manager.register_tool(mock_tool)
        
        assert "test_tool" in tool_manager.tools
        assert tool_manager.tools["test_tool"] == mock_tool
    
    def test_register_tool_no_name(self, tool_manager):
        """Test error when registering tool without name"""
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "No name"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            tool_manager.register_tool(mock_tool)
    
    def test_get_tool_definitions(self, tool_manager, mock_tool):
        """Test getting all tool definitions"""
        tool_manager.register_tool(mock_tool)
        
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"
    
    def test_execute_tool(self, tool_manager, mock_tool):
        """Test executing a registered tool"""
        tool_manager.register_tool(mock_tool)
        
        result = tool_manager.execute_tool("test_tool", param="value")
        
        mock_tool.execute.assert_called_once_with(param="value")
        assert result == "Tool result"
    
    def test_execute_nonexistent_tool(self, tool_manager):
        """Test executing a tool that doesn't exist"""
        result = tool_manager.execute_tool("nonexistent_tool")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, tool_manager, mock_vector_store):
        """Test getting sources from search tools"""
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = ["Source 1", "Source 2"]
        
        tool_manager.register_tool(search_tool)
        
        sources = tool_manager.get_last_sources()
        assert sources == ["Source 1", "Source 2"]
    
    def test_reset_sources(self, tool_manager, mock_vector_store):
        """Test resetting sources from all tools"""
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = ["Source 1", "Source 2"]
        
        tool_manager.register_tool(search_tool)
        tool_manager.reset_sources()
        
        assert search_tool.last_sources == []