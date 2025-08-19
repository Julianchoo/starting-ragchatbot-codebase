import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
from tests.fixtures import TestDataFixtures
from vector_store import SearchResults, VectorStore


class TestSearchResults:

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"meta1": "value1"}, {"meta2": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"meta1": "value1"}, {"meta2": "value2"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Test error message")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"

    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        non_empty_results = SearchResults(
            documents=["doc"], metadata=[{}], distances=[0.1]
        )

        assert empty_results.is_empty() is True
        assert non_empty_results.is_empty() is False


class TestVectorStore:

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for ChromaDB testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_chroma_client(self):
        """Create mock ChromaDB client"""
        with patch("vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock collections
            mock_catalog = Mock()
            mock_content = Mock()
            mock_client.get_or_create_collection.side_effect = [
                mock_catalog,
                mock_content,
            ]

            yield mock_client, mock_catalog, mock_content

    @pytest.fixture
    def mock_embedding_function(self):
        """Create mock embedding function"""
        with patch(
            "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ) as mock_embedding:
            yield mock_embedding.return_value

    @pytest.fixture
    def vector_store(
        self, temp_chroma_path, mock_chroma_client, mock_embedding_function
    ):
        """Create VectorStore instance with mocked dependencies"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        store = VectorStore(temp_chroma_path, "test-model", max_results=5)
        store.course_catalog = mock_catalog
        store.course_content = mock_content
        return store, mock_catalog, mock_content

    def test_initialization(
        self, temp_chroma_path, mock_chroma_client, mock_embedding_function
    ):
        """Test VectorStore initialization"""
        mock_client, mock_catalog, mock_content = mock_chroma_client

        store = VectorStore(temp_chroma_path, "test-model", max_results=10)

        assert store.max_results == 10
        mock_client.get_or_create_collection.assert_any_call(
            name="course_catalog", embedding_function=mock_embedding_function
        )
        mock_client.get_or_create_collection.assert_any_call(
            name="course_content", embedding_function=mock_embedding_function
        )

    def test_search_successful(self, vector_store):
        """Test successful search operation"""
        store, mock_catalog, mock_content = vector_store

        # Mock successful search results
        mock_content.query.return_value = {
            "documents": [["Test document content"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        results = store.search("test query")

        mock_content.query.assert_called_once_with(
            query_texts=["test query"], n_results=5, where=None
        )
        assert not results.is_empty()
        assert results.documents == ["Test document content"]
        assert results.error is None

    def test_search_with_course_name(self, vector_store):
        """Test search with course name filter"""
        store, mock_catalog, mock_content = vector_store

        # Mock course resolution
        mock_catalog.query.return_value = {
            "documents": [["Test Course"]],
            "metadatas": [[{"title": "Test Course"}]],
        }

        # Mock content search
        mock_content.query.return_value = {
            "documents": [["Test content"]],
            "metadatas": [[{"course_title": "Test Course"}]],
            "distances": [[0.1]],
        }

        results = store.search("test query", course_name="Test Course")

        # Verify course resolution was called
        mock_catalog.query.assert_called_once_with(
            query_texts=["Test Course"], n_results=1
        )

        # Verify content search with filter
        mock_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"course_title": "Test Course"},
        )

    def test_search_with_lesson_number(self, vector_store):
        """Test search with lesson number filter"""
        store, mock_catalog, mock_content = vector_store

        mock_content.query.return_value = {
            "documents": [["Lesson content"]],
            "metadatas": [[{"lesson_number": 2}]],
            "distances": [[0.1]],
        }

        results = store.search("test query", lesson_number=2)

        mock_content.query.assert_called_once_with(
            query_texts=["test query"], n_results=5, where={"lesson_number": 2}
        )

    def test_search_with_both_filters(self, vector_store):
        """Test search with both course name and lesson number filters"""
        store, mock_catalog, mock_content = vector_store

        # Mock course resolution
        mock_catalog.query.return_value = {
            "documents": [["Test Course"]],
            "metadatas": [[{"title": "Test Course"}]],
        }

        mock_content.query.return_value = {
            "documents": [["Filtered content"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        results = store.search("test query", course_name="Test Course", lesson_number=1)

        expected_filter = {
            "$and": [{"course_title": "Test Course"}, {"lesson_number": 1}]
        }

        mock_content.query.assert_called_once_with(
            query_texts=["test query"], n_results=5, where=expected_filter
        )

    def test_search_course_not_found(self, vector_store):
        """Test search when course name cannot be resolved"""
        store, mock_catalog, mock_content = vector_store

        # Mock failed course resolution
        mock_catalog.query.return_value = {"documents": [[]], "metadatas": [[]]}

        results = store.search("test query", course_name="Nonexistent Course")

        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()

    def test_search_exception_handling(self, vector_store):
        """Test search error handling"""
        store, mock_catalog, mock_content = vector_store

        mock_content.query.side_effect = Exception("Database error")

        results = store.search("test query")

        assert results.error == "Search error: Database error"
        assert results.is_empty()

    def test_resolve_course_name_success(self, vector_store):
        """Test successful course name resolution"""
        store, mock_catalog, mock_content = vector_store

        mock_catalog.query.return_value = {
            "documents": [["Found course"]],
            "metadatas": [[{"title": "Exact Course Title"}]],
        }

        resolved = store._resolve_course_name("Partial Course")

        assert resolved == "Exact Course Title"
        mock_catalog.query.assert_called_once_with(
            query_texts=["Partial Course"], n_results=1
        )

    def test_resolve_course_name_not_found(self, vector_store):
        """Test course name resolution when no match found"""
        store, mock_catalog, mock_content = vector_store

        mock_catalog.query.return_value = {"documents": [[]], "metadatas": [[]]}

        resolved = store._resolve_course_name("Nonexistent Course")

        assert resolved is None

    def test_build_filter_combinations(self, vector_store):
        """Test different filter combinations"""
        store, mock_catalog, mock_content = vector_store

        # No filters
        filter_result = store._build_filter(None, None)
        assert filter_result is None

        # Course only
        filter_result = store._build_filter("Test Course", None)
        assert filter_result == {"course_title": "Test Course"}

        # Lesson only
        filter_result = store._build_filter(None, 2)
        assert filter_result == {"lesson_number": 2}

        # Both filters
        filter_result = store._build_filter("Test Course", 2)
        expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 2}]}
        assert filter_result == expected

    def test_add_course_metadata(self, vector_store):
        """Test adding course metadata"""
        store, mock_catalog, mock_content = vector_store

        course = TestDataFixtures.create_sample_course("Test Course")

        store.add_course_metadata(course)

        # Verify the catalog add was called
        mock_catalog.add.assert_called_once()
        call_args = mock_catalog.add.call_args

        # Check arguments
        assert call_args[1]["documents"] == ["Test Course"]
        assert call_args[1]["ids"] == ["Test Course"]

        # Check metadata structure
        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == "Test Course"
        assert metadata["instructor"] == "Test Instructor"
        assert metadata["lesson_count"] == 2
        assert "lessons_json" in metadata

    def test_add_course_content(self, vector_store):
        """Test adding course content chunks"""
        store, mock_catalog, mock_content = vector_store

        chunks = TestDataFixtures.create_course_chunks("Test Course")

        store.add_course_content(chunks)

        mock_content.add.assert_called_once()
        call_args = mock_content.add.call_args

        # Verify documents were added correctly
        assert len(call_args[1]["documents"]) == len(chunks)
        assert len(call_args[1]["metadatas"]) == len(chunks)
        assert len(call_args[1]["ids"]) == len(chunks)

        # Check first chunk metadata
        first_metadata = call_args[1]["metadatas"][0]
        assert first_metadata["course_title"] == "Test Course"
        assert first_metadata["lesson_number"] == 1
        assert first_metadata["chunk_index"] == 0

    def test_add_empty_course_content(self, vector_store):
        """Test adding empty course content"""
        store, mock_catalog, mock_content = vector_store

        store.add_course_content([])

        # Should not call add when chunks are empty
        mock_content.add.assert_not_called()

    def test_clear_all_data(self, vector_store):
        """Test clearing all data"""
        store, mock_catalog, mock_content = vector_store
        mock_client = store.client

        store.clear_all_data()

        # Verify collections were deleted
        mock_client.delete_collection.assert_any_call("course_catalog")
        mock_client.delete_collection.assert_any_call("course_content")

        # Verify collections were recreated
        assert (
            mock_client.get_or_create_collection.call_count >= 4
        )  # 2 initial + 2 recreated

    def test_get_existing_course_titles(self, vector_store):
        """Test getting existing course titles"""
        store, mock_catalog, mock_content = vector_store

        mock_catalog.get.return_value = {"ids": ["Course 1", "Course 2", "Course 3"]}

        titles = store.get_existing_course_titles()

        assert titles == ["Course 1", "Course 2", "Course 3"]
        mock_catalog.get.assert_called_once()

    def test_get_existing_course_titles_error(self, vector_store):
        """Test error handling when getting course titles"""
        store, mock_catalog, mock_content = vector_store

        mock_catalog.get.side_effect = Exception("Database error")

        titles = store.get_existing_course_titles()

        assert titles == []

    def test_get_course_count(self, vector_store):
        """Test getting course count"""
        store, mock_catalog, mock_content = vector_store

        mock_catalog.get.return_value = {"ids": ["Course 1", "Course 2"]}

        count = store.get_course_count()

        assert count == 2

    def test_get_all_courses_metadata(self, vector_store):
        """Test getting all courses metadata"""
        store, mock_catalog, mock_content = vector_store

        mock_catalog.get.return_value = {
            "metadatas": [
                {
                    "title": "Course 1",
                    "instructor": "Instructor 1",
                    "lessons_json": '[{"lesson_number": 1, "lesson_title": "Intro"}]',
                }
            ]
        }

        metadata = store.get_all_courses_metadata()

        assert len(metadata) == 1
        assert metadata[0]["title"] == "Course 1"
        assert "lessons" in metadata[0]
        assert "lessons_json" not in metadata[0]  # Should be parsed and removed

    def test_get_course_link(self, vector_store):
        """Test getting course link"""
        store, mock_catalog, mock_content = vector_store

        mock_catalog.get.return_value = {
            "metadatas": [{"course_link": "https://example.com/course"}]
        }

        link = store.get_course_link("Test Course")

        assert link == "https://example.com/course"
        mock_catalog.get.assert_called_once_with(ids=["Test Course"])

    def test_get_lesson_link(self, vector_store):
        """Test getting lesson link"""
        store, mock_catalog, mock_content = vector_store

        lessons_json = '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}, {"lesson_number": 2, "lesson_link": "https://example.com/lesson2"}]'
        mock_catalog.get.return_value = {"metadatas": [{"lessons_json": lessons_json}]}

        link = store.get_lesson_link("Test Course", 2)

        assert link == "https://example.com/lesson2"

    def test_get_lesson_link_not_found(self, vector_store):
        """Test getting lesson link when lesson doesn't exist"""
        store, mock_catalog, mock_content = vector_store

        lessons_json = (
            '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}]'
        )
        mock_catalog.get.return_value = {"metadatas": [{"lessons_json": lessons_json}]}

        link = store.get_lesson_link("Test Course", 99)

        assert link is None
