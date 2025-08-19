from typing import Any, Dict, List

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults


class TestDataFixtures:

    @staticmethod
    def create_sample_course(title: str = "Test Course") -> Course:
        return Course(
            title=title,
            instructor="Test Instructor",
            course_link=f"https://example.com/{title.lower().replace(' ', '-')}",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Introduction",
                    lesson_link=f"https://example.com/{title.lower().replace(' ', '-')}/lesson1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Advanced Topics",
                    lesson_link=f"https://example.com/{title.lower().replace(' ', '-')}/lesson2",
                ),
            ],
        )

    @staticmethod
    def create_multiple_courses() -> List[Course]:
        return [
            TestDataFixtures.create_sample_course(
                "Building Towards Computer Use with Anthropic"
            ),
            TestDataFixtures.create_sample_course(
                "MCP: Build Rich-Context AI Apps with Anthropic"
            ),
            TestDataFixtures.create_sample_course(
                "Advanced Retrieval for AI with Chroma"
            ),
        ]

    @staticmethod
    def create_course_chunks(course_title: str = "Test Course") -> List[CourseChunk]:
        return [
            CourseChunk(
                course_title=course_title,
                lesson_number=1,
                chunk_index=0,
                content="This is an introduction to the course. It covers the basic concepts and overview.",
            ),
            CourseChunk(
                course_title=course_title,
                lesson_number=1,
                chunk_index=1,
                content="In this section, we dive deeper into the fundamentals and core principles.",
            ),
            CourseChunk(
                course_title=course_title,
                lesson_number=2,
                chunk_index=2,
                content="Advanced techniques are covered in this lesson, including practical examples.",
            ),
            CourseChunk(
                course_title=course_title,
                lesson_number=2,
                chunk_index=3,
                content="This chunk contains implementation details and best practices for the course topic.",
            ),
        ]

    @staticmethod
    def create_search_results_with_content() -> SearchResults:
        return SearchResults(
            documents=[
                "This is an introduction to the course. It covers the basic concepts and overview.",
                "In this section, we dive deeper into the fundamentals and core principles.",
                "Advanced techniques are covered in this lesson, including practical examples.",
            ],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 1},
                {"course_title": "Test Course", "lesson_number": 2, "chunk_index": 2},
            ],
            distances=[0.1, 0.15, 0.2],
        )

    @staticmethod
    def create_course_metadata() -> List[Dict[str, Any]]:
        return [
            {
                "title": "Building Towards Computer Use with Anthropic",
                "instructor": "Colt Steele",
                "course_link": "https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
                "lessons": [
                    {
                        "lesson_number": 1,
                        "lesson_title": "Introduction",
                        "lesson_link": "https://example.com/lesson1",
                    },
                    {
                        "lesson_number": 2,
                        "lesson_title": "Advanced Topics",
                        "lesson_link": "https://example.com/lesson2",
                    },
                ],
            },
            {
                "title": "MCP: Build Rich-Context AI Apps with Anthropic",
                "instructor": "Test Instructor",
                "course_link": "https://example.com/mcp-course",
                "lessons": [
                    {
                        "lesson_number": 1,
                        "lesson_title": "MCP Basics",
                        "lesson_link": "https://example.com/mcp-lesson1",
                    }
                ],
            },
        ]

    @staticmethod
    def create_anthropic_tool_response():
        from unittest.mock import Mock

        # Mock tool use response
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test query"}

        response = Mock()
        response.content = [tool_block]
        response.stop_reason = "tool_use"

        return response

    @staticmethod
    def create_anthropic_final_response():
        from unittest.mock import Mock

        content_block = Mock()
        content_block.text = "This is the final response from the AI after using tools."

        response = Mock()
        response.content = [content_block]
        response.stop_reason = "end_turn"

        return response

    @staticmethod
    def create_anthropic_text_response(text: str = "Default text response"):
        from unittest.mock import Mock

        content_block = Mock()
        content_block.text = text

        response = Mock()
        response.content = [content_block]
        response.stop_reason = "end_turn"

        return response


# Common test queries for consistent testing
TEST_QUERIES = {
    "content_query": "What are the main concepts covered in the course?",
    "specific_lesson_query": "Tell me about lesson 1",
    "course_structure_query": "What lessons are in this course?",
    "invalid_course_query": "What is in the non-existent course?",
    "empty_query": "",
    "complex_query": "How do the advanced techniques in lesson 2 relate to the fundamentals from lesson 1?",
}

# Expected responses for validation
EXPECTED_RESPONSES = {
    "successful_search": "[Test Course - Lesson 1]\nThis is an introduction to the course. It covers the basic concepts and overview.\n\n[Test Course - Lesson 1]\nIn this section, we dive deeper into the fundamentals and core principles.",
    "no_results": "No relevant content found.",
    "error_response": "Database connection failed",
}
