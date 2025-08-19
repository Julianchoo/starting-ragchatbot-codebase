from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI with lesson link if available
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
                # Try to get lesson link from vector store
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
                if lesson_link:
                    # Embed the link invisibly in the source text
                    source += f"|{lesson_link}"
            sources.append(source)

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outlines with lesson lists"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get complete course outline including course title, link, and all lessons",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title to get outline for (partial matches work, e.g. 'MCP', 'Introduction')",
                    }
                },
                "required": ["course_title"],
            },
        }

    def execute(self, course_title: str) -> str:
        """
        Execute the outline tool to get course structure.

        Args:
            course_title: Course title to search for

        Returns:
            Formatted course outline or error message
        """
        try:
            # Get all courses metadata
            all_courses = self.store.get_all_courses_metadata()

            if not all_courses:
                return "No courses found in the database."

            # Find best matching course using fuzzy matching
            best_match = self._find_best_course_match(course_title, all_courses)

            if not best_match:
                return f"No course found matching '{course_title}'. Available courses: {', '.join([course['title'] for course in all_courses])}"

            # Format and return the course outline
            return self._format_course_outline(best_match)

        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"

    def _find_best_course_match(
        self, search_title: str, courses: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching course using simple string matching"""
        search_lower = search_title.lower()

        # First try exact match
        for course in courses:
            if course["title"].lower() == search_lower:
                return course

        # Then try partial match
        for course in courses:
            if (
                search_lower in course["title"].lower()
                or course["title"].lower() in search_lower
            ):
                return course

        return None

    def _format_course_outline(self, course: Dict[str, Any]) -> str:
        """Format course outline for display"""
        outline = f"**{course['title']}**\n\n"

        if course.get("instructor"):
            outline += f"**Instructor:** {course['instructor']}\n"

        if course.get("course_link"):
            # Format URL to prevent auto-linking by removing protocol and using code formatting
            url = course["course_link"]
            # Remove protocol and use code block formatting
            clean_url = url.replace("https://", "").replace("http://", "")
            outline += f"**Course URL:** `{clean_url}`\n"

        outline += "\n**Course Lessons:**\n"

        # Get lessons from the metadata
        lessons = course.get("lessons", [])
        if lessons:
            for lesson in sorted(lessons, key=lambda x: x.get("lesson_number", 0)):
                lesson_num = lesson.get("lesson_number")
                lesson_title = lesson.get("lesson_title")
                if lesson_num is not None and lesson_title:
                    outline += f"{lesson_num}. {lesson_title}\n"
        else:
            outline += "No lessons found for this course.\n"

        return outline


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
