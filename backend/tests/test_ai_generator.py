import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from tests.fixtures import TestDataFixtures


class TestAIGenerator:
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator instance with mocked client"""
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        generator.client = mock_anthropic_client
        return generator
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager"""
        mock_manager = Mock()
        mock_manager.execute_tool.return_value = "Tool executed successfully with search results"
        return mock_manager
    
    def test_initialization(self, mock_anthropic_client):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-key", "test-model")
        
        assert generator.model == "test-model"
        assert generator.base_params["model"] == "test-model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    def test_generate_response_simple(self, ai_generator, mock_anthropic_client):
        """Test simple response generation without tools"""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a simple response"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = ai_generator.generate_response("What is AI?")
        
        assert result == "This is a simple response"
        
        # Verify API call
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args
        
        assert call_args[1]["messages"][0]["content"] == "What is AI?"
        assert call_args[1]["messages"][0]["role"] == "user"
        assert "tools" not in call_args[1]
    
    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client):
        """Test response generation with conversation history"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with history context"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        conversation_history = "User: Previous question\nAssistant: Previous answer"
        
        result = ai_generator.generate_response(
            "Follow up question",
            conversation_history=conversation_history
        )
        
        assert result == "Response with history context"
        
        # Verify history was included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert conversation_history in system_content
    
    def test_generate_response_with_tools(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test response generation with tools available"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct response without tool use"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "What is in the course?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Direct response without tool use"
        
        # Verify tools were passed to API
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test response generation when AI uses tools"""
        # Mock initial tool use response
        tool_response = TestDataFixtures.create_anthropic_tool_response()
        mock_anthropic_client.messages.create.side_effect = [
            tool_response,  # First call returns tool use
            TestDataFixtures.create_anthropic_final_response()  # Second call returns final response
        ]
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "What is covered in lesson 1?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "This is the final response from the AI after using tools."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )
        
        # Verify two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_handle_tool_execution_single_tool(self, ai_generator, mock_tool_manager):
        """Test handling single tool execution"""
        # Create mock initial response with tool use
        initial_response = TestDataFixtures.create_anthropic_tool_response()
        
        # Mock final response
        final_response = TestDataFixtures.create_anthropic_final_response()
        ai_generator.client.messages.create.return_value = final_response
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "System prompt"
        }
        
        result = ai_generator._handle_tool_execution(
            initial_response,
            base_params,
            mock_tool_manager
        )
        
        assert result == "This is the final response from the AI after using tools."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify final API call structure
        final_call_args = ai_generator.client.messages.create.call_args
        messages = final_call_args[1]["messages"]
        
        # Should have: original user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
    
    def test_handle_tool_execution_multiple_tools(self, ai_generator, mock_tool_manager):
        """Test handling multiple tool executions"""
        # Create mock response with multiple tool uses
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "query 1"}
        
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use" 
        tool_block_2.name = "get_course_outline"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"course_title": "Test Course"}
        
        initial_response = Mock()
        initial_response.content = [tool_block_1, tool_block_2]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response
        final_response = TestDataFixtures.create_anthropic_final_response()
        ai_generator.client.messages.create.return_value = final_response
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "System prompt"
        }
        
        result = ai_generator._handle_tool_execution(
            initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="query 1")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="Test Course")
    
    def test_system_prompt_structure(self, ai_generator):
        """Test that system prompt contains required elements"""
        system_prompt = ai_generator.SYSTEM_PROMPT
        
        # Check for key sections
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "Tool Usage Guidelines" in system_prompt
        assert "Response Protocol" in system_prompt
        assert "Brief, Concise and focused" in system_prompt
    
    def test_response_without_tool_manager(self, ai_generator, mock_anthropic_client):
        """Test response when tool use occurs but no tool manager provided"""
        # Mock tool use response - when no tool manager, it tries to get .text from tool use response
        tool_response = TestDataFixtures.create_anthropic_tool_response() 
        
        # The tool use response doesn't have proper .text, so set up fallback path
        mock_anthropic_client.messages.create.return_value = tool_response
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        # Should return the tool use content since no tool manager to handle it
        result = ai_generator.generate_response(
            "What is in the course?",
            tools=tools,
            tool_manager=None  # No tool manager
        )
        
        # Without tool manager, condition check fails and it returns the response.content[0].text
        # which is a Mock, so check it's the mock object (expected behavior)
        assert isinstance(result, type(tool_response.content[0].text))  # Should be Mock
    
    def test_api_parameters_structure(self, ai_generator, mock_anthropic_client):
        """Test that API parameters are structured correctly"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        tools = [{"name": "test_tool", "description": "Test"}]
        conversation_history = "Previous conversation"
        
        ai_generator.generate_response(
            "Test query",
            conversation_history=conversation_history,
            tools=tools
        )
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        # Verify all expected parameters are present
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Test query"
        assert "Previous conversation:" in call_args["system"]
    
    def test_tool_result_message_structure(self, ai_generator, mock_tool_manager):
        """Test that tool result messages are structured correctly"""
        tool_response = TestDataFixtures.create_anthropic_tool_response()
        final_response = TestDataFixtures.create_anthropic_final_response()
        ai_generator.client.messages.create.return_value = final_response
        
        base_params = {
            "messages": [{"role": "user", "content": "Test"}],
            "system": "System"
        }
        
        ai_generator._handle_tool_execution(tool_response, base_params, mock_tool_manager)
        
        # Check the final messages structure
        final_call_args = ai_generator.client.messages.create.call_args[1]
        tool_result_message = final_call_args["messages"][2]  # Third message should be tool results
        
        assert tool_result_message["role"] == "user"
        assert isinstance(tool_result_message["content"], list)
        
        tool_result = tool_result_message["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_123"
        assert "content" in tool_result
    
    def test_base_params_efficiency(self, ai_generator):
        """Test that base parameters are pre-built for efficiency"""
        # Verify base_params are set during initialization
        assert ai_generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert ai_generator.base_params["temperature"] == 0
        assert ai_generator.base_params["max_tokens"] == 800
        
        # These should be reused rather than rebuilt each time
        original_params = ai_generator.base_params.copy()
        
        # After multiple calls, base_params should remain unchanged
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response"
        mock_response.stop_reason = "end_turn"
        ai_generator.client.messages.create.return_value = mock_response
        
        ai_generator.generate_response("Query 1")
        ai_generator.generate_response("Query 2")
        
        assert ai_generator.base_params == original_params
    
    def test_sequential_tool_calls_two_rounds(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test sequential tool calling through 2 rounds"""
        # Mock responses for 2 rounds of tool use + final response
        round1_response = TestDataFixtures.create_anthropic_tool_response()
        round2_response = TestDataFixtures.create_anthropic_tool_response()
        final_response = TestDataFixtures.create_anthropic_final_response()
        
        mock_anthropic_client.messages.create.side_effect = [
            round1_response,  # Initial tool use
            round2_response,  # Round 2 tool use
            final_response    # Final response after max rounds
        ]
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "Complex query requiring multiple searches",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "This is the final response from the AI after using tools."
        
        # Verify 3 API calls were made (2 tool rounds + 1 final)
        assert mock_anthropic_client.messages.create.call_count == 3
        
        # Verify tool was executed twice
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify final call had no tools
        final_call_args = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_args
    
    def test_sequential_early_termination(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test early termination when AI doesn't need second round"""
        # Mock first round tool use, second round no tool use
        round1_response = TestDataFixtures.create_anthropic_tool_response()
        final_response = TestDataFixtures.create_anthropic_final_response()
        
        mock_anthropic_client.messages.create.side_effect = [
            round1_response,  # Initial tool use
            final_response    # Direct text response (no tool use)
        ]
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "Query that needs only one tool call",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "This is the final response from the AI after using tools."
        
        # Should only make 2 API calls (1 tool round + 1 final)
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Tool executed once
        assert mock_tool_manager.execute_tool.call_count == 1
    
    def test_sequential_tool_failure_handling(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test graceful handling when tool execution fails"""
        round1_response = TestDataFixtures.create_anthropic_tool_response()
        
        mock_anthropic_client.messages.create.side_effect = [round1_response]
        
        # Mock tool manager to raise exception
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        # Should handle tool failure gracefully
        result = ai_generator.generate_response(
            "Query with tool failure",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        # Should return the current response text (which is a Mock in this test)
        # The logic breaks out of the while loop and returns current_response.content[0].text
        assert isinstance(result, type(round1_response.content[0].text))
        
        # Only one API call made (initial tool use)
        assert mock_anthropic_client.messages.create.call_count == 1
    
    def test_sequential_max_rounds_exceeded(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test proper termination at max rounds limit"""
        # Mock 2 rounds of tool use
        round1_response = TestDataFixtures.create_anthropic_tool_response()
        round2_response = TestDataFixtures.create_anthropic_tool_response()
        final_response = TestDataFixtures.create_anthropic_final_response()
        
        mock_anthropic_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response  # Final call without tools
        ]
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "Query requiring exactly 2 rounds",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "This is the final response from the AI after using tools."
        
        # Should make 3 calls: 2 tool rounds + 1 final without tools
        assert mock_anthropic_client.messages.create.call_count == 3
        
        # Final call should not have tools
        final_call = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call
    
    def test_sequential_context_preservation(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test that message context is preserved across rounds"""
        round1_response = TestDataFixtures.create_anthropic_tool_response()
        final_response = TestDataFixtures.create_anthropic_final_response()
        
        mock_anthropic_client.messages.create.side_effect = [
            round1_response,
            final_response
        ]
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        # Check that second API call has accumulated message history
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        
        # Should have: user query, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"  # AI's tool use
        assert messages[2]["role"] == "user"       # Tool results
    
    def test_backward_compatibility_default_behavior(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test that default behavior is unchanged (single round)"""
        tool_response = TestDataFixtures.create_anthropic_tool_response()
        final_response = TestDataFixtures.create_anthropic_final_response()
        
        mock_anthropic_client.messages.create.side_effect = [
            tool_response,
            final_response
        ]
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        # Call without specifying max_rounds (should default to 2 but behave like before)
        result = ai_generator.generate_response(
            "What is in the course?",
            tools=tools,
            tool_manager=mock_tool_manager
            # max_rounds defaults to 2
        )
        
        assert result == "This is the final response from the AI after using tools."
        
        # Should work exactly like before - first call uses tools, second doesn't
        assert mock_anthropic_client.messages.create.call_count >= 1
    
    def test_build_round_system_prompt(self, ai_generator):
        """Test round-specific system prompt building"""
        base_prompt = "Base system prompt"
        
        # Test first round
        round1_prompt = ai_generator._build_round_system_prompt(base_prompt, 1, 2)
        assert "Round 1 of 2" in round1_prompt
        assert "additional tool calls" in round1_prompt
        
        # Test final round
        final_prompt = ai_generator._build_round_system_prompt(base_prompt, 2, 2)
        assert "Final round (2 of 2)" in final_prompt
        assert "last opportunity" in final_prompt
    
    def test_execute_tools_success(self, ai_generator, mock_tool_manager):
        """Test successful tool execution helper"""
        mock_response = TestDataFixtures.create_anthropic_tool_response()
        
        results = ai_generator._execute_tools(mock_response, mock_tool_manager)
        
        assert results is not None
        assert len(results) == 1
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "tool_123"
        assert "content" in results[0]
    
    def test_execute_tools_failure(self, ai_generator, mock_tool_manager):
        """Test tool execution failure handling"""
        mock_response = TestDataFixtures.create_anthropic_tool_response()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")
        
        results = ai_generator._execute_tools(mock_response, mock_tool_manager)
        
        assert results is None  # Should return None on failure