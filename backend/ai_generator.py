import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **search_course_content**: For questions about specific course content or detailed educational materials
2. **get_course_outline**: For questions about course structure, lesson lists, or course overviews

Tool Usage Guidelines:
- **Course outline queries**: Use get_course_outline for questions about course structure, lesson lists, course overviews, or "what's in this course"
- **Content questions**: Use search_course_content for specific educational content within courses
- **Sequential tool usage**: Use multiple rounds of tool calls for complex queries requiring:
  * Comparisons between different courses/lessons
  * Multi-part questions needing different search strategies  
  * Information gathering from multiple sources
- **Maximum 2 rounds of tool calls per query**
- Each round can use multiple tools simultaneously
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Course Outline Responses:
When using get_course_outline, always include in your response:
- Course title
- Course link (if available)
- Complete lesson list with lesson numbers and titles
- Present information in a clear, structured format

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course structure questions**: Use get_course_outline tool first, then answer
- **Course content questions**: Use search_course_content tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager, max_rounds)
        
        # Return direct response
        try:
            return response.content[0].text
        except (AttributeError, IndexError):
            # Fallback for edge cases 
            return "Unable to process response"
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager, max_rounds: int = 2):
        """
        Handle execution of tool calls with support for sequential rounds.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds
            
        Returns:
            Final response text after tool execution
        """
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0
        
        while round_count < max_rounds and current_response.stop_reason == "tool_use":
            round_count += 1
            
            # Add AI's tool use response
            messages.append({"role": "assistant", "content": current_response.content})
            
            # Execute tools and add results
            tool_results = self._execute_tools(current_response, tool_manager)
            if not tool_results:  # Tool execution failed
                break
                
            messages.append({"role": "user", "content": tool_results})
            
            # If this was the last round or we have no more tools, make final call without tools
            if round_count >= max_rounds:
                final_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"]
                }
                final_response = self.client.messages.create(**final_params)
                return final_response.content[0].text
            
            # Continue with tools available for next round
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": self._build_round_system_prompt(base_params["system"], round_count, max_rounds),
                "tools": base_params.get("tools"),
                "tool_choice": {"type": "auto"}
            }
            
            current_response = self.client.messages.create(**next_params)
            
            # If no tool use in response, we're done
            if current_response.stop_reason != "tool_use":
                break
        
        # Return the final response (current_response already contains the result)
        try:
            return current_response.content[0].text
        except (AttributeError, IndexError):
            # Fallback for edge cases (e.g., no tool manager, malformed response)
            return "Unable to process response"
    
    def _execute_tools(self, response, tool_manager) -> Optional[List]:
        """
        Execute all tool calls in a response and return formatted results.
        
        Args:
            response: Anthropic response containing tool use blocks
            tool_manager: Manager to execute tools
            
        Returns:
            List of tool results or None if execution failed
        """
        tool_results = []
        try:
            for content_block in response.content:
                if content_block.type == "tool_use":
                    result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    })
            return tool_results if tool_results else None
        except Exception:
            return None  # Tool execution failed
    
    def _build_round_system_prompt(self, base_system: str, current_round: int, max_rounds: int) -> str:
        """
        Build system prompt with round-specific context.
        
        Args:
            base_system: Base system prompt
            current_round: Current round number (1-based)
            max_rounds: Maximum number of rounds
            
        Returns:
            Enhanced system prompt with round context
        """
        if current_round < max_rounds:
            round_context = f"\n\nRound {current_round} of {max_rounds}: You can make additional tool calls if needed for complex queries requiring multiple searches or comparisons."
        else:
            round_context = f"\n\nFinal round ({current_round} of {max_rounds}): This is your last opportunity to use tools before providing your final answer."
        
        return base_system + round_context