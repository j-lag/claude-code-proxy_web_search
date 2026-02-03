import json
import logging
from typing import Any, Dict, List, Optional

from venv import logger

from src.core.config import config
from src.core.constants import Constants
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest, ClaudeTool

logger = logging.getLogger(__name__)


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Convert messages
    openai_messages = []

    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            openai_messages.append(
                {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
            )

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg)
            openai_messages.append(openai_message)

            # Check if next message contains tool results
            if i + 1 < len(claude_request.messages):
                next_msg = claude_request.messages[i + 1]
                if (
                    next_msg.role == Constants.ROLE_USER
                    and isinstance(next_msg.content, list)
                    and any(
                        block.type == Constants.CONTENT_TOOL_RESULT
                        for block in next_msg.content
                        if hasattr(block, "type")
                    )
                ):
                    # Process tool results
                    i += 1  # Skip to tool result message
                    tool_results = convert_claude_tool_results(next_msg)
                    openai_messages.extend(tool_results)

        i += 1

    # Build OpenAI request
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        #"max_tokens": min(
        #    max(claude_request.max_tokens, config.min_tokens_limit),
        #    config.max_tokens_limit,
        #),
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }
    logger.debug(
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools
    if claude_request.tools:
        openai_tools = []
        for tool in claude_request.tools:
            if tool.name and tool.name.strip():
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema or {},
                        },
                    }
                )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    return openai_request


def has_web_search_tool(tools: Optional[List[ClaudeTool]]) -> bool:
    """Detect if the Claude request enables the web_search tool."""
    if not tools:
        return False

    for tool in tools:
        tool_name = (tool.name or "").strip().lower()
        tool_type = (tool.type or "").strip().lower()
        if tool_name == "web_search" and (
            not tool_type or tool_type.startswith("web_search_")
        ):
            return True
    return False


def convert_claude_to_responses(
    claude_request: ClaudeMessagesRequest,
    model_manager,
    include_web_search: bool,
) -> Dict[str, Any]:
    """Convert a Claude request into an OpenAI Responses API payload."""

    openai_request = convert_claude_to_openai(claude_request, model_manager)
    openai_messages = openai_request.get("messages", [])

    responses_input = _convert_messages_to_responses_input(openai_messages)
    if not responses_input:
        responses_input = [
            _build_responses_message("user", [{"type": "input_text", "text": ""}])
        ]

    responses_tools: List[Dict[str, Any]] = []
    if include_web_search:
        responses_tools.append({"type": "web_search", "external_web_access": True})

    function_tools = _convert_function_tools(claude_request.tools, include_web_search)
    if function_tools:
        responses_tools.extend(function_tools)

    responses_request: Dict[str, Any] = {
        "model": openai_request["model"],
        "input": responses_input,
        "tool_choice": "auto",
        "temperature": claude_request.temperature,
    }

    if responses_tools:
        responses_request["tools"] = responses_tools

    if claude_request.stop_sequences:
        responses_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        responses_request["top_p"] = claude_request.top_p

    return responses_request


def _convert_function_tools(
    tools: Optional[List[ClaudeTool]], include_web_search: bool
) -> List[Dict[str, Any]]:
    """Convert Claude tools (excluding built-in web search) to OpenAI format."""
    if not tools:
        return []

    converted_tools = []
    for tool in tools:
        tool_name = (tool.name or "").strip().lower()
        if tool_name == "web_search" and include_web_search:
            # Skip the server-side web search entry; we provide the Responses tool instead
            continue
        # If include_web_search is False, fall through and treat it like a user-defined tool

        if tool.name and tool.input_schema is not None:
            converted_tools.append(
                {
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema or {},
                    },
                }
            )
    return converted_tools


def _convert_messages_to_responses_input(
    openai_messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Transform chat-completion-style messages into Responses input format."""
    responses_input: List[Dict[str, Any]] = []
    for message in openai_messages:
        role = message.get("role")
        responses_role = _map_role_to_responses(role)
        if not responses_role:
            continue

        content_items = _convert_message_content_to_responses(message)
        responses_input.append(_build_responses_message(responses_role, content_items))
    return responses_input


def _map_role_to_responses(role: Optional[str]) -> Optional[str]:
    if not role:
        return None
    if role == Constants.ROLE_SYSTEM:
        return "developer"
    if role == Constants.ROLE_TOOL:
        return "developer"
    if role in (Constants.ROLE_USER, Constants.ROLE_ASSISTANT):
        return role
    return None


def _convert_message_content_to_responses(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a single OpenAI chat message into Responses content blocks."""
    content_items: List[Dict[str, Any]] = []
    message_content = message.get("content")

    if isinstance(message_content, list):
        for block in message_content:
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    content_items.append({"type": "input_text", "text": text})
            elif block_type == "image_url":
                content_items.append(
                    {"type": "input_image", "image_url": block.get("image_url", {})}
                )
            else:
                content_items.append(
                    {
                        "type": "input_text",
                        "text": json.dumps(block, ensure_ascii=False),
                    }
                )
    elif isinstance(message_content, str):
        content_items.append({"type": "input_text", "text": message_content})
    elif message_content is not None:
        content_items.append(
            {"type": "input_text", "text": json.dumps(message_content, ensure_ascii=False)}
        )

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
        name = function_data.get("name", "tool_call")
        arguments = function_data.get("arguments", "")
        summary = f"[tool_call:{name}] {arguments}"
        content_items.append({"type": "input_text", "text": summary})

    if not content_items:
        content_items.append({"type": "input_text", "text": ""})

    return content_items


def _build_responses_message(role: str, content: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "message", "role": role, "content": content}


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": block.text})
        elif block.type == Constants.CONTENT_IMAGE:
            # Convert Claude image format to OpenAI format
            if (
                isinstance(block.source, dict)
                and block.source.get("type") == "base64"
                and "media_type" in block.source
                and "data" in block.source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                        },
                    }
                )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": None}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            text_parts.append(block.text)
        elif block.type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    openai_message = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = None

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            if block.type == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(block.content)
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": block.tool_use_id,
                        "content": content,
                    }
                )

    return tool_messages


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item, ensure_ascii=False))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"
