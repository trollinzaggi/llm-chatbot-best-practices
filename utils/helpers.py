"""
Common utilities for all LLM libraries
"""
import time
import json
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
import tiktoken


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(min(delay, max_delay))
                        delay *= exponential_base
                    else:
                        raise last_exception
            
            return None
        return wrapper
    return decorator


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer
    
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def truncate_text_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    Truncate text to fit within token limit
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenizer
    
    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def format_conversation_history(
    messages: List[Dict[str, str]],
    max_messages: Optional[int] = None,
    max_tokens: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Format and optionally truncate conversation history
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_messages: Maximum number of messages to keep
        max_tokens: Maximum total tokens for conversation
    
    Returns:
        Formatted message list
    """
    if max_messages and len(messages) > max_messages:
        # Keep system message if present, then most recent messages
        if messages[0].get("role") == "system":
            messages = [messages[0]] + messages[-(max_messages-1):]
        else:
            messages = messages[-max_messages:]
    
    if max_tokens:
        total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
        while total_tokens > max_tokens and len(messages) > 1:
            # Remove oldest non-system message
            for i in range(len(messages)):
                if messages[i].get("role") != "system":
                    messages.pop(i)
                    break
            total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
    
    return messages


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
    
    Returns:
        Sanitized text
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces/newlines with single ones
    text = " ".join(text.split())
    
    # Truncate if necessary
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Safely parse JSON response from LLM
    
    Args:
        response: String response that should contain JSON
    
    Returns:
        Parsed JSON as dictionary
    """
    # Try to find JSON in the response
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No JSON object found in response")
    
    json_str = response[start_idx:end_idx + 1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common issues
        # Remove trailing commas
        json_str = json_str.replace(',}', '}').replace(',]', ']')
        # Try again
        return json.loads(json_str)


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            time.sleep(self.min_interval - time_since_last_call)
        
        self.last_call_time = time.time()
