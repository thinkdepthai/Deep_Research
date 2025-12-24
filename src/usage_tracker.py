"""
Token and API Usage Tracking Module

Tracks token usage for OpenAI/OpenRouter and API calls for Tavily separately.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UsageStats:
    """Container for usage statistics."""
    # OpenAI/OpenRouter token usage
    openai_prompt_tokens: int = 0
    openai_completion_tokens: int = 0
    openai_total_tokens: int = 0

    # Tavily API usage
    tavily_api_calls: int = 0

    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add_openai_usage(self, prompt_tokens: int, completion_tokens: int):
        """Add OpenAI token usage."""
        self.openai_prompt_tokens += prompt_tokens
        self.openai_completion_tokens += completion_tokens
        self.openai_total_tokens += (prompt_tokens + completion_tokens)

    def add_tavily_call(self):
        """Increment Tavily API call count."""
        self.tavily_api_calls += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "openai": {
                "prompt_tokens": self.openai_prompt_tokens,
                "completion_tokens": self.openai_completion_tokens,
                "total_tokens": self.openai_total_tokens
            },
            "tavily": {
                "api_calls": self.tavily_api_calls
            },
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }

    def get_summary(self) -> str:
        """Get a formatted summary string."""
        lines = [
            "=" * 60,
            "API Usage Summary",
            "=" * 60,
            "",
            "OpenAI/OpenRouter:",
            f"  Input tokens:  {self.openai_prompt_tokens:,}",
            f"  Output tokens: {self.openai_completion_tokens:,}",
            f"  Total tokens:  {self.openai_total_tokens:,}",
            "",
            "Tavily:",
            f"  API calls:     {self.tavily_api_calls:,}",
            ""
        ]

        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            lines.append(f"Duration: {duration}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class UsageTracker:
    """Global usage tracker instance."""

    def __init__(self):
        self.stats = UsageStats()
        self.stats.start_time = datetime.now()

    def track_openai_response(self, response: Any):
        """Extract and track token usage from OpenAI/LangChain response."""
        try:
            # Method 1: Try response_metadata (LangChain ChatModel format)
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if metadata:
                    usage = metadata.get('token_usage', {})
                    if usage:
                        prompt_tokens = usage.get('prompt_tokens', 0) or usage.get('prompt', 0)
                        completion_tokens = usage.get('completion_tokens', 0) or usage.get('completion', 0)
                        if prompt_tokens or completion_tokens:
                            self.stats.add_openai_usage(prompt_tokens, completion_tokens)
                            return

            # Method 2: Try usage attribute (direct OpenAI format)
            if hasattr(response, 'usage'):
                usage = response.usage
                if usage:
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'prompt', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'completion', 0)
                    if prompt_tokens or completion_tokens:
                        self.stats.add_openai_usage(prompt_tokens, completion_tokens)
                        return

            # Method 3: Try to get from response_metadata if it's a dict
            if isinstance(response, dict):
                metadata = response.get('response_metadata', {})
                usage = metadata.get('token_usage', {})
                if usage:
                    prompt_tokens = usage.get('prompt_tokens', 0) or usage.get('prompt', 0)
                    completion_tokens = usage.get('completion_tokens', 0) or usage.get('completion', 0)
                    if prompt_tokens or completion_tokens:
                        self.stats.add_openai_usage(prompt_tokens, completion_tokens)
                        return

            # Method 4: For AIMessage, check if it has usage in response_metadata
            from langchain_core.messages import AIMessage
            if isinstance(response, AIMessage):
                if hasattr(response, 'response_metadata'):
                    metadata = response.response_metadata
                    if metadata:
                        usage = metadata.get('token_usage', {})
                        if usage:
                            prompt_tokens = usage.get('prompt_tokens', 0) or usage.get('prompt', 0)
                            completion_tokens = usage.get('completion_tokens', 0) or usage.get('completion', 0)
                            if prompt_tokens or completion_tokens:
                                self.stats.add_openai_usage(prompt_tokens, completion_tokens)
                                return

        except Exception:
            # Silently fail - don't break the main flow
            # Token tracking is optional and shouldn't interrupt research
            pass

    def track_tavily_call(self):
        """Track a Tavily API call."""
        self.stats.add_tavily_call()

    def finalize(self):
        """Mark tracking as complete."""
        self.stats.end_time = datetime.now()

    def get_stats(self) -> UsageStats:
        """Get current usage statistics."""
        return self.stats

    def reset(self):
        """Reset all statistics."""
        self.stats = UsageStats()
        self.stats.start_time = datetime.now()


# Global tracker instance
_global_tracker = UsageTracker()


def get_tracker() -> UsageTracker:
    """Get the global usage tracker instance."""
    return _global_tracker


def reset_tracker():
    """Reset the global usage tracker."""
    _global_tracker.reset()

