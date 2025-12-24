import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

# Load .env file from project root (parent of src directory)
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Also try current directory as fallback
    load_dotenv()

def get_light_model():
    """
    Get a light model for summarization and other tasks.
    """
    return get_model(model_name=os.getenv("OPENROUTER_LIGHT_MODEL", "openai/gpt-5-nano"))

def get_model(model_name: str = None, max_tokens: int = None, **kwargs):
    """
    Initialize a chat model with OpenRouter or OpenAI.

    Args:
        model_name: Model identifier
            - For OpenRouter: e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet"
            - For OpenAI: e.g., "gpt-5", "gpt-4o"
        max_tokens: Maximum tokens for the model
        **kwargs: Additional model parameters

    Returns:
        Initialized chat model
    """
    # Check if OpenRouter is configured
    use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if use_openrouter and openrouter_key:
        # Use OpenRouter via ChatOpenAI with OpenRouter's endpoint
        if model_name is None:
            model_name = os.getenv("OPENROUTER_DEFAULT_MODEL", "openai/gpt-4o")

        return ChatOpenAI(
            model=model_name,
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=max_tokens,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", ""),  # Optional: for analytics
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "Deep Research"),  # Optional: for analytics
            },
            **kwargs
        )
    else:
        # Use standard OpenAI via init_chat_model
        if model_name is None:
            model_name = os.getenv("OPENAI_MODEL", "gpt-5")

        return init_chat_model(
            model=f"openai:{model_name}",
            max_tokens=max_tokens,
            **kwargs
        )