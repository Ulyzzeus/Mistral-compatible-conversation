"""Constants for the Mistral Compatible Conversation integration."""

import logging
from types import MappingProxyType

DOMAIN = "mistral_compatible_conversation"
LOGGER = logging.getLogger(__package__)

CONF_NAME = "name"
DEFAULT_AGENT_NAME = "MISTRAL Compatible Agent"

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "mistral-small-latest"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 600
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
CONF_BASE_URL = "base_url"
RECOMMENDED_BASE_URL = "https://api.mistral.ai/v1/"
CONF_NO_THINK = "no_think"
CONF_STRIP_THINK_TAGS = "strip_think_tags"

DEFAULT_AI_TASK_NAME = "Mistral AI Task"
# Use a plain dict, not a MappingProxyType, to avoid JSON serialization errors
RECOMMENDED_AI_TASK_OPTIONS = {
    "recommended": True,
}

