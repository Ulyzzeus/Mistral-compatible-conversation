"""AI Task integration for OpenAI Compatible APIs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import openai
from openai._types import NOT_GIVEN

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry

    from . import OpenAICompatibleConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [OpenAICompatibleAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class OpenAICompatibleAITaskEntity(
    ai_task.AITaskEntity,
):
    """OpenAI Compatible AI Task entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OpenAICompatibleConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self.client = entry.runtime_data
        self.options = subentry.data

        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="OpenAI Compatible",
            model=subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        self._attr_supported_features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA
            | ai_task.AITaskEntityFeature.GENERATE_IMAGE
            # Add SUPPORT_ATTACHMENTS if your model/API can handle images
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        
        # This implementation is based on your conversation.py logic
        # It does not support streaming, as ai_task does not support it.
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, # This was the line with the SyntaxError
            {"role": "user", "content": task.instructions}
        ]
        
        # Add system prompt from chat_log if it exists (it's the first message)
        if chat_log.content and chat_log.content[0].role == "system":
            messages[0]["content"] = chat_log.content[0].content
            messages[1]["content"] = task.instructions
        else:
             messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
             messages.append({"role": "user", "content": task.instructions})


        # Note: attachments are not handled in this basic example
        # You would need to add logic similar to Google/OpenAI to handle task.attachments

        model_args: dict[str, Any] = {
            "model": self.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "messages": messages,
            "max_tokens": self.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            "top_p": self.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "temperature": self.options.get(
                CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
            ),
            "stream": False, # ai_task does not support streaming
        }

        # Handle structured data if requested
        if task.structure:
            model_args["response_format"] = {"type": "json_object"}
            # You might need to adjust the prompt to instruct the model to use the JSON format
            messages.append(
                {
                    "role": "user",
                    "content": f"Please provide the output in a JSON format matching this schema: {json.dumps(task.structure.schema)}",
                }
            )

        try:
            response = await self.client.chat.completions.create(**model_args)
            
            response_text = response.choices[0].message.content or ""

        except openai.RateLimitError as err:
            LOGGER.error("Rate limited by API: %s", err)
            raise HomeAssistantError("Rate limited or insufficient funds") from err
        except openai.OpenAIError as err:
            LOGGER.error("Error talking to API: %s", err)
            error_body = getattr(err, "body", None)
            if error_body:
                LOGGER.error("API Error Body: %s", error_body)
            raise HomeAssistantError(f"Error talking to API: {err}") from err
        except Exception as err:
            LOGGER.error("Unexpected error: %s", err)
            raise HomeAssistantError(f"An unexpected error occurred: {err}") from err

        data_response: Any
        if task.structure:
            try:
                data_response = json_loads(response_text)
            except json.JSONDecodeError as err:
                LOGGER.error(
                    "Failed to parse JSON response: %s. Response: %s",
                    err,
                    response_text,
                )
                raise HomeAssistantError(
                    f"Model returned invalid JSON: {err}"
                ) from err
        else:
            data_response = response_text

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data_response,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        
        # This uses the same logic as your `generate_image` service
        try:
            response = await self.client.images.generate(
                model="dall-e-3", # You may want to make this configurable
                prompt=task.instructions,
                response_format="b64_json", # Request base64 data
                n=1,
            )
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        if (
            not response.data
            or not response.data[0].b64_json
        ):
            raise HomeAssistantError("No image data returned from API")

        image_data = await self.hass.async_add_executor_job(
            base64.b64decode, response.data[0].b64_json
        )

        return ai_task.GenImageTaskResult(
            image_data=image_data,
            conversation_id=chat_log.conversation_id,
            mime_type="image/png", # DALL-E returns PNG
            revised_prompt=response.data[0].revised_prompt,
        )

