"""AI Task integration for OpenAI Compatible APIs (v2 API)."""

from __future__ import annotations

import base64
import json
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    ImageGenerationCall,
    ResponseInputParam,
)

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
    """OpenAI Compatible AI Task entity (v2 API)."""

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
            # | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        
        # Get system prompt from chat log
        system_prompt = "You are a helpful assistant."
        if chat_log.content and chat_log.content[0].role == "system":
            system_prompt = chat_log.content[0].content
        
        messages: ResponseInputParam = [
            EasyInputMessageParam(
                type="message", role="system", content=system_prompt
            ),
            EasyInputMessageParam(
                type="message", role="user", content=task.instructions
            ),
        ]
        
        # Note: attachments are not handled in this basic example
        # You would need to add logic similar to OpenAI's official integration
        # to handle task.attachments and convert them to v2 API format.

        model_args: dict[str, Any] = {
            "model": self.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "input": messages,
            "max_output_tokens": self.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            "top_p": self.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "temperature": self.options.get(
                CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
            ),
            "stream": False, # ai_task does not support streaming
            "store": False,
            "user": chat_log.conversation_id,
        }
        
        # Remove params that might not be supported
        base_url = str(getattr(self.client, "base_url", ""))
        is_mistral = "mistral.ai" in base_url.lower()
        if is_mistral:
            model_args.pop("user", None)

        # Handle structured data if requested
        if task.structure:
            # v2 API uses text.format.type = "json_schema"
            # This is complex to add without knowing if the compatible API supports it.
            # We'll use the simpler "instruct the model" approach.
            model_args["input"].append(
                EasyInputMessageParam(
                    type="message",
                    role="user",
                    content=f"Please provide the output in a JSON format matching this schema: {json.dumps(task.structure.schema)}",
                )
            )

        try:
            response = await self.client.responses.create(**model_args)
            
            response_text = response.output_text or ""

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
            except JSONDecodeError as err:
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
        
        # This implementation is based on the official OpenAI integration
        # It assumes the v2 API structure for image generation
        
        messages: ResponseInputParam = [
            EasyInputMessageParam(
                type="message", role="user", content=task.instructions
            ),
        ]
        
        model_args: dict[str, Any] = {
            "model": self.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "input": messages,
            "tool_choice": {"type": "image_generation"},
            "stream": False,
            "store": True, # Store=True is needed to get image results
            "user": chat_log.conversation_id,
        }
        
        # Remove params that might not be supported
        base_url = str(getattr(self.client, "base_url", ""))
        is_mistral = "mistral.ai" in base_url.lower()
        if is_mistral:
            model_args.pop("user", None)

        try:
            response = await self.client.responses.create(**model_args)
            
            image_call: ImageGenerationCall | None = None
            if response.output:
                for item in response.output:
                    if isinstance(item, ImageGenerationCall):
                        image_call = item
                        break
            
            if image_call is None or image_call.result is None:
                raise HomeAssistantError("No image returned from API")

            image_data = base64.b64decode(image_call.result)
            image_call.result = None # Clear data to save memory

            if hasattr(image_call, "output_format") and (
                output_format := image_call.output_format
            ):
                mime_type = f"image/{output_format}"
            else:
                mime_type = "image/png"

            return ai_task.GenImageTaskResult(
                image_data=image_data,
                conversation_id=chat_log.conversation_id,
                mime_type=mime_type,
                model=image_call.model,
                revised_prompt=image_call.revised_prompt,
            )

        except openai.RateLimitError as err:
            LOGGER.error("Rate limited by API: %s", err)
            raise HomeAssistantError("Rate limited or insufficient funds") from err
        except openai.OpenAIError as err:
            LOGGER.error("Error generating image: %s", err)
            raise HomeAssistantError(f"Error generating image: {err}") from err
        except Exception as err:
            LOGGER.error("Unexpected error generating image: %s", err)
            raise HomeAssistantError(f"An unexpected error occurred: {err}") from err

