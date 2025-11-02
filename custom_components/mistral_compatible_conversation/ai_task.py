"""AI Task integration for OpenAI Compatible APIs."""

from __future__ import annotations

import base64
from json import JSONDecodeError
import logging
from typing import TYPE_CHECKING

from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
import openai

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .const import (
    CONF_CHAT_MODEL,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
)

# Import the base class from the ai_task integration
from homeassistant.components.ai_task.entity import AITaskEntity
from homeassistant.components.ai_task.const import AITaskEntityFeature

if TYPE_CHECKING:
    from . import OpenAICompatibleConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAICompatibleConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up your AI Task entity from a config entry."""

    # This logic assumes you use config subentries like the examples.
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type == "ai_task_data":
            client = config_entry.runtime_data
            async_add_entities(
                [OpenAICompatibleAITaskEntity(client, subentry)],
                config_subentry_id=subentry.subentry_id,
            )


class OpenAICompatibleAITaskEntity(AITaskEntity):  # Inherit from AITaskEntity
    """Your integration's AI Task entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, client: AsyncClient, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self._client = client
        self._options = subentry.data
        self._attr_unique_id = subentry.subentry_id
        self._attr_name = subentry.title

        # Define what your entity supports
        self._attr_supported_features = (
            AITaskEntityFeature.GENERATE_DATA
            # | AITaskEntityFeature.SUPPORT_ATTACHMENTS # Add if your API/model supports it
            | AITaskEntityFeature.GENERATE_IMAGE
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        # 1. Get the last user message from the chat log
        user_message = chat_log.content[-1]

        # 2. Extract instructions
        instructions = user_message.content
        # attachments = user_message.attachments # Uncomment if you add SUPPORT_ATTACHMENTS

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant." MAPPING_KEY},
            {"role": "user", "content": instructions},
        ]

        model_args: dict[str, Any] = {
            "model": self._options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "messages": messages,
        }

        # 3. Call YOUR AI API with the instructions
        if task.structure:
            model_args["response_format"] = ResponseFormat(type="json_object")

        try:
            response = await self._client.chat.completions.create(**model_args)
            api_response_text = response.choices[0].message.content or ""
        except openai.OpenAIError as err:
            LOGGER.error("Error generating data: %s", err)
            raise HomeAssistantError(f"Error generating data: {err}") from err

        # 4. Return the result
        if task.structure:
            try:
                data = json_loads(api_response_text)
            except JSONDecodeError as err:
                LOGGER.error(
                    "Failed to parse JSON response: %s. Response: %s",
                    err,
                    api_response_text,
                )
                raise HomeAssistantError("Failed to parse structured response") from err
        else:
            # Otherwise, just return the raw text
            data = api_response_text

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        if not (self.supported_features & AITaskEntityFeature.GENERATE_IMAGE):
            raise NotImplementedError("Image generation is not supported.")

        user_message = chat_log.content[-1]
        instructions = user_message.content

        # 1. Call your image generation API
        try:
            response = await self._client.images.generate(
                model="dall-e-3",  # Or make this configurable
                prompt=instructions,
                size="1024x1024",  # Or make this configurable
                quality="standard",  # Or make this configurable
                style="vivid",  # Or make this configurable
                response_format="b64_json",  # Request base64 data
                n=1,
            )
            
            if not response.data or not response.data[0].b64_json:
                raise HomeAssistantError("API did not return image data.")

            image_bytes = base64.b64decode(response.data[0].b64_json)
            mime_type = "image/png"
            revised_prompt = response.data[0].revised_prompt

        except openai.OpenAIError as err:
            LOGGER.error("Error generating image: %s", err)
            raise HomeAssistantError(f"Error generating image: {err}") from err
        except Exception as err:
            LOGGER.error("Unexpected error generating image: %s", err)
            raise HomeAssistantError(f"Unexpected error generating image: {err}") from err


        # 2. Return the image data
        return ai_task.GenImageTaskResult(
            image_data=image_bytes,
            conversation_id=chat_log.conversation_id,
            mime_type=mime_type,
            revised_prompt=revised_prompt,
        )
