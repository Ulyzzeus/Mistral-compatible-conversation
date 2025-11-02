"""Conversation support for OpenAI Compatible APIs."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import openai
from openai._streaming import AsyncStream
from openai._types import NOT_GIVEN
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseCodeInterpreterToolCall,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from openai.types.responses.response_input_param import FunctionCallOutput
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import OpenAICompatibleConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_NO_THINK,
    CONF_PROMPT,
    CONF_STRIP_THINK_TAGS,
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
    from . import OpenAICompatibleConfigEntry

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAICompatibleConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities from subentries."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue
        agent = OpenAICompatibleConversationEntity(config_entry, subentry)
        async_add_entities([agent], config_subentry_id=subentry.subentry_id)


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> FunctionToolParam:
    """Format tool specification."""
    return FunctionToolParam(
        type="function",
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
        description=tool.description,
        strict=False,
    )


def _convert_content_to_param(
    chat_content: list[conversation.Content],
) -> ResponseInputParam:
    """Convert HA chat messages to the OpenAI v2 native format."""
    messages: ResponseInputParam = []

    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            messages.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=content.tool_call_id,
                    output=json.dumps(content.tool_result),
                )
            )
            continue

        if content.content:
            role: Literal["user", "assistant", "system"] = content.role  # type: ignore[assignment]
            # v2 API uses 'system' role, not 'developer'
            if role == "system":
                messages.append(
                    EasyInputMessageParam(
                        type="message", role="system", content=content.content
                    )
                )
            elif role == "user":
                messages.append(
                    EasyInputMessageParam(
                        type="message", role="user", content=content.content
                    )
                )
            elif role == "assistant":
                messages.append(
                    EasyInputMessageParam(
                        type="message", role="assistant", content=content.content
                    )
                )

        if isinstance(content, conversation.AssistantContent):
            if content.tool_calls:
                for tool_call in content.tool_calls:
                    messages.append(
                        ResponseFunctionToolCall(
                            type="function_call",
                            name=tool_call.tool_name,
                            arguments=json.dumps(tool_call.tool_args),
                            call_id=tool_call.id,
                        )
                    )
    return messages


async def _openai_to_ha_stream(
    chat_log: conversation.ChatLog,
    stream: AsyncStream[ResponseStreamEvent],
    strip_think_tags: bool,
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict,
    None,
]:
    """Transform an OpenAI v2 delta stream into HA format."""
    last_role: Literal["assistant", "tool_result"] | None = None
    
    buffer = ""
    is_in_think_block = False
    start_tag = "<think>"
    end_tag = "</think>"
    
    active_tool_calls_by_index: dict[int, dict[str, Any]] = {}
    current_tool_call: ResponseFunctionToolCall | None = None

    async for event in stream:
        LOGGER.debug("Received event: %s", event)

        if isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseFunctionToolCall):
                if last_role != "assistant":
                    yield {"role": "assistant"}
                    last_role = "assistant"
                current_tool_call = event.item
            elif (
                isinstance(event.item, ResponseOutputMessage)
                or last_role != "assistant"
            ):
                yield {"role": "assistant"}
                last_role = "assistant"

        elif isinstance(event, ResponseTextDeltaEvent):
            content_text = event.delta
            if not content_text:
                continue
            
            if not strip_think_tags:
                yield {"content": content_text}
                continue

            # Start of buffer logic for think tags
            buffer += content_text
            while True:
                if is_in_think_block:
                    end_tag_pos = buffer.find(end_tag)
                    if end_tag_pos != -1:
                        is_in_think_block = False
                        buffer = buffer[end_tag_pos + len(end_tag) :]
                        continue
                    break
                else:
                    start_tag_pos = buffer.find(start_tag)
                    if start_tag_pos != -1:
                        content_to_yield = buffer[:start_tag_pos]
                        if content_to_yield:
                            yield {"content": content_to_yield}
                        is_in_think_block = True
                        buffer = buffer[start_tag_pos + len(start_tag) :]
                        continue

                    if len(buffer) > len(start_tag):
                        safe_yield_len = len(buffer) - len(start_tag)
                        yield {"content": buffer[:safe_yield_len]}
                        buffer = buffer[safe_yield_len:]
                    break
            # End of buffer logic

        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            if current_tool_call:
                current_tool_call.arguments += event.delta

        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            if current_tool_call:
                current_tool_call.status = "completed"
                try:
                    parsed_args = json.loads(current_tool_call.arguments or "{}")
                    yield {
                        "tool_calls": [
                            llm.ToolInput(
                                id=current_tool_call.call_id,
                                tool_name=current_tool_call.name,
                                tool_args=parsed_args,
                            )
                        ]
                    }
                except json.JSONDecodeError:
                    LOGGER.error(
                        "Failed to decode JSON for tool %s: %s",
                        current_tool_call.name,
                        current_tool_call.arguments,
                    )
                current_tool_call = None

        elif isinstance(event, ResponseCompletedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
        elif isinstance(event, (ResponseIncompleteEvent, ResponseFailedEvent)):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
            reason = "unknown reason"
            if isinstance(event, ResponseIncompleteEvent):
                if (
                    event.response.incomplete_details
                    and event.response.incomplete_details.reason
                ):
                    reason = event.response.incomplete_details.reason
                if reason == "max_output_tokens":
                    reason = "max output tokens reached"
                elif reason == "content_filter":
                    reason = "content filter triggered"
                raise HomeAssistantError(f"OpenAI response incomplete: {reason}")
            
            if isinstance(event, ResponseFailedEvent):
                if event.response.error is not None:
                    reason = event.response.error.message
                raise HomeAssistantError(f"OpenAI response failed: {reason}")

        elif isinstance(event, ResponseErrorEvent):
            raise HomeAssistantError(f"OpenAI response error: {event.message}")

    if strip_think_tags and not is_in_think_block and buffer:
        yield {"content": buffer}


class OpenAICompatibleConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Mistral Compatible conversation agent (OpenAI v2 API)."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(
        self,
        entry: OpenAICompatibleConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the agent."""
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
        if self.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a message."""
        
        assert user_input.agent_id
        options = self.options
        client = self.client

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[FunctionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                messages = _convert_content_to_param(chat_log.content)
            except Exception as err:
                LOGGER.error("Error during history regeneration: %s", err)
                raise HomeAssistantError(f"Failed to process history: {err}") from err

            if _iteration == 0:
                no_think_enabled = options.get(CONF_NO_THINK, False)
                if no_think_enabled and messages and messages[-1]["role"] == "user":
                    user_message = messages[-1]
                    current_content = str(user_message.get("content", ""))
                    if not current_content.endswith("/no_think"):
                        user_message["content"] = current_content + "/no_think"

            model_args = ResponseCreateParamsStreaming(
                model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                input=messages,
                tools=tools or NOT_GIVEN,
                tool_choice="auto" if tools else NOT_GGIVEN,
                max_output_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                temperature=options.get(
                    CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                ),
                stream=True,
                store=False, # v2 API uses 'store'
                user=chat_log.conversation_id, # v2 API uses 'user'
            )
            
            # Remove params that might not be supported by all compatible APIs
            base_url = str(getattr(client, "base_url", ""))
            is_mistral = "mistral.ai" in base_url.lower()
            if is_mistral:
                model_args.pop("user", None)
            
            # Add v2 reasoning for compatible models (if any)
            # For now, we'll stick to the core API

            try:
                response_stream = await client.responses.create(**model_args)
                strip_tags = options.get(CONF_STRIP_THINK_TAGS, False)
                
                async for _ in chat_log.async_add_delta_content_stream(
                    user_input.agent_id,
                    _openai_to_ha_stream(chat_log, response_stream, strip_think_tags=strip_tags),
                ):
                    pass

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
                LOGGER.error("Unexpected streaming error: %s", err)
                raise HomeAssistantError(f"An unexpected error occurred: {err}") from err

            if not chat_log.unresponded_tool_results:
                break

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

