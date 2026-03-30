"""Tests for Responses API streaming event models and SSE patching."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from pydantic import TypeAdapter, ValidationError

from gateway.models.responses import (
    ContentPart,
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    CreateResponseResponse,
    FuncCallArgsDeltaEvent,
    FuncCallArgsDoneEvent,
    OutputItem,
    OutputItemAddedEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    RefusalDoneEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseStreamEvent,
)
from gateway.providers.llamacpp import _parse_sse_event

_adapter = TypeAdapter(ResponseStreamEvent)


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------


class TestCreateResponseResponse:
    """CreateResponseResponse fills missing OpenAI-spec fields."""

    def test_minimal_input_fills_defaults(self):
        import time

        before = int(time.time())
        data = {"id": "resp-1", "model": "m"}
        resp = CreateResponseResponse.model_validate(data)
        after = int(time.time())
        d = resp.model_dump(mode="json")

        assert d["object"] == "response"
        assert d["status"] == "completed"
        assert before <= d["created_at"] <= after
        assert d["output"] == []
        assert d["metadata"] == {}
        assert d["tools"] == []
        assert d["tool_choice"] == "auto"
        assert d["temperature"] == 1.0
        assert d["top_p"] == 1.0
        assert d["truncation"] == "disabled"
        assert d["parallel_tool_calls"] is True
        assert d["store"] is True
        assert d["text"] == {"format": {"type": "text"}}
        assert d["instructions"] is None
        assert d["max_output_tokens"] is None
        assert d["previous_response_id"] is None
        assert d["reasoning"] is None
        assert d["error"] is None
        assert d["incomplete_details"] is None
        assert d["usage"] is None
        assert d["user"] is None

    def test_explicit_values_override_defaults(self):
        data = {
            "id": "resp-2",
            "model": "m",
            "temperature": 0.5,
            "metadata": {"key": "val"},
            "tools": [{"type": "function", "name": "f"}],
        }
        resp = CreateResponseResponse.model_validate(data)
        d = resp.model_dump(mode="json")
        assert d["temperature"] == 0.5
        assert d["metadata"] == {"key": "val"}
        assert len(d["tools"]) == 1

    def test_extra_fields_preserved(self):
        data = {"id": "resp-3", "model": "m", "custom_field": 42}
        resp = CreateResponseResponse.model_validate(data)
        d = resp.model_dump(mode="json")
        assert d["custom_field"] == 42


class TestOutputItem:
    def test_fills_status_default(self):
        item = OutputItem.model_validate({"type": "message"})
        d = item.model_dump(mode="json")
        assert d["status"] == "in_progress"
        assert d["id"] == ""

    def test_preserves_extra_fields(self):
        item = OutputItem.model_validate({
            "type": "message",
            "id": "msg-1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
        })
        d = item.model_dump(mode="json")
        assert d["role"] == "assistant"
        assert d["content"] == [{"type": "output_text", "text": "hi"}]
        assert d["status"] == "in_progress"


class TestContentPart:
    def test_fills_text_and_annotations(self):
        part = ContentPart.model_validate({"type": "output_text"})
        d = part.model_dump(mode="json")
        assert d["text"] == ""
        assert d["annotations"] == []

    def test_explicit_text_preserved(self):
        part = ContentPart.model_validate({
            "type": "output_text",
            "text": "hello",
            "annotations": [{"a": 1}],
        })
        d = part.model_dump(mode="json")
        assert d["text"] == "hello"
        assert d["annotations"] == [{"a": 1}]


# ---------------------------------------------------------------------------
# Discriminated union tests
# ---------------------------------------------------------------------------


class TestResponseStreamEvent:
    """The discriminated union dispatches on ``type`` and fills defaults."""

    def test_response_created(self):
        import time

        before = int(time.time())
        data = {
            "type": "response.created",
            "response": {"id": "resp-1", "model": "m", "status": "in_progress"},
        }
        event = _adapter.validate_python(data)
        after = int(time.time())
        assert isinstance(event, ResponseCreatedEvent)
        d = _adapter.dump_python(event, mode="json")

        assert d["type"] == "response.created"
        assert d["sequence_number"] == 0
        r = d["response"]
        assert r["tools"] == []
        assert r["metadata"] == {}
        assert r["text"] == {"format": {"type": "text"}}
        assert before <= r["created_at"] <= after

    def test_response_completed(self):
        data = {
            "type": "response.completed",
            "sequence_number": 42,
            "response": {"id": "r", "model": "m"},
        }
        event = _adapter.validate_python(data)
        assert isinstance(event, ResponseCompletedEvent)
        assert event.sequence_number == 42

    def test_output_item_added(self):
        data = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg-1", "role": "assistant"},
        }
        event = _adapter.validate_python(data)
        assert isinstance(event, OutputItemAddedEvent)
        d = _adapter.dump_python(event, mode="json")
        assert d["item"]["status"] == "in_progress"
        assert d["item"]["role"] == "assistant"

    def test_content_part_added(self):
        data = {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text"},
        }
        event = _adapter.validate_python(data)
        assert isinstance(event, ContentPartAddedEvent)
        d = _adapter.dump_python(event, mode="json")
        assert d["part"]["text"] == ""
        assert d["part"]["annotations"] == []

    def test_content_part_done(self):
        data = {
            "type": "response.content_part.done",
            "part": {"type": "output_text", "text": "hello"},
        }
        event = _adapter.validate_python(data)
        assert isinstance(event, ContentPartDoneEvent)

    def test_text_delta(self):
        event = _adapter.validate_python({
            "type": "response.output_text.delta",
        })
        assert isinstance(event, OutputTextDeltaEvent)
        d = _adapter.dump_python(event, mode="json")
        assert d["delta"] == ""
        assert d["output_index"] == 0
        assert d["content_index"] == 0

    def test_text_done(self):
        event = _adapter.validate_python({
            "type": "response.output_text.done",
        })
        assert isinstance(event, OutputTextDoneEvent)
        d = _adapter.dump_python(event, mode="json")
        assert d["text"] == ""

    def test_refusal_done(self):
        event = _adapter.validate_python({
            "type": "response.refusal.done",
        })
        assert isinstance(event, RefusalDoneEvent)
        d = _adapter.dump_python(event, mode="json")
        assert d["refusal"] == ""

    def test_func_call_delta(self):
        event = _adapter.validate_python({
            "type": "response.function_call_arguments.delta",
        })
        assert isinstance(event, FuncCallArgsDeltaEvent)
        d = _adapter.dump_python(event, mode="json")
        assert d["item_id"] == ""
        assert d["delta"] == ""

    def test_func_call_done(self):
        event = _adapter.validate_python({
            "type": "response.function_call_arguments.done",
        })
        assert isinstance(event, FuncCallArgsDoneEvent)
        d = _adapter.dump_python(event, mode="json")
        assert d["item_id"] == ""
        assert d["arguments"] == ""

    def test_unknown_type_raises(self):
        with pytest.raises(ValidationError):
            _adapter.validate_python({"type": "response.some_future_event"})

    EXPECTED_TYPES: ClassVar[list[str]] = [
        "response.created",
        "response.in_progress",
        "response.completed",
        "response.failed",
        "response.incomplete",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.refusal.delta",
        "response.refusal.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
    ]

    @pytest.mark.parametrize("event_type", EXPECTED_TYPES)
    def test_all_event_types_validate(self, event_type: str):
        """Every expected event type is accepted by the union."""
        # Build a minimal valid payload for each type.
        data: dict = {"type": event_type}
        if event_type.startswith("response.") and event_type.count(".") == 1:
            # Response lifecycle events need a response object.
            data["response"] = {"id": "r", "model": "m"}
        if "output_item" in event_type:
            data["item"] = {"type": "message"}
        if "content_part" in event_type:
            data["part"] = {"type": "output_text"}
        event = _adapter.validate_python(data)
        assert event.type == event_type


# ---------------------------------------------------------------------------
# SSE event parsing tests
# ---------------------------------------------------------------------------


class TestParseSseEvent:
    """_parse_sse_event validates SSE blocks into typed event models."""

    def test_response_created_fills_defaults(self):
        """A response.created event with a minimal response gets defaults."""
        import time

        before = int(time.time())
        payload = {
            "type": "response.created",
            "response": {"id": "resp-1", "model": "m", "status": "in_progress"},
        }
        block = b"data: " + json.dumps(payload).encode()
        event, seq = _parse_sse_event(block, 0)
        after = int(time.time())

        assert event is not None
        assert isinstance(event, ResponseCreatedEvent)
        assert event.sequence_number == 0
        assert event.response.tools == []
        assert event.response.metadata == {}
        assert event.response.temperature == 1.0
        assert before <= event.response.created_at <= after
        assert seq == 1

    def test_sequence_number_assigned_when_missing(self):
        """sequence_number is filled from the counter when absent."""
        payload = {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hi",
        }
        block = b"data: " + json.dumps(payload).encode()
        event, seq = _parse_sse_event(block, 5)
        assert event is not None
        assert event.sequence_number == 5
        assert seq == 6

    def test_sequence_number_preserved_when_present(self):
        """sequence_number from upstream is kept; counter still advances."""
        payload = {
            "type": "response.output_text.delta",
            "sequence_number": 99,
            "output_index": 0,
            "content_index": 0,
            "delta": "Hi",
        }
        block = b"data: " + json.dumps(payload).encode()
        event, seq = _parse_sse_event(block, 5)
        assert event is not None
        assert event.sequence_number == 99
        assert seq == 6

    def test_done_returns_none(self):
        block = b"data: [DONE]"
        event, seq = _parse_sse_event(block, 0)
        assert event is None
        assert seq == 0

    def test_unknown_event_type_returns_none(self):
        """Unknown event types return None (cannot validate)."""
        payload = {"type": "response.some_future_event", "foo": "bar"}
        block = b"data: " + json.dumps(payload).encode()
        event, seq = _parse_sse_event(block, 0)
        assert event is None
        assert seq == 0

    def test_non_json_returns_none(self):
        block = b"data: not-json"
        event, seq = _parse_sse_event(block, 0)
        assert event is None
        assert seq == 0

    def test_no_data_line_returns_none(self):
        block = b": keep-alive"
        event, seq = _parse_sse_event(block, 0)
        assert event is None
        assert seq == 0

    def test_multiple_data_lines_returns_none(self):
        block = b"data: first\ndata: second"
        event, seq = _parse_sse_event(block, 0)
        assert event is None
        assert seq == 0

    def test_event_line_ignored_data_parsed(self):
        """Non-data SSE lines (like event:) are ignored; data is parsed."""
        payload = {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "x",
        }
        block = b"event: response.output_text.delta\ndata: " + json.dumps(payload).encode()
        event, _seq = _parse_sse_event(block, 0)
        assert event is not None
        assert isinstance(event, OutputTextDeltaEvent)
        assert event.sequence_number == 0

    def test_output_item_added_fills_status(self):
        payload = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg-1"},
        }
        block = b"data: " + json.dumps(payload).encode()
        event, _ = _parse_sse_event(block, 0)
        assert event is not None
        assert isinstance(event, OutputItemAddedEvent)
        assert event.item.status == "in_progress"

    def test_content_part_added_fills_annotations(self):
        payload = {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text"},
        }
        block = b"data: " + json.dumps(payload).encode()
        event, _ = _parse_sse_event(block, 0)
        assert event is not None
        assert isinstance(event, ContentPartAddedEvent)
        assert event.part.text == ""
        assert event.part.annotations == []

    def test_data_without_leading_space(self):
        """Handles ``data:{...}`` (no space after colon)."""
        payload = {"type": "response.output_text.delta", "delta": "y"}
        block = b"data:" + json.dumps(payload).encode()
        event, seq = _parse_sse_event(block, 0)
        assert event is not None
        assert isinstance(event, OutputTextDeltaEvent)
        assert event.delta == "y"
        assert seq == 1

    def test_no_type_field_returns_none(self):
        """JSON without a ``type`` key returns None."""
        payload = {"foo": "bar"}
        block = b"data: " + json.dumps(payload).encode()
        event, seq = _parse_sse_event(block, 0)
        assert event is None
        assert seq == 0


# ---------------------------------------------------------------------------
# Full stream parsing (async generator)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
class TestParseResponseStream:
    """_parse_response_stream yields typed events from an SSE byte stream."""

    async def test_parses_multi_event_stream(self):
        from gateway.providers.llamacpp import _parse_response_stream

        events = [
            {"type": "response.created", "response": {"id": "r", "model": "m"}},
            {"type": "response.output_text.delta", "delta": "Hi"},
            {"type": "response.output_text.done", "text": "Hi"},
            {"type": "response.completed", "response": {"id": "r", "model": "m"}},
        ]
        raw_bytes = b""
        for evt in events:
            raw_bytes += b"data: " + json.dumps(evt).encode() + b"\n\n"
        raw_bytes += b"data: [DONE]\n\n"

        class FakeResp:
            async def aiter_bytes(self):
                yield raw_bytes

            async def aclose(self):
                pass

        collected = []
        async for event in _parse_response_stream(FakeResp()):  # type: ignore[arg-type]
            collected.append(event)

        assert len(collected) == 4
        assert isinstance(collected[0], ResponseCreatedEvent)
        assert isinstance(collected[1], OutputTextDeltaEvent)
        assert isinstance(collected[2], OutputTextDoneEvent)
        assert isinstance(collected[3], ResponseCompletedEvent)

        # Defaults filled
        assert collected[0].response.tools == []
        assert collected[0].response.metadata == {}

        # Sequence numbers assigned
        assert collected[0].sequence_number == 0
        assert collected[1].sequence_number == 1
        assert collected[2].sequence_number == 2
        assert collected[3].sequence_number == 3

    async def test_unknown_events_skipped(self):
        """Events with unrecognized types are silently dropped."""
        from gateway.providers.llamacpp import _parse_response_stream

        raw = (
            b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
            b'data: {"type":"response.unknown_future_event","x":1}\n\n'
            b'data: {"type":"response.output_text.done","text":"Hi"}\n\n'
        )

        class FakeResp:
            async def aiter_bytes(self):
                yield raw

            async def aclose(self):
                pass

        collected = []
        async for event in _parse_response_stream(FakeResp()):  # type: ignore[arg-type]
            collected.append(event)

        assert len(collected) == 2
        assert isinstance(collected[0], OutputTextDeltaEvent)
        assert isinstance(collected[1], OutputTextDoneEvent)

    async def test_handles_chunked_delivery(self):
        """Events split across multiple chunks are reassembled correctly."""
        from gateway.providers.llamacpp import _parse_response_stream

        event = b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
        chunk1 = event[:20]
        chunk2 = event[20:]

        class FakeResp:
            async def aiter_bytes(self):
                yield chunk1
                yield chunk2

            async def aclose(self):
                pass

        collected = []
        async for event in _parse_response_stream(FakeResp()):  # type: ignore[arg-type]
            collected.append(event)

        assert len(collected) == 1
        assert isinstance(collected[0], OutputTextDeltaEvent)
        assert collected[0].delta == "Hi"
        assert collected[0].sequence_number == 0

    async def test_aclose_called_on_completion(self):
        """The upstream response is always closed."""
        from gateway.providers.llamacpp import _parse_response_stream

        closed = False

        class FakeResp:
            async def aiter_bytes(self):
                yield b"data: [DONE]\n\n"

            async def aclose(self):
                nonlocal closed
                closed = True

        async for _ in _parse_response_stream(FakeResp()):  # type: ignore[arg-type]
            pass

        assert closed

    async def test_aclose_called_on_error(self):
        """The upstream response is closed even if iteration fails."""
        from gateway.providers.llamacpp import _parse_response_stream

        closed = False

        class FakeResp:
            async def aiter_bytes(self):
                yield b'data: {"type":"response.output_text.delta","delta":"x"}\n\n'
                raise ConnectionError("boom")

            async def aclose(self):
                nonlocal closed
                closed = True

        with pytest.raises(ConnectionError):
            async for _ in _parse_response_stream(FakeResp()):  # type: ignore[arg-type]
                pass

        assert closed
