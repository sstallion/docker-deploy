# qwen3_xml_safe.py — vLLM 0.22.1 tool-parser plugin
#
# Fixes, for the Qwen3 XML tool format on v0.22.1:
#   - `Tool "None" not found`  (null function name on the first streamed delta)
#   - malformed / incomplete tool-call argument JSON
# Both are artifacts of the stock streaming JSON assembler reacting to chunk
# boundaries. This wrapper does NOT assemble JSON incrementally: it waits until
# a <tool_call> block is fully closed, then parses that complete prefix with the
# stock one-shot parser (`extract_tool_calls`, which feeds the whole string to
# `parse_single_streaming_chunks` in a single chunk — the same path used in
# non-streaming mode) and emits each call as one complete delta.
#
# Load:
#   --tool-parser-plugin /path/qwen3_xml_safe.py --tool-call-parser qwen3_xml_safe
#
# Notes:
#   * Per-tool-call argument streaming is intentionally given up; the name +
#     full arguments are emitted together once the call closes. Clients key by
#     `index`, so this is OpenAI-compatible.
#   * Plain content before the first <tool_call> is streamed normally.
#   * Trailing prose AFTER the final </tool_call> is not streamed (rare for Qwen
#     tool turns). Extend `_content_cut` handling if you need it.
#   * A truncated / unclosed final tool call is simply not emitted, so the client
#     never receives a malformed call to dispatch (strictly better than before).
#   * Correctness no longer depends on speculative decoding; keep MTP only for
#     latency if you want it.

from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.qwen3xml_tool_parser import Qwen3XMLToolParser


@ToolParserManager.register_module("qwen3_xml_safe")
class Qwen3XMLSafeToolParser(Qwen3XMLToolParser):
    """Parse each <tool_call> block only once complete; never reassemble JSON
    across chunk boundaries."""

    def _content_cut(self, text: str, start: str) -> int:
        """End index of streamable plain content (everything before the first
        <tool_call>), holding back a possible partial start-token prefix so we
        never leak a partial '<tool_call>' as content."""
        cut = text.find(start)
        if cut != -1:
            return cut
        for k in range(len(start) - 1, 0, -1):
            if text.endswith(start[:k]):
                return len(text) - k
        return len(text)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> "DeltaMessage | None":
        start = self.parser.tool_call_start_token  # "<tool_call>"
        end = self.parser.tool_call_end_token  # "</tool_call>"

        # Reset per-response state at the start of a new stream.
        if not previous_text:
            self._emitted = 0
            self._content_sent = 0

        # 1) Stream plain content up to the first tool call.
        content_end = self._content_cut(current_text, start)
        if self._content_sent < content_end:
            chunk = current_text[self._content_sent : content_end]
            self._content_sent = content_end
            if chunk:
                return DeltaMessage(content=chunk)

        if start not in current_text:
            return None

        # 2) Emit any tool calls whose </tool_call> has closed. Parse the
        #    complete closed prefix in one shot -> correct name + valid JSON.
        closed = current_text.count(end)
        if closed > self._emitted:
            complete = current_text[: current_text.rfind(end) + len(end)]
            tool_calls = self.extract_tool_calls(complete, request).tool_calls or []
            upto = min(closed, len(tool_calls))
            deltas = [
                DeltaToolCall(
                    index=i,
                    id=tool_calls[i].id or make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=tool_calls[i].function.name,
                        arguments=tool_calls[i].function.arguments,
                    ),
                )
                for i in range(self._emitted, upto)
            ]
            self._emitted = upto
            if deltas:
                return DeltaMessage(tool_calls=deltas)

        return None
