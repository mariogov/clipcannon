"""Leapable MCP HTTP client for meeting transcript storage.

Sends JSON-RPC tool calls to the Leapable MCP endpoint (default port 4100).
Performs the MCP initialize handshake on first call, then uses the
server-assigned session ID for all subsequent calls.

Leapable's MCP endpoint returns SSE (Server-Sent Events) framed responses:
    event: message
    data: {"jsonrpc": "2.0", "result": {...}, "id": 1}

This client parses the SSE `data:` line to extract the JSON-RPC payload, then
unwraps the MCP `result.content[0].text` envelope into the tool's JSON result.

Leapable tools return ``{"success": bool, "data": {...}}``. This client returns
that inner object verbatim; callers must check ``success`` and read ``data``.

The Leapable runtime MUST be running and reachable. If it is not, every call
fails fast with MeetingTranscriptStoreError -- there is no silent fallback.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx

from voiceagent.meeting.errors import MeetingTranscriptStoreError

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:4100/mcp"
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 120.0


def _parse_sse_response(text: str) -> dict[str, Any]:
    """Parse an SSE (Server-Sent Events) response body into JSON-RPC data.

    Leapable's MCP endpoint frames responses as:
        event: message
        data: {"jsonrpc": "2.0", "result": {...}, "id": 1}

    Args:
        text: Raw response body text.

    Returns:
        Parsed JSON dict from the ``data:`` line (or the whole body if it is
        already plain JSON).

    Raises:
        MeetingTranscriptStoreError: If no valid JSON payload is found.
    """
    # Try direct JSON first (in case the response is plain JSON).
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Parse SSE framing -- find the `data:` line and decode it.
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            json_str = line[5:].strip()
            if json_str:
                try:
                    return json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    continue

    raise MeetingTranscriptStoreError(
        f"No valid JSON found in Leapable SSE response: {text[:300]}"
    )


class LeapableClient:
    """HTTP JSON-RPC client for the Leapable MCP server.

    Automatically performs the MCP initialize/initialized handshake on the
    first tool call and uses the server-assigned session ID thereafter.

    Args:
        base_url: URL of the Leapable MCP endpoint.
    """

    def __init__(self, base_url: str = DEFAULT_URL) -> None:
        self._base_url = base_url.rstrip("/")
        # Empty until the server assigns one during the initialize handshake.
        # Per the Streamable-HTTP MCP spec the client MUST NOT send a
        # Mcp-Session-Id on the initialize request; the server issues it.
        self._session_id: str = ""
        self._request_id = 0
        self._client: httpx.AsyncClient | None = None
        self._initialized = False
        logger.info("LeapableClient created: url=%s", self._base_url)

    @property
    def session_id(self) -> str:
        """The MCP session ID (server-assigned after init)."""
        return self._session_id

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=CONNECT_TIMEOUT,
                    read=READ_TIMEOUT,
                    write=30.0,
                    pool=30.0,
                ),
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        """Headers for MCP requests. Includes the session ID once assigned."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        # Only send the session header after the server has assigned one.
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _ensure_initialized(self) -> None:
        """Perform the MCP initialize handshake if not yet done.

        Sends the initialize request, captures the server-assigned session ID,
        then sends notifications/initialized. Must complete before any
        tool calls.

        Raises:
            MeetingTranscriptStoreError: If the handshake fails.
        """
        if self._initialized:
            return

        client = self._get_client()
        headers = self._headers()

        init_payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "clipcannon-meeting-clone",
                    "version": "1.0",
                },
            },
            "id": 1,
        }

        try:
            resp = await client.post(
                self._base_url, json=init_payload, headers=headers,
            )
        except httpx.HTTPError as e:
            raise MeetingTranscriptStoreError(
                f"Leapable MCP initialize handshake failed at "
                f"{self._base_url}: {e}"
            ) from e

        if resp.status_code != 200:
            raise MeetingTranscriptStoreError(
                f"Leapable MCP initialize returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )

        # Use the server-assigned session ID for all subsequent calls.
        server_session = resp.headers.get("mcp-session-id", "")
        if server_session:
            self._session_id = server_session
            logger.info("Leapable assigned session: %s", server_session)

        notif_payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        try:
            await client.post(
                self._base_url,
                json=notif_payload,
                headers=self._headers(),
            )
        except httpx.HTTPError:
            pass  # Notification responses are optional per the MCP spec.

        self._initialized = True
        self._request_id = 1
        logger.info("Leapable MCP session initialized: %s", self._session_id)

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a Leapable MCP tool via JSON-RPC over HTTP.

        Automatically performs the MCP initialize handshake on the first call
        and parses the SSE response framing.

        Args:
            tool_name: Name of the Leapable tool (e.g. "leapable_db_create").
            arguments: Tool arguments as a dict.

        Returns:
            The tool result as a dict. For Leapable tools this is the
            ``{"success": bool, "data": {...}}`` envelope.

        Raises:
            MeetingTranscriptStoreError: If the HTTP request fails, the server
                returns a JSON-RPC error, or the response is malformed.
        """
        await self._ensure_initialized()

        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
            "id": self._request_id,
        }

        client = self._get_client()

        try:
            response = await client.post(
                self._base_url, json=payload, headers=self._headers(),
            )
        except httpx.ConnectError as e:
            raise MeetingTranscriptStoreError(
                f"Cannot connect to Leapable at {self._base_url}. "
                f"Is the Leapable runtime running? Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise MeetingTranscriptStoreError(
                f"Timeout calling Leapable tool '{tool_name}': {e}"
            ) from e
        except httpx.HTTPError as e:
            raise MeetingTranscriptStoreError(
                f"HTTP error calling Leapable tool '{tool_name}': {e}"
            ) from e

        if response.status_code not in (200, 202):
            raise MeetingTranscriptStoreError(
                f"Leapable returned HTTP {response.status_code} "
                f"for tool '{tool_name}': {response.text[:500]}"
            )

        body = _parse_sse_response(response.text)

        # Surface JSON-RPC protocol errors loudly.
        if "error" in body:
            err = body["error"]
            raise MeetingTranscriptStoreError(
                f"Leapable tool '{tool_name}' returned a JSON-RPC error: "
                f"code={err.get('code')}, message={err.get('message')}"
            )

        # Unwrap the MCP result envelope: result.content[0].text holds the
        # tool's JSON payload as a string.
        result = body.get("result", {})
        if isinstance(result, dict):
            if result.get("isError"):
                raise MeetingTranscriptStoreError(
                    f"Leapable tool '{tool_name}' reported isError=true: "
                    f"{str(result)[:500]}"
                )
            content = result.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    try:
                        return json.loads(first["text"])
                    except (json.JSONDecodeError, ValueError):
                        return {"text": first["text"]}
            return result
        return {"result": result}

    async def close(self) -> None:
        """Close the HTTP client and release all connections."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            self._initialized = False
            logger.info(
                "LeapableClient closed (session=%s)", self._session_id,
            )
