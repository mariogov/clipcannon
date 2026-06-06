"""Meeting response generation via local Ollama LLM with Leapable RAG.

Generates direct, concise answers to questions directed at a clone. When a
Leapable client is provided, searches meeting history for relevant context
before generating a response (Retrieval-Augmented Generation).

Rules:
- Answer ONLY the specific question asked
- 1-3 sentences maximum (unless question requires detail)
- No preamble, no extra context, no follow-up questions
- No markdown, no lists -- spoken conversation
- Stop immediately after answering
- Use meeting history context when available and relevant
"""
from __future__ import annotations

import logging

import httpx

from voiceagent.meeting.config import ResponseConfig
from voiceagent.meeting.errors import MeetingResponseError
from voiceagent.meeting.mcp_client import LeapableClient
from voiceagent.meeting.transcript_format import MeetingSegment

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = (
    "You are {clone_name}, attending a video meeting. "
    "You have been directly asked a question. "
    "Answer ONLY the specific question asked. "
    "Be direct and concise -- 1-3 sentences maximum. "
    "Do not volunteer additional information. "
    "Do not ask follow-up questions. "
    "Do not use filler phrases like 'great question' or 'that is interesting'. "
    "Stop talking as soon as you have answered the question. "
    "No markdown, no lists, no formatting -- this is spoken conversation. "
    "Use full prosody -- vary your pitch, pace, and emphasis naturally. "
    "Speak as a real person would in conversation, not as a text reader."
)

SYSTEM_PROMPT_WITH_HISTORY = (
    "You are {clone_name}, attending a video meeting. "
    "You have been directly asked a question. "
    "You have access to context from past meetings and the current meeting. "
    "Use this context to give informed, accurate answers. "
    "If the context contains the answer, use it. If not, say so honestly. "
    "Answer ONLY the specific question asked. "
    "Be direct and concise -- 1-3 sentences for simple questions, "
    "up to 5 sentences if the question requires detail from meeting history. "
    "Do not volunteer additional information beyond what was asked. "
    "No markdown, no lists, no formatting -- this is spoken conversation. "
    "Use full prosody -- vary your pitch, pace, and emphasis naturally."
)


# Maximum time (seconds) to wait for Ollama to return a response.
_OLLAMA_TIMEOUT_S = 60.0

# Maximum number of recent meeting segments to include as context.
_MAX_CONTEXT_SEGMENTS = 20

# Maximum number of Leapable search results to include as history context.
_MAX_HISTORY_RESULTS = 5

# Leapable's hybrid search requires natural-language queries of at least this
# many words; shorter queries are rejected by the server, so we skip them.
_MIN_SEARCH_WORDS = 8


class MeetingResponder:
    """Generate meeting responses via local Ollama LLM with optional RAG.

    When a Leapable client is provided, searches meeting history for relevant
    context before generating responses, letting the clone answer questions
    about past meetings, action items, and historical context.

    Args:
        config: Response generation configuration (model, temperature, etc.).
        clone_name: Name of the clone generating responses.
        leapable_client: Optional Leapable client for meeting-history search.
    """

    def __init__(
        self,
        config: ResponseConfig,
        clone_name: str = "Nate",
        leapable_client: LeapableClient | None = None,
    ) -> None:
        self._config = config
        self._clone_name = clone_name
        self._base_url = "http://localhost:11434"
        self._leapable_client = leapable_client
        self._system_prompt = (
            config.system_prompt_override or (
                SYSTEM_PROMPT_WITH_HISTORY if leapable_client
                else SYSTEM_PROMPT_TEMPLATE
            )
        ).format(clone_name=clone_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_response(
        self,
        question: str,
        meeting_context: list[MeetingSegment] | None = None,
    ) -> str:
        """Generate a direct answer to a question.

        Args:
            question: The question asked of the clone.
            meeting_context: Recent meeting segments for context (last ~2 min).

        Returns:
            Response text (1-3 sentences, no formatting).

        Raises:
            MeetingResponseError: If Ollama is unreachable, times out,
                returns a non-200 status, or produces an empty response.
        """
        context_str = self._build_context(meeting_context)
        history_str = await self._search_history(question)

        # Build prompt with all available context
        parts: list[str] = []
        if history_str:
            parts.append(f"Relevant context from past meetings:\n{history_str}")
        if context_str:
            parts.append(f"Current meeting context:\n{context_str}")
        parts.append(f"Question directed at you: {question}")

        prompt = "\n\n".join(parts)

        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "system": self._system_prompt,
            "stream": False,
            "options": {
                "num_predict": self._config.max_tokens,
                "temperature": self._config.temperature,
                "top_p": 0.8,
                "top_k": 20,
            },
        }

        resp = await self._call_ollama(payload)
        return self._parse_response(resp, question)

    # ------------------------------------------------------------------
    # Leapable history search (RAG)
    # ------------------------------------------------------------------

    async def _search_history(self, question: str) -> str:
        """Search Leapable meeting history for context relevant to a question.

        Best-effort: if no Leapable client is configured, the query is too
        short for Leapable's hybrid search, or the search fails, returns an
        empty string and logs. Never raises -- history is augmentation, not a
        hard dependency of answering.

        Args:
            question: The question to find relevant prior context for.

        Returns:
            Formatted relevant meeting-history excerpts, or "" if unavailable.
        """
        if self._leapable_client is None:
            return ""

        # Leapable rejects short keyword queries; only search real questions.
        if len(question.split()) < _MIN_SEARCH_WORDS:
            logger.debug(
                "Skipping Leapable history search; query has <%d words: %r",
                _MIN_SEARCH_WORDS, question,
            )
            return ""

        try:
            envelope = await self._leapable_client.call_tool(
                "leapable_search",
                {"query": question, "limit": _MAX_HISTORY_RESULTS},
            )
        except Exception as exc:  # noqa: BLE001 -- best-effort augmentation
            logger.debug("Leapable history search failed: %s", exc)
            return ""

        return self._format_search_results(envelope)

    @staticmethod
    def _format_search_results(envelope: dict) -> str:
        """Format a Leapable search envelope into a context string.

        Respects Leapable's ``low_confidence`` flag: when the Vault does not
        cover the question, returns "" rather than feeding noise to the LLM.

        Args:
            envelope: Raw ``{"success", "data"}`` envelope from leapable_search.

        Returns:
            Formatted relevant excerpts, or "" if nothing useful.
        """
        if not isinstance(envelope, dict):
            return ""
        data = envelope.get("data", envelope)
        if not isinstance(data, dict):
            return ""
        # Do not fabricate context when Leapable flags low confidence.
        if data.get("low_confidence"):
            return ""

        results = data.get("results", [])
        if not isinstance(results, list) or not results:
            return ""

        lines: list[str] = []
        for item in results[:_MAX_HISTORY_RESULTS]:
            if not isinstance(item, dict):
                continue
            text = item.get("original_text") or item.get("text") or ""
            if not text:
                continue
            source = item.get("source_file_name", "")
            excerpt = text[:300].strip()
            if len(text) > 300:
                excerpt += "..."
            lines.append(f"[{source}] {excerpt}" if source else excerpt)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(
        meeting_context: list[MeetingSegment] | None,
    ) -> str:
        """Format recent meeting segments into a context string."""
        if not meeting_context:
            return ""
        lines: list[str] = []
        for seg in meeting_context[-_MAX_CONTEXT_SEGMENTS:]:
            speaker = seg.speaker_name or seg.speaker_id or "Unknown"
            lines.append(f"{speaker}: {seg.text}")
        return "\n".join(lines)

    async def _call_ollama(self, payload: dict) -> httpx.Response:
        """POST to Ollama and return the raw response.

        Raises:
            MeetingResponseError: On connection or timeout failures.
        """
        async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT_S) as client:
            try:
                resp = await client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
            except httpx.ConnectError as exc:
                raise MeetingResponseError(
                    f"Cannot connect to Ollama at {self._base_url}. "
                    f"Is Ollama running? Start with: ollama serve. "
                    f"Error: {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise MeetingResponseError(
                    f"Ollama response timed out after "
                    f"{_OLLAMA_TIMEOUT_S:.0f}s: {exc}"
                ) from exc
        return resp

    @staticmethod
    def _parse_response(resp: httpx.Response, question: str) -> str:
        """Extract and validate response text from the Ollama HTTP response.

        Raises:
            MeetingResponseError: On non-200 status, bad JSON, or empty text.
        """
        if resp.status_code != 200:
            raise MeetingResponseError(
                f"Ollama returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )

        try:
            body = resp.json()
        except ValueError as exc:
            raise MeetingResponseError(
                f"Invalid JSON from Ollama: {exc}"
            ) from exc

        response_text = body.get("response", "").strip()
        if not response_text:
            raise MeetingResponseError(
                f"Ollama returned empty response for question: "
                f"'{question[:100]}'"
            )

        logger.info(
            "Response generated (%d chars): '%s'",
            len(response_text),
            response_text[:100],
        )
        return response_text
