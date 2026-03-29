import logging
import os
import requests
import time
import uuid
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.types import ChatMessage, get_text_content_as_str

logger = logging.getLogger(__name__)


class PicoclawLLM(BasePipelineElement):
    """An adapter for the PicoClaw agent platform."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        if base_url is None:
            base_url = os.getenv("PICOCLAW_BASE_URL")
        if base_url is None:
            host = os.getenv("PICOCLAW_HOST", "localhost")
            port = os.getenv("PICOCLAW_PORT", "18789")
            base_url = f"http://{host}:{port}"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("PICOCLAW_API_KEY")

    def query(
        self,
        query: str,
        runtime,
        env,
        messages: list[ChatMessage],
        extra_args: dict,
    ) -> tuple[str, dict, dict, list[ChatMessage], dict]:
        logger.info(f"PicoclawLLM query: {query[:100]}...")

        chat_url = f"{self.base_url}/chat"
        reply = ""
        updated_messages = list(messages)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        # Use a unique session ID per query
        session_id = str(uuid.uuid4())
        headers["X-Session-ID"] = session_id

        last_user_message = next((m for m in reversed(messages) if m["role"] == "user"), None)
        message_text = get_text_content_as_str(last_user_message["content"]) if last_user_message else query

        # 1. POST to start the async request, capture session_id from response
        try:
            response = requests.post(chat_url, json={"message": message_text}, headers=headers)
            response.raise_for_status()
            post_data = response.json()
            session_id = post_data.get("session_id", session_id)
            logger.info(f"PicoClaw job started, session_id: {session_id}")
        except Exception as e:
            logger.error(f"Error calling PicoClaw: {e}")
            reply = f"Error: {e}"
            updated_messages = list(messages) + [
                {"role": "assistant", "content": [{"type": "text", "content": reply}], "tool_calls": []}
            ]
            return reply, runtime, env, updated_messages, extra_args

        # 2. Poll GET /chat?session_id=... until status == "completed"
        poll_headers = {"X-API-Key": self.api_key} if self.api_key else {}
        for attempt in range(30):  # max 30 attempts, 60 seconds total
            time.sleep(2)
            try:
                poll_response = requests.get(
                    chat_url,
                    params={"session_id": session_id},
                    headers=poll_headers
                )
                poll_response.raise_for_status()
                data = poll_response.json()
                logger.info(f"Poll attempt {attempt + 1}: status={data.get('status')}")

                status = data.get("status", "")
                if status == "completed":
                    reply = data.get("response", "")
                    break
            except Exception as e:
                logger.error(f"Poll error: {e}")
                break

        if not reply:
            logger.warning(f"PicoClaw returned empty response after polling for session {session_id}")

        updated_messages = list(messages) + [
            {"role": "assistant", "content": [{"type": "text", "content": reply}], "tool_calls": []}
        ]
        return reply, runtime, env, updated_messages, extra_args
