import logging
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.types import ChatMessage

logger = logging.getLogger(__name__)

class PicoclawLLM(BasePipelineElement):
    """A mock LLM that uses the picoclaw agent to generate responses."""

    def __init__(self) -> None:
        self.call_count = 0

    def query(
        self,
        query: str,
        runtime,
        env,
        messages: list[ChatMessage],
        extra_args: dict,
    ) -> tuple[str, dict, dict, list[ChatMessage], dict]:
        logger.info(f"PicoclawLLM called with query: {query[:100]}...")
        self.call_count += 1

        # For the first call, return a simple response.
        # For subsequent calls, we can return something else to avoid infinite loops.
        if self.call_count == 1:
            response = "I am the picoclaw agent. I will now use the finish tool to indicate that I have completed the task."
        else:
            response = "I have already finished the task."

        # We return the response as the new query, and we don't change the runtime, env, messages, or extra_args.
        return response, runtime, env, messages, extra_args