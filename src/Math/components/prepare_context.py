from typing import Any

from langchain_openai import ChatOpenAI
from src.Math import logger
from src.Math.entity.config_entity import GraphState


class Prepare_Context:
    def __init__(
        self,
        llm: ChatOpenAI,
        summary_threshold: int = 102400,
    ):
        self.llm = llm
        self.summary_threshold = summary_threshold

    def prepare_context(self, state: GraphState) -> dict[str, Any]:
        """Prepare conversation context from history.

        Handles two cases:
        1. History below threshold: Use full history
        2. History above threshold: Summarize with LLM

        Returns:
            Dictionary with summary and updated history_tokens
        """
        logger.debug("Running prepare_context node")
        history = state.history
        question = state.question
        current_history_tokens = state.history_tokens

        if not history:
            logger.debug("No conversation history")
            return {"summary": "", "history_tokens": 0}

        logger.info(
            f"History token count: {current_history_tokens}/{self.summary_threshold}"
        )

        if current_history_tokens <= self.summary_threshold:
            # Use full history
            logger.info("Using full conversation history")
            prompt_history = []
            for msg in history:
                role = "User" if msg.get("sender") == "user" else "Assistant"
                prompt_history.append(f"{role}: {msg.get('text', '')}")
            history_str = "\n".join(prompt_history)

            return {
                "summary": history_str,
                "history_tokens": current_history_tokens,
            }
        else:
            # Summarize history
            logger.info("History exceeds threshold. Summarizing...")
            prompt_history = []
            for msg in history:
                role = "User" if msg.get("sender") == "user" else "Assistant"
                prompt_history.append(f"{role}: {msg.get('text', '')}")
            history_str = "\n".join(prompt_history)

            summary_prompt = (
                "You are a helpful summarization assistant. "
                "Condense the following conversation into a concise summary. "
                "Preserve key mathematical concepts, formulas, and important details. "
                "Focus on information relevant to the user's new question.\n\n"
                "--- CONVERSATION HISTORY ---\n"
                f"{history_str}\n\n"
                "--- USER'S NEW QUESTION ---\n"
                f"{question}\n\n"
                "Provide a brief, focused summary:\n"
            )

            try:
                summarizer_llm = self.llm
                response = summarizer_llm.invoke(summary_prompt)

                new_summary = getattr(response, "content", str(response))
                new_token_count = response.response_metadata.get("token_usage", {}).get(
                    "total_tokens", 0
                )

                logger.info(f"✅ Summarization complete. New tokens: {new_token_count}")
                return {"summary": new_summary, "history_tokens": new_token_count}

            except Exception as e:
                logger.error(f"❌ Summarization failed: {e}")
                return {"summary": "", "history_tokens": 0}
