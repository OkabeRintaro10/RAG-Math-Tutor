# Save as: src/Math/components/query_rewrite.py

from typing import Any
from langchain_openai import ChatOpenAI
from src.Math import logger
from src.Math.entity.config_entity import GraphState


class QueryRewriter:
    """
    A component to rewrite the user's question based on chat history
    to make it a standalone query.
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.rewrite_prompt = (
            "You are a query-rewriting expert. "
            "Given a chat history and a new user question, rewrite the question "
            "to be a standalone, clear query that can be understood without the history. "
            "If the question is already standalone, return it as is.\n\n"
            "--- CHAT HISTORY ---\n"
            "{history}\n\n"
            "--- USER QUESTION ---\n"
            "{question}\n\n"
            "--- REWRITTEN QUERY ---"
        )

    async def rewrite_query(self, state: GraphState) -> dict[str, Any]:
        """Rewrite the user's question based on chat history."""
        logger.info("Running rewrite_query node")
        question = state.question
        history = state.history

        if not history:
            logger.info("No history, using original question.")
            # This is the first question, no rewriting needed.
            return {"question": question}

        # Format history for the prompt
        prompt_history = []
        for msg in history:
            role = "User" if msg.get("sender") == "user" else "Assistant"
            prompt_history.append(f"{role}: {msg.get('text', '')}")
        history_str = "\n".join(prompt_history)

        try:
            prompt = self.rewrite_prompt.format(history=history_str, question=question)

            response = await self.llm.ainvoke(prompt)
            rewritten_question = getattr(response, "content", str(response)).strip()

            # Remove potential quotes from the LLM's output
            rewritten_question = rewritten_question.strip('"')

            logger.info(f"Original question: '{question}'")
            logger.info(f"Rewritten question: '{rewritten_question}'")

            # Update the state with the new, better question
            return {"question": rewritten_question}

        except Exception as e:
            logger.error(f"❌ Query rewriting failed: {e}. Using original question.")
            # Fallback to the original question on error
            return {"question": question}
