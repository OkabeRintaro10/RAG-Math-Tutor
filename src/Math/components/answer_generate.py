from typing import Any

from guardrails.errors import ValidationError
from langchain_openai import ChatOpenAI

from src.Math import logger
from src.Math.entity.config_entity import GraphState


class Generate:
    def __init__(self, output_guard, llm: ChatOpenAI):
        self.output_guard = output_guard
        self.llm = llm

    async def generate(self, state: GraphState) -> dict[str, Any]:
        """Generate answer using LLM with context and output validation.

        Returns:
            Dictionary with generation and updated history_tokens
        """
        logger.debug("Running generate node")
        question = state.question
        documents = state.documents
        summary = state.summary
        current_history_tokens = state.history_tokens

        # Prepare context from documents
        valid_docs = [doc for doc in documents if doc and doc.strip()]
        context_str = "\n\n".join(valid_docs) if valid_docs else ""

        # Build prompt based on available context
        if context_str:
            prompt = (
                "You are a helpful math assistant. Use the following context to "
                "answer the question accurately and clearly.\n\n"
                "If the context doesn't contain enough information, say so honestly.\n\n"
                f"--- CONTEXT ---\n{context_str}\n\n"
            )
            if summary:
                prompt += f"--- CONVERSATION HISTORY ---\n{summary}\n\n"
            prompt += f"--- QUESTION ---\n{question}\n\n--- ANSWER ---\n"
        else:
            prompt = (
                "You are a helpful math assistant. Answer the following question "
                "to the best of your ability.\n\n"
            )
            if summary:
                prompt += f"--- CONVERSATION HISTORY ---\n{summary}\n\n"
            prompt += f"--- QUESTION ---\n{question}\n\n--- ANSWER ---\n"

        try:
            response = await self.llm.ainvoke(prompt)

            # Track token usage
            token_usage = response.response_metadata.get("token_usage", {})
            generation_cost = token_usage.get("total_tokens", 0)
            new_total_tokens = current_history_tokens + generation_cost

            content = getattr(response, "content", str(response))

            # Validate output with guardrails
            try:
                self.output_guard.validate(text_to_validate=content)
                logger.info(
                    f"✅ Generation complete and validated ({len(content)} chars, {generation_cost} tokens)"
                )
            except ValidationError as ve:
                logger.warning(f"⚠️ Output validation failed: {ve}")
                # Continue anyway but log the issue

            return {"generation": content, "history_tokens": new_total_tokens}

        except ValidationError as ve:
            logger.error(f"❌ Output guardrail rejection: {ve}")
            return {
                "generation": (
                    "I apologize, but I couldn't generate a satisfactory answer. "
                    "Could you please rephrase your question?"
                ),
                "history_tokens": current_history_tokens,
            }

        except Exception as exc:
            logger.exception(f"❌ Generation failed: {exc}")
            return {
                "generation": (
                    "An error occurred while generating the answer. Please try again."
                ),
                "history_tokens": current_history_tokens,
            }
