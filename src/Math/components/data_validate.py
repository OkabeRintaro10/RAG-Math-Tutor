from guardrails.errors import ValidationError
from src.Math import logger
from src.Math.entity.config_entity import GraphState
from typing import Any


class ValidateQuestion:
    def __init__(self, input_guard):
        self.input_guard = input_guard

    def validate_question(self, state: GraphState) -> dict[str, Any]:
        """Validate that the question is math-related using Guardrails.

        Returns:
            Dictionary with is_valid flag and optional error message
        """
        logger.info("Running validate_question node")
        question = state.question

        try:
            self.input_guard.validate(text_to_validate=question)
            logger.info("✅ Input validation passed")
            return {"is_valid": True}
        except ValidationError as e:
            logger.warning(f"❌ Input validation failed: {e}")
            return {
                "is_valid": False,
                "generation": "I can only answer math-related questions. Please ask about mathematical concepts, problems, or theories.",
            }
