"""Input and output guards for a math-related chatbot.

This module provides validation for user inputs and LLM outputs using
the Guardrails AI library with custom validators.
"""

from __future__ import annotations

import json
import os

import litellm
from guardrails import Guard
from guardrails.errors import ValidationError
from guardrails.hub import RestrictToTopic
from openai import OpenAI

from src.Math import logger

# Global Client for Embeddings
try:
    embedding_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
except Exception as e:
    print(f"Warning: Failed to initialize embedding client: {e}")
    embedding_client = None

DEVICE = -1


def _openrouter_llm_callable(prompt: str, topics: list[str]) -> list[str]:
    """Call OpenRouter LLM for topic classification.

    This function formats the prompt as JSON and calls the LLM to determine
    which topics are present in the given text.

    Args:
        prompt: The text to analyze
        topics: List of topics to check for

    Returns:
        List of topics found in the text
    """
    logger.info("--- DEBUGGING VALIDATOR ---")
    logger.info(f"Validator received text:\n{prompt[:100]}...")
    logger.info(f"Validator is using topics:\n{topics}")

    json_prompt = f"""
            Given a text and a list of topics, return a valid JSON list of which topics \
            are present in the text. If none, just return an empty list.

            Output Format:
            -------------
            {{"topics_present": []}}

            Text:
            ----
            "{prompt}"

            Topics:
            ------
            {topics}

            Result:
            ------
            """

    try:
        response = litellm.completion(
            model="openrouter/google/gemma-3-4b-it",
            messages=[{"role": "user", "content": json_prompt}],
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            response_format={"type": "json_object"},
        )

        raw_response = response.choices[0].message.content
        # Parse JSON response
        json_str = raw_response[raw_response.find("{") : raw_response.rfind("}") + 1]
        json_data = json.loads(json_str)

        topics_found = json_data.get("topics_present", [])

        logger.info(f"Validator is returning this list: {topics_found}")

        return topics_found

    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        logger.error(f"Validator failed to parse JSON or access response data: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred in the validator: {e}")
        return []


class InputGuard:
    """Validates user input to ensure it's math-related.

    This guard checks if the user's question is related to mathematics
    using semantic similarity and LLM-based classification.
    """

    def __init__(self) -> None:
        """Initialize the input guard with math-related topics."""
        if not embedding_client:
            raise ConnectionError("Embedding client not initialized.")

        self.guard = Guard().use(
            RestrictToTopic(
                valid_topics=[
                    "math",
                    "mathematics",
                    "algebra",
                    "calculus",
                    "geometry",
                    "trigonometry",
                    "equations",
                    "statistics",
                    "probability",
                    "theorems",
                    "proofs",
                    "arithmetic",
                    "number theory",
                    "linear algebra",
                    "differential equations",
                ],
                device=DEVICE,
                llm_callable=_openrouter_llm_callable,
                disable_classifier=False,
                disable_llm=False,
                on_fail="exception",
            )
        )
        self.embedding_client = embedding_client

    def validate(self, text_to_validate: str) -> bool:
        """Validate that the input is math-related.

        Args:
            text_to_validate: The user's question to validate

        Returns:
            True if validation passes

        Raises:
            ValidationError: If the input is not math-related
        """
        if not text_to_validate or not text_to_validate.strip():
            raise ValidationError("Input text is empty.")

        validation_result = self.guard.parse(
            llm_output=text_to_validate,
        )

        if not validation_result.validation_passed:
            raise ValidationError("Input is not a math-related topic.")

        return True


class OutputGuard:
    """Validates LLM output to ensure it's not a refusal.

    This guard checks if the LLM's response is a refusal or unhelpful
    response using semantic similarity to common refusal patterns.
    """

    def __init__(self) -> None:
        """Initialize the output guard with refusal patterns."""
        if not embedding_client:
            raise ConnectionError("Embedding client not initialized.")

        self.guard = Guard().use(
            RestrictToTopic(
                invalid_topics=[
                    "I don't know",
                    "I cannot answer",
                    "I'm sorry, I can't",
                    "I am not programmed to",
                    "I don't have enough information",
                    "As an AI, I cannot help with that",
                    "I'm unable to assist",
                    "I apologize, but I cannot",
                    "I don't have access to",
                    "I'm not able to",
                ],
                device=DEVICE,
                llm_callable=_openrouter_llm_callable,
                disable_classifier=True,
                disable_llm=False,
                on_fail="noop",
            )
        )
        self.embedding_client = embedding_client

    def validate(self, text_to_validate: str) -> bool:
        """Validate that the output is not a refusal.

        Args:
            text_to_validate: The LLM's response to validate

        Returns:
            True if validation passes (response is helpful)

        Raises:
            ValidationError: If the response is a refusal
        """
        if not text_to_validate or not text_to_validate.strip():
            raise ValidationError("Output text is empty.")

        validation_result = self.guard.parse(
            llm_output=text_to_validate,
        )

        # Reverse logic: invalid_topics means FAILED = found refusal (bad)
        if not validation_result.validation_passed:
            print(f"Output validation failed: {validation_result}")
            raise ValidationError("The LLM refused to answer.")

        return True
