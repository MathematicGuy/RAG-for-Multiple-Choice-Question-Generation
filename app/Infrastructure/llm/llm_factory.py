"""
LLM Factory for creating different LLM providers.
Implements factory pattern for LLM repository creation.
"""

from typing import Dict, Any, Optional
from Domain.repositories.llm_repository import LLMRepository
from Infrastructure.llm.together_ai_client import TogetherAIClient


class LLMFactory:
    """Factory for creating LLM repository instances"""

    @staticmethod
    def create_llm(provider: str, config: Dict[str, Any]) -> LLMRepository:
        """
        Create LLM repository instance based on provider

        Args:
            provider: LLM provider name
            config: Configuration dictionary

        Returns:
            LLM repository instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider.lower() == "together_ai":
            return LLMFactory._create_together_ai(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _create_together_ai(config: Dict[str, Any]) -> TogetherAIClient:
        """
        Create Together AI client

        Args:
            config: Together AI configuration

        Returns:
            TogetherAIClient instance

        Raises:
            ValueError: If required configuration is missing
        """
        required_keys = ["api_key"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration: {key}")

        return TogetherAIClient(
            api_key=config["api_key"],
            model=config.get("model", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
            base_url=config.get("base_url", "https://api.together.xyz/v1"),
            max_retries=config.get("max_retries", 3),
            timeout=config.get("timeout", 30)
        )

    @staticmethod
    def get_supported_providers() -> list:
        """
        Get list of supported LLM providers

        Returns:
            List of supported provider names
        """
        return ["together_ai"]

    @staticmethod
    def validate_config(provider: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a provider

        Args:
            provider: LLM provider name
            config: Configuration to validate

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if provider.lower() == "together_ai":
            return LLMFactory._validate_together_ai_config(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _validate_together_ai_config(config: Dict[str, Any]) -> bool:
        """
        Validate Together AI configuration

        Args:
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ["api_key"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration for Together AI: {key}")

            if not config[key] or not isinstance(config[key], str):
                raise ValueError(f"Invalid value for {key}: must be a non-empty string")

        # Validate optional parameters
        if "model" in config and not isinstance(config["model"], str):
            raise ValueError("Model must be a string")

        if "max_retries" in config:
            if not isinstance(config["max_retries"], int) or config["max_retries"] < 1:
                raise ValueError("max_retries must be a positive integer")

        if "timeout" in config:
            if not isinstance(config["timeout"], (int, float)) or config["timeout"] <= 0:
                raise ValueError("timeout must be a positive number")

        return True
