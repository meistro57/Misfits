"""
LLM Interface - Abstraction layer for different LLM providers.

This module provides a unified interface for communicating with different
Local LLM providers like Ollama, LM Studio, GPT4All, etc.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    GPT4ALL = "gpt4all"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class LLMConfig:
    """Configuration for LLM connection."""
    provider: LLMProvider
    host: str = "localhost"
    port: int = 11434
    model: str = "llama2:7b-chat"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    api_key: Optional[str] = None


class LLMResponse:
    """Response from LLM with additional metadata."""

    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}
        self.tokens_used = metadata.get("tokens_used", 0)
        self.response_time = metadata.get("response_time", 0.0)
        self.model_used = metadata.get("model", "unknown")


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate a response to the given prompt."""
        pass

    @abstractmethod
    async def generate_streaming_response(self, prompt: str,
                                        context: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Generate a streaming response to the given prompt."""
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """Check if the LLM service is healthy and responsive."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama local LLM service."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = f"http://{config.host}:{config.port}"

    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate response using Ollama API."""
        import time
        start_time = time.time()

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }

        # Add context if provided
        if context:
            system_prompt = self._build_system_prompt(context)
            payload["system"] = system_prompt

        try:
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")

                result = await response.json()

                response_time = time.time() - start_time

                return LLMResponse(
                    text=result.get("response", ""),
                    metadata={
                        "tokens_used": result.get("eval_count", 0),
                        "response_time": response_time,
                        "model": self.config.model,
                        "provider": "ollama"
                    }
                )

        except Exception as e:
            return LLMResponse(
                text=f"Error generating response: {str(e)}",
                metadata={"error": True, "provider": "ollama"}
            )

    async def generate_streaming_response(self, prompt: str,
                                        context: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Generate streaming response using Ollama API."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }

        if context:
            system_prompt = self._build_system_prompt(context)
            payload["system"] = system_prompt

        try:
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status != 200:
                    yield f"Error: Ollama API error {response.status}"
                    return

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                yield data['response']
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            yield f"Error: {str(e)}"

    async def check_health(self) -> bool:
        """Check Ollama service health."""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except:
            return False

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt from context."""
        character_id = context.get("character_id", "character")
        traits = context.get("traits")

        system_prompt = f"You are {character_id}, a character in a life simulation game."

        if traits and hasattr(traits, 'base_traits'):
            trait_list = ", ".join(traits.base_traits)
            system_prompt += f" Your personality traits are: {trait_list}."

            if hasattr(traits, 'dialogue_style'):
                system_prompt += f" You speak in a {traits.dialogue_style} manner."

        system_prompt += " Respond naturally and stay in character."

        return system_prompt


class LMStudioProvider(BaseLLMProvider):
    """Provider for LM Studio local LLM service."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = f"http://{config.host}:{config.port}"

    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate response using LM Studio OpenAI-compatible API."""
        import time
        start_time = time.time()

        messages = []

        if context:
            system_prompt = self._build_system_prompt(context)
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }

        try:
            async with self.session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
                if response.status != 200:
                    raise Exception(f"LM Studio API error: {response.status}")

                result = await response.json()

                response_time = time.time() - start_time

                message_content = result["choices"][0]["message"]["content"]

                return LLMResponse(
                    text=message_content,
                    metadata={
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                        "response_time": response_time,
                        "model": self.config.model,
                        "provider": "lm_studio"
                    }
                )

        except Exception as e:
            return LLMResponse(
                text=f"Error generating response: {str(e)}",
                metadata={"error": True, "provider": "lm_studio"}
            )

    async def generate_streaming_response(self, prompt: str,
                                        context: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Generate streaming response using LM Studio API."""
        messages = []

        if context:
            system_prompt = self._build_system_prompt(context)
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True
        }

        try:
            async with self.session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
                if response.status != 200:
                    yield f"Error: LM Studio API error {response.status}"
                    return

                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            line_str = line_str[6:]
                            if line_str == '[DONE]':
                                break
                            try:
                                data = json.loads(line_str)
                                delta = data["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            yield f"Error: {str(e)}"

    async def check_health(self) -> bool:
        """Check LM Studio service health."""
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                return response.status == 200
        except:
            return False

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt from context."""
        # Same as Ollama implementation
        character_id = context.get("character_id", "character")
        traits = context.get("traits")

        system_prompt = f"You are {character_id}, a character in a life simulation game."

        if traits and hasattr(traits, 'base_traits'):
            trait_list = ", ".join(traits.base_traits)
            system_prompt += f" Your personality traits are: {trait_list}."

            if hasattr(traits, 'dialogue_style'):
                system_prompt += f" You speak in a {traits.dialogue_style} manner."

        system_prompt += " Respond naturally and stay in character. Keep responses concise and character-appropriate."

        return system_prompt


class LLMInterface:
    """Main interface for LLM communication with fallback and retry logic."""

    def __init__(self, configs: List[LLMConfig], primary_provider: LLMProvider = None):
        self.configs = configs
        self.primary_provider = primary_provider or (configs[0].provider if configs else LLMProvider.OLLAMA)
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.health_status: Dict[LLMProvider, bool] = {}

        # Initialize providers
        for config in configs:
            if config.provider == LLMProvider.OLLAMA:
                self.providers[config.provider] = OllamaProvider(config)
            elif config.provider == LLMProvider.LM_STUDIO:
                self.providers[config.provider] = LMStudioProvider(config)
            # Add other providers as needed

    async def __aenter__(self):
        for provider in self.providers.values():
            await provider.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for provider in self.providers.values():
            await provider.__aexit__(exc_type, exc_val, exc_tb)

    async def generate_response(self, prompt: str, context: Dict[str, Any] = None,
                              retry_count: int = 2) -> LLMResponse:
        """Generate response with fallback logic."""

        # Try primary provider first
        if self.primary_provider in self.providers:
            try:
                response = await self.providers[self.primary_provider].generate_response(prompt, context)
                if not response.metadata.get("error", False):
                    return response
            except Exception as e:
                print(f"Primary provider {self.primary_provider.value} failed: {e}")

        # Try other providers as fallback
        for provider_type, provider in self.providers.items():
            if provider_type != self.primary_provider:
                try:
                    response = await provider.generate_response(prompt, context)
                    if not response.metadata.get("error", False):
                        return response
                except Exception as e:
                    print(f"Fallback provider {provider_type.value} failed: {e}")

        # If all providers failed, return error response
        return LLMResponse(
            text="I'm having trouble thinking right now...",
            metadata={"error": True, "all_providers_failed": True}
        )

    async def generate_streaming_response(self, prompt: str,
                                        context: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Generate streaming response with fallback."""

        # Try primary provider first
        if self.primary_provider in self.providers:
            try:
                async for chunk in self.providers[self.primary_provider].generate_streaming_response(prompt, context):
                    yield chunk
                return
            except Exception as e:
                print(f"Primary streaming provider {self.primary_provider.value} failed: {e}")

        # Try fallback to non-streaming
        try:
            response = await self.generate_response(prompt, context)
            yield response.text
        except Exception as e:
            yield f"Error: All providers failed - {e}"

    async def check_all_providers_health(self) -> Dict[str, bool]:
        """Check health of all configured providers."""
        health_results = {}

        for provider_type, provider in self.providers.items():
            try:
                is_healthy = await provider.check_health()
                health_results[provider_type.value] = is_healthy
                self.health_status[provider_type] = is_healthy
            except Exception as e:
                health_results[provider_type.value] = False
                self.health_status[provider_type] = False

        return health_results

    def get_available_providers(self) -> List[str]:
        """Get list of available healthy providers."""
        return [
            provider_type.value for provider_type, is_healthy
            in self.health_status.items() if is_healthy
        ]

    def set_primary_provider(self, provider: LLMProvider):
        """Set the primary provider to use."""
        if provider in self.providers:
            self.primary_provider = provider

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about provider usage and health."""
        return {
            "primary_provider": self.primary_provider.value,
            "available_providers": self.get_available_providers(),
            "health_status": {k.value: v for k, v in self.health_status.items()},
            "total_providers": len(self.providers)
        }


class MockLLMProvider(BaseLLMProvider):
    """Mock provider for testing without actual LLM service."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.responses = [
            "That sounds like a good idea!",
            "I'm not sure about that...",
            "Let me think about it for a moment.",
            "That's interesting! I hadn't considered that.",
            "I suppose we could try that approach.",
            "Hmm, that might work actually.",
            "I have a different perspective on this.",
            "That reminds me of something similar.",
        ]

    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate mock response."""
        import random
        import asyncio

        # Simulate processing time
        await asyncio.sleep(random.uniform(0.5, 2.0))

        response_text = random.choice(self.responses)

        # Add character-specific response if context provided
        if context and context.get("character_id"):
            char_id = context["character_id"]
            response_text = f"*{char_id} thinks* {response_text}"

        return LLMResponse(
            text=response_text,
            metadata={
                "tokens_used": len(response_text.split()),
                "response_time": random.uniform(0.5, 2.0),
                "model": "mock",
                "provider": "mock"
            }
        )

    async def generate_streaming_response(self, prompt: str,
                                        context: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Generate mock streaming response."""
        import random
        import asyncio

        response = await self.generate_response(prompt, context)
        words = response.text.split()

        for word in words:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            yield word + " "

    async def check_health(self) -> bool:
        """Mock provider is always healthy."""
        return True


def create_llm_interface_from_config(config_dict: Dict[str, Any]) -> LLMInterface:
    """Create LLM interface from configuration dictionary."""

    configs = []

    # Parse provider configurations
    for provider_name, provider_config in config_dict.get("providers", {}).items():
        try:
            provider_type = LLMProvider(provider_name.lower())

            config = LLMConfig(
                provider=provider_type,
                host=provider_config.get("host", "localhost"),
                port=provider_config.get("port", 11434),
                model=provider_config.get("model", "llama2:7b-chat"),
                temperature=provider_config.get("temperature", 0.7),
                max_tokens=provider_config.get("max_tokens", 1000),
                timeout=provider_config.get("timeout", 30),
                api_key=provider_config.get("api_key")
            )

            configs.append(config)

        except ValueError:
            print(f"Unknown provider: {provider_name}")
            continue

    # Add mock provider if no real providers configured
    if not configs:
        configs.append(LLMConfig(provider=LLMProvider.OLLAMA))  # Will use mock in testing

    # Set primary provider
    primary_provider_name = config_dict.get("primary_provider", "ollama")
    try:
        primary_provider = LLMProvider(primary_provider_name)
    except ValueError:
        primary_provider = LLMProvider.OLLAMA

    return LLMInterface(configs, primary_provider)