"""
LLM Client for Azure OpenAI o3

This module provides a wrapper around the Azure OpenAI API for making
LLM calls during the reasoning process.
"""

import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from pydantic import BaseModel


class Message(BaseModel):
    """Represents a message in the conversation"""
    role: str  # "system", "user", or "assistant"
    content: str


class LLMResponse(BaseModel):
    """Response from the LLM"""
    content: str
    finish_reason: str
    usage: Dict[str, int]
    raw_response: Any = None


class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI o3 model.
    
    This client handles all LLM API calls during the reasoning process.
    """
    
    def __init__(
        self,
        azure_endpoint: Optional[str] = "https://ovalnairr.openai.azure.com/",
        api_key: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        deployment_name: str = "gpt-4.1",
    ):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version to use
            deployment_name: Name of the o3 deployment
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.deployment_name = deployment_name
        
        if not self.azure_endpoint or not self.api_key:
            raise ValueError("Azure endpoint and API key must be provided")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
    
    def chat_completion(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """
        Make a chat completion request.
        
        Args:
            messages: List of conversation messages
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters for the API
            
        Returns:
            LLMResponse containing the model's response
        """
        
        # Convert Message objects to dicts
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=message_dicts,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            finish_reason=response.choices[0].finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            raw_response=response
        )
    
    def stream_chat_completion(
        self,
        messages: List[Message],
        **kwargs
    ):
        """
        Make a streaming chat completion request.
        
        Args:
            messages: List of conversation messages
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters for the API
            
        Yields:
            Chunks of the response as they arrive
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        stream = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=message_dicts,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
