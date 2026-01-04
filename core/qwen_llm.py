"""Custom Qwen LLM implementation for handling API-specific parameters"""
import logging
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
import requests
import json


class QwenChat(BaseChatModel):
    """Custom Chat Model for Qwen API that handles enable_thinking parameter properly"""
    
    model_name: str = Field(default="Qwen/Qwen3-32B")
    api_key: str
    base_url: str = Field(default="https://api-inference.modelscope.cn/v1")
    temperature: float = Field(default=0.1)
    timeout: int = Field(default=60)
    
    @property
    def _llm_type(self) -> str:
        return "qwen-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completions with proper Qwen API parameters"""
        
        # Format messages for API
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            else:
                formatted_messages.append({"role": "user", "content": str(msg.content)})

        # Prepare the request payload with enable_thinking set to false for non-streaming
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "enable_thinking": False,  # Explicitly set to false for non-streaming calls
            "stream": False
        }
        
        # Add any additional parameters from kwargs
        payload.update(kwargs)
        
        if stop:
            payload["stop"] = stop

        # Make the API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            response_data = response.json()
            
            # Extract the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                message = AIMessage(content=content)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            else:
                raise Exception(f"Unexpected API response format: {response_data}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making API request: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing Qwen API response: {str(e)}")

    def _create_chat_result(self, response: Dict[str, Any]) -> ChatResult:
        """Create a ChatResult from the API response"""
        # This method is called internally by _generate
        pass