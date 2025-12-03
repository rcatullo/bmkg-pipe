from openai import AzureOpenAI
import logging

class OpenAIUtils:
    def __init__(self, api_key: str):
        
        self.azure_client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-12-01-preview",
            azure_endpoint="https://ovalnairr.openai.azure.com/",
        )
    
    def _query_azure_openai(self, prompt: str, system_prompt: str = "") -> str:
        """
        Query Azure OpenAI for schema description.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            
        Returns:
            Model response as string
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.azure_client.chat.completions.create(
                model="o3",  # your Azure deployment name
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Azure OpenAI API error: {e}")
            raise