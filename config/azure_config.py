"""
Azure OpenAI Configuration Module

This module handles all Azure OpenAI configuration and client initialization.
"""
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, ClientSecretCredential

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


class AzureOpenAIConfig:
    """Configuration class for Azure OpenAI"""
    
    def __init__(self):
        # Required configurations
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
        
        # Optional different deployments
        self.gpt4_deployment = os.getenv('AZURE_OPENAI_GPT4_DEPLOYMENT', self.deployment_name)
        self.gpt35_deployment = os.getenv('AZURE_OPENAI_GPT35_DEPLOYMENT', self.deployment_name)
        self.embedding_deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
        
        # Azure AD authentication (optional)
        self.tenant_id = os.getenv('AZURE_TENANT_ID')
        self.client_id = os.getenv('AZURE_CLIENT_ID')
        self.client_secret = os.getenv('AZURE_CLIENT_SECRET')
        
        # API Settings
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.timeout_seconds = int(os.getenv('TIMEOUT_SECONDS', '60'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '2000'))
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate required configuration"""
        required = ['endpoint', 'deployment_name', 'api_version']
        
        # Check if we have either API key or Azure AD credentials
        if not self.api_key and not (self.tenant_id and self.client_id and self.client_secret):
            raise ValueError("Either AZURE_OPENAI_API_KEY or Azure AD credentials must be provided")
        
        for field in required:
            if not getattr(self, field):
                raise ValueError(f"Missing required configuration: {field.upper()}")
    
    def get_client(self) -> AzureOpenAI:
        """Get Azure OpenAI client"""
        if self.api_key:
            # Use API key authentication
            return AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                max_retries=self.max_retries,
                timeout=self.timeout_seconds
            )
        else:
            # Use Azure AD authentication
            credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            return AzureOpenAI(
                azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                max_retries=self.max_retries,
                timeout=self.timeout_seconds
            )
    
    def get_model_config(self, model_type: str = "default") -> Dict[str, Any]:
        """Get model configuration based on type"""
        deployment_map = {
            "default": self.deployment_name,
            "gpt4": self.gpt4_deployment,
            "gpt35": self.gpt35_deployment,
            "embedding": self.embedding_deployment
        }
        
        return {
            "deployment_name": deployment_map.get(model_type, self.deployment_name),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for library-specific use)"""
        return {
            "api_key": self.api_key,
            "endpoint": self.endpoint,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


# Global configuration instance
config = AzureOpenAIConfig()
