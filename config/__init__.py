# Make config module importable
from .azure_config import config, AzureOpenAIConfig

__all__ = ['config', 'AzureOpenAIConfig']
