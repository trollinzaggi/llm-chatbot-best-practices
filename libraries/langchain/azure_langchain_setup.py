"""
LangChain Setup for Azure OpenAI

This module demonstrates how to set up and use LangChain with Azure OpenAI.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Dict, List, Optional, Any
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import StdOutCallbackHandler
from config import config
from utils import setup_logger, log_api_call

# Set up logging
logger = setup_logger(__name__)


class AzureLangChainSetup:
    """Azure OpenAI integrated LangChain setup"""
    
    def __init__(self):
        """Initialize LangChain with Azure OpenAI"""
        self.llm = self._create_llm()
        self.embeddings = self._create_embeddings()
        self.memory = None
        logger.info("Initialized LangChain with Azure OpenAI")
    
    def _create_llm(self) -> AzureChatOpenAI:
        """Create Azure OpenAI LLM instance"""
        return AzureChatOpenAI(
            azure_deployment=config.deployment_name,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            callbacks=[StdOutCallbackHandler()]
        )
    
    def _create_embeddings(self) -> Optional[AzureOpenAIEmbeddings]:
        """Create Azure OpenAI Embeddings instance"""
        if config.embedding_deployment:
            return AzureOpenAIEmbeddings(
                azure_deployment=config.embedding_deployment,
                azure_endpoint=config.endpoint,
                api_key=config.api_key,
                api_version=config.api_version
            )
        return None
    
    def create_simple_chain(self, template: str) -> LLMChain:
        """
        Create a simple LLM chain with a prompt template
        
        Args:
            template: Prompt template string with variables in {brackets}
        
        Returns:
            LLMChain instance
        """
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        logger.info("Created simple chain")
        return chain
    
    def create_conversation_chain(
        self,
        system_prompt: str = "You are a helpful AI assistant.",
        use_summary_memory: bool = False
    ) -> ConversationChain:
        """
        Create a conversation chain with memory
        
        Args:
            system_prompt: System message for the conversation
            use_summary_memory: Whether to use summary memory (for long conversations)
        
        Returns:
            ConversationChain instance
        """
        # Set up memory
        if use_summary_memory:
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                return_messages=True
            )
        else:
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="history"
            )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{input}")
        ])
        
        # Create conversation chain
        chain = ConversationChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=True
        )
        
        logger.info(f"Created conversation chain with {'summary' if use_summary_memory else 'buffer'} memory")
        return chain
    
    def create_sequential_chain(self, chain_configs: List[Dict]) -> Any:
        """
        Create a sequential chain of prompts
        
        Args:
            chain_configs: List of chain configurations
                Each config should have 'name', 'template', and 'output_key'
        
        Returns:
            Sequential chain
        """
        from langchain.chains import SequentialChain
        
        chains = []
        input_variables = set()
        output_variables = []
        
        for config in chain_configs:
            template = config['template']
            prompt = PromptTemplate.from_template(template)
            
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_key=config['output_key']
            )
            chains.append(chain)
            
            # Track variables
            input_variables.update(prompt.input_variables)
            output_variables.append(config['output_key'])
        
        # Remove output variables from input variables (they're intermediate)
        input_variables -= set(output_variables[:-1])
        
        sequential_chain = SequentialChain(
            chains=chains,
            input_variables=list(input_variables),
            output_variables=output_variables,
            verbose=True
        )
        
        logger.info(f"Created sequential chain with {len(chains)} steps")
        return sequential_chain


# Example chain configurations
def create_analysis_chain(langchain_setup: AzureLangChainSetup):
    """Create an example analysis chain"""
    chain_configs = [
        {
            'name': 'summarizer',
            'template': "Summarize the following text in 2-3 sentences:\n\n{text}",
            'output_key': 'summary'
        },
        {
            'name': 'key_points',
            'template': "Extract 3 key points from this summary:\n\n{summary}",
            'output_key': 'key_points'
        },
        {
            'name': 'action_items',
            'template': "Based on these key points, suggest 2 action items:\n\n{key_points}",
            'output_key': 'action_items'
        }
    ]
    
    return langchain_setup.create_sequential_chain(chain_configs)


def create_qa_chain(langchain_setup: AzureLangChainSetup):
    """Create a question-answering chain with context"""
    template = """Answer the question based on the context below. If the question cannot be answered using the context, say "I don't have enough information to answer that."

Context: {context}

Question: {question}

Answer: """
    
    return langchain_setup.create_simple_chain(template)


def create_creative_chain(langchain_setup: AzureLangChainSetup):
    """Create a creative writing chain"""
    chain_configs = [
        {
            'name': 'brainstorm',
            'template': "Brainstorm 3 creative ideas for a story about: {topic}",
            'output_key': 'ideas'
        },
        {
            'name': 'select',
            'template': "From these ideas, select the most interesting one and explain why:\n\n{ideas}",
            'output_key': 'selected_idea'
        },
        {
            'name': 'outline',
            'template': "Create a brief story outline for this idea:\n\n{selected_idea}",
            'output_key': 'outline'
        },
        {
            'name': 'opening',
            'template': "Write an engaging opening paragraph for this story:\n\n{outline}",
            'output_key': 'opening'
        }
    ]
    
    return langchain_setup.create_sequential_chain(chain_configs)


if __name__ == "__main__":
    # Example usage
    setup = AzureLangChainSetup()
    
    # Test simple chain
    simple_chain = setup.create_simple_chain(
        "Tell me a fun fact about {topic}"
    )
    result = simple_chain.invoke({"topic": "artificial intelligence"})
    print(f"Simple chain result: {result['text']}\n")
    
    # Test conversation chain
    conv_chain = setup.create_conversation_chain()
    response1 = conv_chain.invoke({"input": "Hi! My name is Alex."})
    print(f"Conversation 1: {response1['response']}\n")
    
    response2 = conv_chain.invoke({"input": "What's my name?"})
    print(f"Conversation 2: {response2['response']}\n")
    
    # Test sequential chain
    analysis_chain = create_analysis_chain(setup)
    text = """
    Artificial Intelligence is rapidly transforming industries across the globe. 
    From healthcare to finance, AI systems are improving efficiency and enabling 
    new capabilities. However, this also raises important questions about ethics, 
    job displacement, and the need for regulation.
    """
    result = analysis_chain.invoke({"text": text})
    print(f"Analysis result:\n{result['action_items']}")
