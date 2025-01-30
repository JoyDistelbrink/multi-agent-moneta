import os
import asyncio
import nest_asyncio
# import pandas as pd
# modin allows to parallelise pandas operations. Specifically usefull for function callouts
# cutting down 5 min to 2 min
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.agents.strategies import KernelFunctionSelectionStrategy, KernelFunctionTerminationStrategy

from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions import KernelPlugin, KernelFunctionFromPrompt, KernelFunctionFromMethod
from crm_store import CRMStore
from product_store import ProductStore

load_dotenv(override=True)
nest_asyncio.apply()

#
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
API_VERSION = "2024-06-01"

API_BASE = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}"

print(AZURE_OPENAI_DEPLOYMENT_NAME)

## Configuring python adapters to Product (AI Search) and CRM (Cosmos DB)
credential = DefaultAzureCredential()

crm = CRMStore(
        url=os.getenv("COSMOSDB_ENDPOINT"),
        credential=credential,
        database_name=os.getenv("COSMOSDB_DATABASE_NAME"),
        container_name=os.getenv("COSMOSDB_CONTAINER_CLIENT_NAME")
    )

product = ProductStore()

gpt4o_service = AzureChatCompletion(service_id="gpt-4o", 
                        api_key=os.getenv("AZURE_OPENAI_KEY"), 
                        api_version=API_VERSION, 
                        base_url=API_BASE, 
                        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME)


## Creating the kernel and including the plugins for CRM and Product
agent_kernel = Kernel(
        services=[gpt4o_service],
        plugins=[
                KernelPlugin.from_object(plugin_instance=crm, plugin_name="crm"),
                KernelPlugin.from_object(plugin_instance=product, plugin_name="product"),
                ]
)

# QUERY agent
agent_query = ChatCompletionAgent(
        service_id="gpt-4o",
        kernel=agent_kernel,
        name="QUERY",
        execution_settings=AzureChatPromptExecutionSettings(
                # max_tokens=800,
                temperature=0.7,
                # top_p=0.5,
                # seed=113,
                function_choice_behavior = FunctionChoiceBehavior.Auto(filters={"included_plugins": ["crm", "product"]})
        ),
        description="""Agent processes the user query""",
        instructions="""
        You are an assistant that responds to the user query.
        
        **Your Task:**
        - FIRST carefully check if the customer name or client id is mentioned in the user request.
        - If the request contains client ID or customer name then use CRM functions to retrieve customer policy details.
        - DO NOT ask for the client's name or id. If you receive a request that references a customer without providing ID or Name, treat the question as generic and search information using 'search_products_terms_conditions' function.
        - Provide information about policies, prices, coverages, terms and conditions, etc. offered by using the provided function: 'search_products_terms_conditions'.
        - When using 'search_products_terms_conditions' function YOU MUST include all details that are relevant to the user's inquiry - such as locations, travel plans, etc.
        - When using 'search_products_terms_conditions' function YOU MUST include user question AS IS
        - Don't use your general knowledge to respond. Use only the provided functions.
        - Provide CONCISE and specific to answer the user's question. Do not provide general information.
        - Make sure to provide accurate and relevant information based on the user's inquiry.
""")


## PLANNER Agent
agent_planner = ChatCompletionAgent(
        service_id="gpt-4o",
        kernel=agent_kernel,
        name="PLANNER",
        execution_settings=AzureChatPromptExecutionSettings(
                temperature=0.5,
                # function_choice_behavior = FunctionChoiceBehavior.Auto(filters={"excluded_plugins": ["product", "crm"]})
        ),
        description="The agent when to terminate",
        
        # - If the response is clear and accurate, respond with a single word without explanation: 'terminate'.
        instructions=f"""
        You are an assistant that evaluates the responses.

        **YOUR TASK:**
        - Check if the user query has been correctly responded by the agents.
        - DO NOT ANSWER THE USER'S QUESTION RELYING ON YOUR KNOWLEDGE. REDIRECT TO THE APPROPRIATE AGENT.
        - If the last response is clear and accurate, respond with a single word without explanation: 'terminate'.
""")


class CompletionTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""
    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        if agent.name != "PLANNER":
            return False
        return "terminate" in history[-1].content.lower()
    
selection_function = KernelFunctionFromPrompt(
    function_name="selection",
    prompt_execution_settings=AzureChatPromptExecutionSettings(
        temperature=0,
        ),
    prompt=f"""
You are a speaker selector.

    - You MUST return the agent_id and nothing else.
    - You MUST return agent_id value from the list of available agents.
    - The names are case-sensitive and should not be abbreviated or changed.
    - YOU MUST OBSERVE AGENT USAGE INSTRUCTIONS.
    - ALWAYS CALL THE PLANNER AFTER THE QUERY
    - IF PLANNER does not terminate, call the QUERY agent again.

# AVAILABLE AGENTS

{{{{$agents}}}}

# CHAT HISTORY

{{{{$history}}}}
""",
)

    # - SELECT SUMMARISER AGENT ONLY IF REDIRECTED BY PLANNER
   

async def invoke_system(query):
    
    def sel_agent(result) -> str:
        # print (f"SELECTION RESULT: {result}")
        # print (f"SELECTION VALUE: {result.value}")
        # print (f"SELECTION VALUE 0: {result.value[0]}")
        res_val = result.value[0].content if result.value is not None else 'PLANNER'
        # print (f"SELECTION VALUE CONTENT: {res_val}")
        return res_val
 
    group_chat = AgentGroupChat(
        agents=[agent_query, agent_planner],
        selection_strategy=KernelFunctionSelectionStrategy(
                function=selection_function,
                kernel=agent_kernel,
                result_parser=lambda result: sel_agent(result),
                agent_variable_name="agents",
                history_variable_name="history",
            ),
        termination_strategy=CompletionTerminationStrategy(agents=[agent_planner], maximum_iterations=8)
    )

    await group_chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=query))
    
    # responses = list(reversed([item async for item in group_chat.invoke()]))
    async for item in group_chat.invoke():
        # print("---------------- >>>>>>  -------------------")
        # print(f"{item.name}: {item.content}")
        continue
        
    responses = list(reversed([item async for item in group_chat.get_chat_messages()]))
    
    if  "terminate" in responses[-1].content.lower():
        return { "name" : responses[-2].name, "content" : responses[-2].content, "count" : len(responses) } if len(responses) > 0 else None
    else:
        return { "name" : responses[-1].name, "content" : responses[-1].content, "count" : len(responses) } if len(responses) > 0 else None
        
def invoke(query):
    # print(f"QUERY: {query}")
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result = loop.run_until_complete(invoke_system(query))
    return result
