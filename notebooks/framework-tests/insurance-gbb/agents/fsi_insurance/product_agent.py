# CRM Agent 
from pydantic import BaseModel, Field
import json
import os
import logging
from genai_vanilla_agents.agent import Agent
from agents.fsi_insurance.config import llm
from typing import List, Annotated, Optional
import requests
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

product_agent = Agent(  
    id="Product",
    system_message="""You are an assistant that searches general information about insurance products.

        **Your Responsibilities:**
        - **Handle all user requests that do NOT include a client's name.**
        - Provide information about policies, prices, coverages, terms and conditions, etc. offered by using the provided function: 'search_product'.
        - Offer clear and helpful answers to the user's inquiries. Don't use your general knowledge to respond but only the provided function.
        - If you are unsure ask the planner agent to clarify the user inquiry.
        
    """,  
    llm=llm,  
    description="""Call this Agent if:
        - You need to retrieve generic policies details, terms and conditions or other offering related information.
        DO NOT CALL THIS AGENT IF:  
        - You need to search for specific client's data identified by a client name or id in the request""",
    )  


def search(query: str):
    
    service_endpoint = os.getenv('AI_SEARCH_ENDPOINT')
    index_name = os.getenv('AI_SEARCH_INS_INDEX_NAME')
    key = os.environ["AI_SEARCH_KEY"]
    semantic_configuration_name = os.getenv('AI_SEARCH_INS_SEMANTIC_CONFIGURATION')
    
    logging.debug("Searching for: %s", query)

    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    
    text_vector_query = VectorizableTextQuery(
        kind="text",
        text=query,
        fields="text_vector"
    )

    # Perform the search
    results = search_client.search(
        search_text=query,
        include_total_count=True,
        vector_queries=[text_vector_query],
        query_type="semantic",
        semantic_configuration_name=semantic_configuration_name,
        query_answer="extractive",
        top=3,
        query_answer_count=3
    )

    response = list(results)

    output = []
    for result in response:
        result.pop("parent_id")
        result.pop("chunk_id")
        result.pop("text_vector")
        output.append(result)

    return output


# def search2(query: str):
#     service_endpoint = os.getenv('AI_SEARCH_ENDPOINT')
#     index_name = os.getenv('AI_SEARCH_INS_INDEX_NAME')
#     key = os.environ["AI_SEARCH_KEY"]

#     logging.info(f"---------------------- Searching for: {query}")

#     search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    
# #     {
# #   "search": "area 1, 2, 3 countries",
# #   "count": true,
# #   "vectorQueries": [
# #     {
# #       "kind": "text",
# #       "text": "area 1, 2, 3 countries",
# #       "fields": "text_vector"
# #     }
# #   ],
# #   "queryType": "semantic",
# #   "semanticConfiguration": "insurance-semantic-configuration",
# #   "captions": "extractive",
# #   "answers": "extractive|count-3",
# #   "queryLanguage": "en-us"
# # }
#     payload = json.dumps(
#         {
#             "search": query,
#             "count": True,
#             "vectorQueries": [
#                 {
#                 "kind": "text",
#                 "text": query,
#                 "fields": "text_vector"
#                 }
#             ],
#             "queryType": "semantic",
#             # "semanticConfiguration": os.getenv('AI_SEARCH_INS_SEMANTIC_CONFIGURATION'),
#             "semanticConfiguration": "insurance-semantic-configuration",
#             "captions": "extractive",
#             "answers": "extractive|count-3",
#             "queryLanguage": "en-us"
#         }
#     )

#     response = list(search_client.search(payload))

#     output = []
#     for result in response:
#         result.pop("parent_id")
#         result.pop("chunk_id")
#         result.pop("text_vector")
#         output.append(result)
#     logging.info(f"---------------------- ")    
#     logging.info(f"---------------------- ")    
#     logging.info(f"---------------------- Search results: {output}")    

#     return output
    
    
@product_agent.register_tool(description="Search product policies, terms, conditions")
def search_product(query:Annotated[str,"The query to search for"]) -> str:
    """
    Search general insurance product information regarding policies, coverages and terms and conditions by permorming a POST request to an Azure AI Search using the specified search body.

    Parameters:
    query

    Returns:
    dict: The search results in JSON format.
    """
    return search(query)