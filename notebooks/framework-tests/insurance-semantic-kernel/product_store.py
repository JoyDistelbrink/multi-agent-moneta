from pydantic import BaseModel, Field
import json
import os
from typing import List, Annotated, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from semantic_kernel.functions import kernel_function

class ProductStore:
    def __init__(self):
        return
        
    @kernel_function(
        name="search_products_terms_conditions", 
        description="Search product policies, terms, conditions"
    )
    def search(self, query: Annotated[str,"The query to search for"]) ->  Annotated[str, "The output in JSON format"]:
    
        service_endpoint = os.getenv('AI_SEARCH_ENDPOINT')
        index_name = os.getenv('AI_SEARCH_INS_INDEX_NAME')
        key = os.environ["AI_SEARCH_KEY"]
        semantic_configuration_name = os.getenv('AI_SEARCH_INS_SEMANTIC_CONFIGURATION')
        
        # print(f"PRODUCT: {query}")

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

        return json.dumps(output)