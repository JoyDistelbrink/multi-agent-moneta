import json
import os
import logging

from typing import Annotated
from semantic_kernel.functions import kernel_function

# from crm_store import CRMStore


class KBFacade:
    """
    The class acts as an facade for the crm_store.
    The facade is only required if the same CRM Store to be used by both Vanilla and SK frameworks
    Once a single framwork is adopted it can be retired.
    """

    def __init__(self):
        # self.crm_db = CRMStore(
        #     url=cosmosdb_endpoint,
        #     key=key,
        #     database_name=crm_database_name,
        #     container_name=crm_container_name)
        self.kb_db = None

    @kernel_function(
        name="categorize_ticket",
        description="Use this function to categorize the issue based on the description of the issue",
    )
    def categorize_ticket(
        self, issue_content: Annotated[str, "The issue content to categorize"]
    ) -> Annotated[str, "The output is the category of the issue"]:
        print("categorize_ticket")
        print("Categorizing ticket: server")

        # response = self.crm_db.get_customer_profile_by_full_name(topic_name)
        # return json.dumps(response) if response else None
        return "server"

    @kernel_function(
        name="load_from_kb_by_topic",
        description="Load self service help data from the knowledge base by topic",
    )
    def load_from_kb_by_topic(
        self, topic_name: Annotated[str, "The topic to search for"]
    ) -> Annotated[str, "The output is a solution to the help issue"]:
        # response = self.crm_db.get_customer_profile_by_full_name(topic_name)
        # return json.dumps(response) if response else None
        return "To fix your help issue, please follow the steps: 1. Turn off the device. 2. Turn it back on. 3. If the issue persists, contact support."

    # @kernel_function(
    #     name="load_from_crm_by_client_id",
    #     description="Load insured client data from the CRM from the client_id")
    # def get_customer_profile_by_client_id(self,
    #                                       client_id: Annotated[str,"The customer client_id to search for"]) -> Annotated[str, "The output is a customer profile"]:
    #     response = self.crm_db.get_customer_profile_by_client_id(client_id)
    #     return json.dumps(response) if response else None
