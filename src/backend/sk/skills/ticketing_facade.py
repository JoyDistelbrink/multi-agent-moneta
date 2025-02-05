import json
import os
import logging

from typing import Annotated
from semantic_kernel.functions import kernel_function

# from crm_store import CRMStore


class TicketingFacade:
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
        name="get_ticket_fields",
        description="Get the fields required to create a ticket",
    )
    def get_ticket_fields(
        self, ticket_type: Annotated[str, "The ticket type to search for"]
    ) -> Annotated[str, "The output is the fields required to create a ticket"]:
        print("get_ticket_fields")
        print("Ticket fields: name, email, description, server_name, server_ip")
        # response = self.crm_db.get_customer_profile_by_full_name(topic_name)
        # return json.dumps(response) if response else None
        return "The fields are: name, email, description, server_name, server_ip"

    @kernel_function(
        name="create_ticket",
        description="Create a ticket in the ticketting system",
    )
    def create_ticket(
        self,
        ticket_type: Annotated[str, "The ticket type to create"],
        ticket_data: Annotated[str, "The ticket data to create the ticket"],
    ) -> Annotated[str, "The output is the ticket id"]:
        print("create_ticket")
        print("Creating ticket:", ticket_type)
        print("Ticket data:", ticket_data)
        # response = self.crm_db.get_customer_profile_by_full_name(topic_name)
        # return json.dumps(response) if response else None
        return "14233987"

    # @kernel_function(
    #     name="load_from_crm_by_client_id",
    #     description="Load insured client data from the CRM from the client_id")
    # def get_customer_profile_by_client_id(self,
    #                                       client_id: Annotated[str,"The customer client_id to search for"]) -> Annotated[str, "The output is a customer profile"]:
    #     response = self.crm_db.get_customer_profile_by_client_id(client_id)
    #     return json.dumps(response) if response else None
