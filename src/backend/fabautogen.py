import asyncio
import numpy as np
import ast
import os
from typing import TYPE_CHECKING, List, Annotated
from autogen_core.models import ChatCompletionClient
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.search.documents.models import QueryType, VectorizedQuery
from azure.search.documents.aio import SearchClient
from azure.core.credentials import AzureKeyCredential
from typing import Sequence
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

load_dotenv()

config = {
    "provider": "AzureOpenAIChatCompletionClient",
    "config": {
        "azure_deployment": "gpt-4o",
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "model": "gpt-4o",
        "api_key": os.environ.get(
            "AZURE_OPENAI_API_KEY"
        ),  # TODO: Aad token based authentication
        "api_version": "2024-09-01-preview",
    },
}


def get_incident_categrory(user_issue: str) -> str:
    if "server" in user_issue.lower().split():
        return "Windows - Server - Reboot"
    elif "outlook" in user_issue.lower().split():
        return "Outlook Issue"
    else:
        return "Category could not be determined try again with a more detailed description"


def find_automation_schema(inc_category: str) -> str:
    if inc_category == "Windows - Server - Reboot":
        return """{'name': 'SNT:WIN:UBS:WINDOWSSERVERREBOOT', 'parameters': {'server_name': '', 'env':''}}"""
    else:
        return "No automation available"


def validate_automation_schema(inc_schema: str) -> str:
    dict_ = ast.literal_eval(inc_schema)
    list_p_info = []
    for p in dict_["parameters"]:
        if dict_["parameters"][p] == None or dict_["parameters"][p] == "":
            list_p_info.append(p)

    if len(list_p_info) > 0:
        return f"Could not validate schema of incident.... Please provide{', '.join(list_p_info)}"
    else:
        return f"Ticket Successfully validated. \n {str(dict_)}"


def compute_embedding(query: str) -> List[float]:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    embedding = client.embeddings.create(input=query, model="text-embedding-ada-002")
    return embedding.data[0].embedding


async def search_tool(
    query: Annotated[
        str,
        "String input representing the user's search intent. Designed for vectorization to support hybrid searching.",
    ],
    index_name: Annotated[
        str,
        "Specifies the target index. Options include: -gptacc-assist: Focuses on IT support topics within UBS, including hardware, software, business processes, and procurement. -gptacc-cloud: Dedicated to cloud services documentation, onboarding, and announcements. -gptacc-entities: Contains entities and terms from documentation, ideal for understanding specific acronyms or terms. -gptacc-sharepoint-sites: An index of documents from Sharepoint Sites.",
    ],
) -> str:
    if index_name == "gptacc-entities":
        index_name = "gptacc-cloud-entities"
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
        index_name=index_name,
        credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")),
    )

    query_vector = compute_embedding(query)
    vectorized = VectorizedQuery(
        vector=query_vector, k_nearest_neighbors=3, fields="vector"
    )
    results = await search_client.search(
        search_text=query,
        query_type=QueryType.SEMANTIC,
        vector_queries=[vectorized],
        filter=None,
        semantic_configuration_name="my-semantic-config",
        top=8,
    )
    results = [i async for i in results]
    results_string = "CONTEXT:\n"
    if index_name != "gptacc-cloud-entities":
        for result in results:
            results_string += (
                f"[{result['title']}]({result['url']}): {result['content_chunk']}\n"
            )

    else:
        for result in results:
            results_string += (
                f"{result['title']} ({result['type']}): {result['content_chunk']}\n"
            )

    await search_client.close()
    return results_string


def trigger_automation(automation: str) -> str:
    try:
        ast.literal_eval(automation)
    except Exception as e:
        print("provide a proper json")
    id_str = "INC" + "".join([str(np.random.randint(0, 10)) for _ in range(10)])
    return f"Submitted...Can be tracked under id: {id_str}"


def create_team() -> SelectorGroupChat:
    model_client = ChatCompletionClient.load_component(config)

    user_proxy = UserProxyAgent(name="User")

    planning_agent = AssistantAgent(
        "OrchestratorAgent",
        description="An agent for providing the final response to the user and responsible for planning the strategy to resolving a user's request, this agent should be the first to engage when given a new task and the last.",
        model_client=model_client,
        system_message="""
        You are a manager of the IT support agency, you follow a strict workflow with respect to user requests. You are the first line of defense against IT issues for the SRE Team at UBS. You can communicate with the user and other agents in your team. You always follow the agenda and delegate the tasks to your team or the user. You do not answer based on your knowledge.

        You have access to the following team:
        -   IssueCategorizerAgent
        -   SearchAgent
        -   IncidentAgent
        -   SubmissionAgent
        Here is the workflow you should follow:
        Whenever you receive a user request, you should follow the pattern below working with your team and the user. You start by analyzing the situation. It is always recommended to consult the categorizer agent to find UBS specific categories of user issues and additional hints. It is important to confirm this category with the user before moving on to a next step. Once the user confirms you can move on. From what you found out form the user and the categorizer, you have to make a decision, which has quite an impact, so reflect wisely.
        If the user confirmed a category that is a request and has a link available, you should pass that link directly to the user. If the category is an incident and there are no automations linked, you try to help the user by providing self-service solutions guiding him through the knowledge provided by the search agent. You NEVER answer based on your own knowledge and delegate the search to the Search Agent that can help you with condensed knowledge of UBS knowledge bases. When ever you provide knowledge back to the user, you should always consult the Search Agent and answer based on the response of the agent and use citation to be transparent. If your recommendations and guidance based on Search Results do not help the user, ask questions and try to refine your search a couple of times and repeat searching and guiding the user. Once you tried this 2-3 times and the user does not accept the solution, you should provide a summary of the information and initial category to the ticket creation agent. If the ticket creation agent asks for more information because it cannot validate the ticket, you should inquiry more information from the user – if the user also does not know and cannot help you, you should try searching for more information and nudge the user with targeted questions, information and guidance. Once you gathered the missing information you can pass back to the ticket creation agent. Once the the ticket creation agent provides you with a ticket schema in JSON format you can be sure that this ticket can be submitted by the Submission Agent. But before doing so always provide the ticket to the user and ask for confirmation to submit. If the user confirms provide the schema to the submission agent.
        Always delegate a task to one agent at a time by finishing your message with the respective keyword to speak to the agent.
        -   SearchAgent (SEARCH)
        -   IssueCategorizerAgent (CATEGORIZE)
        -   IncidentAgent (INCIDENT)
        -   SubmissionAgent (SUBMIT)
        -   User (BACK TO USER)
        It is recommended to remind yourself of the above agenda, and where you are in the process and reflect what would be the next best step before calling the next agent.
        """,
    )

    search_agent = AssistantAgent(
        "SearchAgent",
        description="A search agent that has access to UBS internal documentation. ",
        tools=[search_tool],
        model_client=model_client,
        system_message="""
        You are a search agent.
        Your only tool is search_tool - use it to find information.
        When searching try different queries to answer the question asked. Try to structure and summarize the information you return back to the user.
        """,
    )

    categorizer_agent = AssistantAgent(
        "IssueCategorizerAgent",
        description="A categorizer agent that has access to UBS internal incident categories. ",
        tools=[get_incident_categrory],
        model_client=model_client,
        system_message="""You are a agent categorizing user requests to map it with available categories of incidents at UBS.""",
    )

    submission_agent = AssistantAgent(
        "SubmissionAgent",
        description="A submission agent that can submit well formed JSON strucured issues to the central ticketing system.",
        tools=[trigger_automation],
        model_client=model_client,
    )

    incident_agent = AssistantAgent(
        "IncidentAgent",
        description="An Agent that can find ticket schemas and get ",
        model_client=model_client,
        tools=[find_automation_schema, validate_automation_schema],
        system_message="""
        You are a Incident Agent and your task is to take a category and description to check if a automation is available. If yes, the tool to find an automation will return a schema. From the provided context check if you can find the information and add it to the JSON.
        After that validate the schema. If validation fails return it back and ask for more information. If validation is successful provide the schema with the inputs you made.
        """,
    )

    # The termination condition is a combination of text mention termination and max message termination.
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=50)
    # termination = text_mention_termination | max_messages_termination
    termination = text_mention_termination

    # The selector function is a function that takes the current message thread of the group chat
    # and returns the next speaker's name. If None is returned, the LLM-based selection method will be used.
    def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
        if messages[-1].source == planning_agent.name and messages[-1].content.endswith(
            "BACK TO USER"
        ):
            return user_proxy.name

        elif messages[-1].source == categorizer_agent.name:
            return planning_agent.name
        elif messages[-1].source == search_agent.name:
            return planning_agent.name

        elif messages[-1].source == incident_agent.name:
            return planning_agent.name

        elif messages[-1].source == submission_agent.name:
            return planning_agent.name

        elif messages[-1].source == planning_agent.name and messages[
            -1
        ].content.endswith("CATEGORIZE"):
            return categorizer_agent.name

        elif messages[-1].source == planning_agent.name and messages[
            -1
        ].content.endswith("SUBMIT"):
            return submission_agent.name

        elif messages[-1].source == planning_agent.name and messages[
            -1
        ].content.endswith("SEARCH"):
            return search_agent.name
        elif messages[-1].source == planning_agent.name and messages[
            -1
        ].content.endswith("INCIDENT"):
            return incident_agent.name

        else:

            return None

    team = SelectorGroupChat(
        [
            user_proxy,
            planning_agent,
            search_agent,
            categorizer_agent,
            incident_agent,
            submission_agent,
        ],
        model_client=ChatCompletionClient.load_component(
            config
        ),  # TODO: Use a smaller model for the selector.
        termination_condition=termination,
        selector_func=selector_func,
    )
    return team


async def main() -> None:
    team = create_team()
    task = "My server is unaccessible"
    await Console(team.run_stream(task=task))


asyncio.run(main())
