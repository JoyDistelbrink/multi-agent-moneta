import logging
import os
import ast
import json
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.agents.strategies.termination.termination_strategy import (
    TerminationStrategy,
)
from semantic_kernel.agents.strategies import KernelFunctionSelectionStrategy
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions import (
    KernelPlugin,
    KernelFunctionFromPrompt,
    KernelFunctionFromMethod,
    kernel_function,
)

# from sk.skills.crm_facade import CRMFacade
from sk.skills.kb_facade import KBFacade
from sk.skills.ticketing_facade import TicketingFacade
from sk.skills.policies_facade import PoliciesFacade
from sk.orchestrators.semantic_orchestrator import SemanticOrchastrator


class TicketingMultiOrchestrator(SemanticOrchastrator):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Ticketing Multi Orchestrator init")

        kb = KBFacade()
        ticketing = TicketingFacade()

        # product = PoliciesFacade(
        #     credential=DefaultAzureCredential(),
        #     service_endpoint=os.getenv("AI_SEARCH_ENDPOINT"),
        #     index_name=os.getenv("AI_SEARCH_INS_INDEX_NAME"),
        #     semantic_configuration_name="default",
        # )

        self.kernel = Kernel(
            services=[self.gpt4o_service],
            plugins=[
                # KernelPlugin.from_object(plugin_instance=kb, plugin_name="kb"),
            ],
        )
        self.kernelKb = Kernel(
            services=[self.gpt4omini_service],
            plugins=[
                KernelPlugin.from_object(plugin_instance=kb, plugin_name="kb"),
                KernelPlugin.from_object(
                    plugin_instance=ticketing, plugin_name="ticketing"
                ),
            ],
        )

    # --------------------------------------------
    # Selection Strategy
    # --------------------------------------------
    def create_selection_strategy(self, agents, default_agent):
        """Speaker selection strategy for the agent group chat."""
        definitions = "\n".join(
            [f"{agent.name}: {agent.description}" for agent in agents]
        )

        @kernel_function(
            name="SpeakerSelectorLogic",
            description="Selects the next agent based on chat history",
        )
        def select_next_speaker(history: str) -> str:
            """
            Rule-based agent selection logic using purely Python code.
            """
            print(f"History: {history}")
            print(f"History type: {type(history)}")
            print(history)
            try:
                history_list = ast.literal_eval(history)
            except json.JSONDecodeError:
                # If history is not valid JSON, fallback to a default behavior
                # or raise an error. Here we just pick "PlannerAgent" by default.
                print(
                    "History could not be parsed as JSON, defaulting to PlannerAgent."
                )
                return "PlannerAgent"
            # Make sure we have at least one message
            if not history_list:
                print("History is empty or invalid, defaulting to PlannerAgent.")
                return "PlannerAgent"
            last_message = history_list[-1]
            print(f"Last message: {last_message}")
            # Check if the last message is from the user
            if last_message.get("role") == "user":
                # If it is from the user, direct to PlannerAgent
                return "PlannerAgent"
            # Check if the message is a back to user message
            if "USER" in last_message.get("content", "").upper():
                return "SummariserAgent"

            # Check if the message is from the subagents and go back to the PlannerAgent
            elif (
                last_message.get("name") == "SummariserAgent"
                or last_message.get("name") == "CategorizerAgent"
                or last_message.get("name") == "SubmissionAgent"
                or last_message.get("name") == "IncidentAgent"
                or last_message.get("name") == "SearchAgent"
            ):
                return "PlannerAgent"
            elif "CATEGORIZE" in last_message.get("content", "").upper():
                return "CategorizerAgent"
            elif "SUBMIT" in last_message.get("content", "").upper():
                return "SubmissionAgent"
            elif "INCIDENT" in last_message.get("content", "").upper():
                return "IncidentAgent"
            elif "SEARCH" in last_message.get("content", "").upper():
                return "SearchAgent"

            # Otherwise default to PlannerAgent
            return "PlannerAgent"

        selection_function = KernelFunctionFromMethod(
            plugin_name="SpeakerSelector", method=select_next_speaker
        )

        # Could be lambda. Keeping as function for clarity
        # def parse_selection_output(output):
        # print(f"Parsing selection: {output}")
        # self.logger.debug(f"Parsing selection: {output}")
        # if output.value is not None:
        #     return output.value[0].content
        # return default_agent.name
        def parse_selection_output(output):
            print(f"Parsing selection: {output}")
            if output.value and isinstance(output.value, str):
                return output.value
            return default_agent.name

        return KernelFunctionSelectionStrategy(
            kernel=self.kernel,
            function=selection_function,
            result_parser=parse_selection_output,
            agent_variable_name="agents",
            history_variable_name="history",
        )

    # --------------------------------------------
    # Termination Strategy
    # --------------------------------------------
    def create_termination_strategy(self, agents, final_agent, maximum_iterations):
        """
        Create a chat termination strategy that terminates when the final agent is reached.
        params:
            agents: List of agents to trigger termination evaluation
            final_agent: The agent that should trigger termination
            maximum_iterations: Maximum number of iterations before termination
        """

        class CompletionTerminationStrategy(TerminationStrategy):
            async def should_agent_terminate(self, agent, history):
                """Terminate if the last actor is the Responder Agent."""
                logging.getLogger(__name__).debug(history[-1])
                return agent.name == final_agent.name

        return CompletionTerminationStrategy(
            agents=agents, maximum_iterations=maximum_iterations
        )

    # --------------------------------------------
    # Create Agent Group Chat
    # --------------------------------------------
    def create_agent_group_chat(self):

        self.logger.debug("Creating ticketing help chat")

        searcher_agent = self.create_agent(
            service_id="gpt-4o-mini",
            kernel=self.kernelKb,
            definition_file_path="sk/agents/ticketingmulti/searcher.yaml",
        )
        categorizer_agent = self.create_agent(
            service_id="gpt-4o-mini",
            kernel=self.kernelKb,
            definition_file_path="sk/agents/ticketingmulti/categorizer.yaml",
        )
        submiter_agent = self.create_agent(
            service_id="gpt-4o-mini",
            kernel=self.kernelKb,
            definition_file_path="sk/agents/ticketingmulti/submiter.yaml",
        )
        incident_agent = self.create_agent(
            service_id="gpt-4o-mini",
            kernel=self.kernelKb,
            definition_file_path="sk/agents/ticketingmulti/incident.yaml",
        )
        planner_agent = self.create_agent(
            service_id="gpt-4o",
            kernel=self.kernel,
            definition_file_path="sk/agents/ticketingmulti/planner.yaml",
        )
        responder_agent = self.create_agent(
            service_id="gpt-4o",
            kernel=self.kernel,
            definition_file_path="sk/agents/ticketingmulti/responder.yaml",
        )

        agents = [
            searcher_agent,
            categorizer_agent,
            submiter_agent,
            responder_agent,
            planner_agent,
            incident_agent,
        ]

        agent_group_chat = AgentGroupChat(
            agents=agents,
            selection_strategy=self.create_selection_strategy(agents, planner_agent),
            termination_strategy=self.create_termination_strategy(
                agents=agents,
                final_agent=responder_agent,
                maximum_iterations=8,
            ),
        )

        return agent_group_chat
