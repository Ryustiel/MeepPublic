
from typing import Dict, Optional

import langchain_core.runnables
import graphs._mcp as mcp, graphs._llm as llm

class AgentMetadata:
    def __init__(
        self,
        routing_description: str,
        memory_description: str,
        llm: langchain_core.runnables.Runnable,
        prompt: Optional[str] = None,
        include: bool = True
    ):
        self.routing_description = routing_description
        self.memory_description = memory_description
        self.prompt = prompt
        self.llm = llm
        self.include = include

# TODO : Support custom message formatting parameters (min_date, min_message, ...)

AGENTS: Dict[str, AgentMetadata] = {
    "conversing": AgentMetadata(
        routing_description="Parle simplement.",
        memory_description="Agent par défaut.",
        llm=llm.CUSTOM.bind_tools(
            list(
                mcp.get_toolkit(
                    [
                        "setup_reminder",
                    ]
                ).tools.values()
            )
        ),
        include=False,
    ),
    "debug": AgentMetadata(
        routing_description="Peut exécuter les tools \"perform_action_number_...\" 1 à 6.",
        memory_description="Peut exécuter les tools \"perform_action_number_...\" 1 à 6.",
        llm=llm.CUSTOM.bind_tools(
            list(
                mcp.get_toolkit(
                    [
                        "perform_action_number_1", 
                        "perform_action_number_2", 
                        "perform_action_number_3", 
                        "perform_action_number_4",
                        "perform_action_number_5",
                        "perform_action_number_6",
                    ]
                ).tools.values()
            )
        )
    ),
    "generate_image": AgentMetadata(
        routing_description="Si l'utilisateur veut générer une image, cet agent peut le faire.",
        memory_description="Peut générer une image avec le tool \"generate_image\".",
        llm=llm.CUSTOM.bind_tools(
            list(
                mcp.get_toolkit(
                    [
                        "generate_image",
                    ]
                ).tools.values()
            )
        )
    )
}

WAITING = "waiting"

DEFAULT_AGENT = "conversing"

if DEFAULT_AGENT not in AGENTS:
    raise ValueError(f"Default agent not in AGENTS: {DEFAULT_AGENT}")
