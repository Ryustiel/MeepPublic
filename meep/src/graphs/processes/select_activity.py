"""
Décide si l'agent doit répondre ou non, 
Sélectionne quel agent doit répondre.
Emet des recommandations pour l'agent (attitude, potentiels, erreurs à ne pas commettre).
"""

from typing import AsyncIterator, Literal, Optional, Union

import os, datetime, pydantic, langchain_core.messages
import constants, graphs._data as data, graphs._llm as llm, graphs._formatting as formatting, graphs._agents as agents_module

# ========================================================= ROUTING ACTIVITY SYSTEM ==============================================================

class WaitingRoutingResponse(pydantic.BaseModel):
    decision: Literal["check", "take", "skip"]

waiting_routing_llm = llm.DECISION.with_structured_output(WaitingRoutingResponse)

class AgentRoutingResponse(pydantic.BaseModel):
    special_agent: Optional[str] = None

agent_routing_llm = llm.DECISION.with_structured_output(AgentRoutingResponse)

class AgentWaitRoutingResponse(pydantic.BaseModel):
    decision: Literal["check", "take", "skip"]
    special_agent: Optional[str] = None

agent_wait_routing_llm = llm.DECISION.with_structured_output(AgentWaitRoutingResponse)


SPECIAL_AGENT_PROMPT = "".join(
    [
        "\nSi les utilisateurs récents ont des demandes spécifiques, vous pouvez sélectionner l'un des agents spéciaux suivants :"
    ] + [
        f"\n- '{name}': {metadata.routing_description}" 
        for name, metadata in agents_module.AGENTS.items() 
        if metadata.include
    ]
    + [
        "\nSi rien ne s'applique, vous pouvez ne choisir aucun agent spécial."
    ]
)

WAITING_PROMPT = (
    "Choisis \"skip\" pour passer ton tour de conversation."
    "\n\"check\" si tu veux parler mais n'es pas sûr que le timing soit bon. "
    "(par exemple si l'autre personne n'a peut être pas fini d'écrire un enchaînement de messages)"
    "\n\"talk\" pour commencer à parler immédiatement."
)


async def activity_and_waiting_systems(history: data.History, activity: str) -> AsyncIterator[Union[str, dict]]:
    """
    Handle dynamic state switching. Activity represents the state of the LLM.
    """
    
    current_channel = history.get_current_channel()
    messages = formatting.formatted_conversation(
        history=history,
        from_time_ago=datetime.timedelta(minutes=30),
        min_message=2,
        max_message=6,
    )

    waiting_choice = None
    if activity == agents_module.WAITING:

        prompt = WAITING_PROMPT
        
        if current_channel.channel_type == "public":
            prompt += (
                "Tu es Meep. Passe ton tour si les gens ne te parlent pas (se parlent entre eux) ou ne le souhaitent pas."
            )
        else:
            prompt += (
                "Tu es Meep. Il n'y a que toi et un autre utilisateur dans cette conversation. "
                "Passe ton tour si tu n'as pas besoin de répondre aux messages de l'utilisateur. (par exemple, quand il dit merci)"
            )

        waiting_choice: WaitingRoutingResponse = await waiting_routing_llm.ainvoke(
            [langchain_core.messages.SystemMessage(content=prompt)]
            + messages
        )
        
        match waiting_choice.decision:
            
            case "skip":
                yield {"activity": agents_module.WAITING}

            case "check":
                yield "#wait#5"
                yield {"activity": agents_module.WAITING}

            case "take":
                
                yield "#typing#"
                
                response: AgentRoutingResponse = await agent_routing_llm.ainvoke(
                    [langchain_core.messages.SystemMessage(content=SPECIAL_AGENT_PROMPT)]
                    + messages
                )

                if response.special_agent is None:
                    agent_choice = agents_module.DEFAULT_AGENT
                else:
                    agent_choice = response.special_agent
                    
                if agent_choice != activity:
                    yield "#activity#" + agent_choice

                yield {"activity": agent_choice}

    else:
        
        yield "#typing#"
        
        prompt = WAITING_PROMPT + SPECIAL_AGENT_PROMPT

        if current_channel.channel_type == "public":
            prompt += (
                "\nVous êtes Meep. Passez votre tour si les gens ne vous parlent pas ou ne le souhaitent pas."
            )
        else:
            prompt += (
                "\nVous êtes Meep. Il n'y a que vous et un autre utilisateur dans cette conversation. "
                "\nPassez votre tour si vous n'avez pas besoin de répondre aux messages de l'utilisateur. (par exemple, quand il dit merci)"
            )
            
        response: AgentWaitRoutingResponse = await agent_wait_routing_llm.ainvoke(
            [langchain_core.messages.SystemMessage(content=prompt)]
            + messages
        )

        if response.decision == "skip":
            yield "#activity#" + agents_module.WAITING
            yield {"activity": agents_module.WAITING}

        elif response.decision == "check":
            yield "#wait#5"
            yield {"activity": agents_module.WAITING}

        elif response.decision == "take":
            
            if response.special_agent is None:
                agent_choice = agents_module.DEFAULT_AGENT
            else:
                agent_choice = response.special_agent
                
            if agent_choice != activity:
                yield "#activity#" + agent_choice

            yield {"activity": agent_choice}
            