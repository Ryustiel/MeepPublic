"""
Plan : 
1. Lancer les tools + Summarization si besoin + Mémorisation si summarization + Activité & Wait
2. Appliquer le routage et utiliser les agents => Node de contrôle
3. Post processing : Déclencher l'exécution des Tool Calls pré autorisés + Template afterthought
"""

import json
from typing import Any, Dict, List, TypedDict, Union, AsyncIterator, Annotated
from langchain_core.messages import SystemMessage, AIMessageChunk

import graphs._data as data, graphs._formatting as formatting, graphs._agents as agents

async def agent_conversation(history: data.History, activity: str) -> AsyncIterator[Union[str, data.InternalUpdates]]:
    
    updates = data.InternalUpdates()
    current_channel = history.get_current_channel()

    prompt = f"""
    Tu t'appelles Meep.
    Tu discutes dans le channel {current_channel.name}.
    Tu peux voir les liens et images grace à des indications entre [].
    """

    if "discord" in current_channel.name.lower():
        prompt += """
        \nTu peux créer un lien vers le message d'un utilisateur (mentionner)
        en réécrivant le début de son message encapsulé par ¤¤ au début de ta réponse.
        Par exemple "¤le code est 12¤tu as donné le code ici"
        """

    if activity not in agents.AGENTS:
        raise ValueError(f"Unknown activity: {activity}")
    
    agent_metadata = agents.AGENTS[activity]
    langchain_llm = agent_metadata.llm
    if agent_metadata.prompt:
        prompt += f"\n{agent_metadata.prompt}"
        
    messages = formatting.formatted_conversation(history)

    gen: AsyncIterator[AIMessageChunk] = langchain_llm.astream(
        [SystemMessage(content=prompt),]
        + messages,  # Get messages from the current channel
    )
    
    buffer: AIMessageChunk = None
    reference_buffer: AIMessageChunk = None  # Store ¤¤ references
    
    async for msg in gen:
        
        if "¤" in msg.content:

            # Keep tracking the token counts
            if buffer is None:
                buffer = msg
            else:
                buffer += msg

            msg = msg.model_copy(deep=True)
            components = msg.content.split("¤")
            msg.content = components[-1] if components else ""

            if reference_buffer is None:
                reference_buffer = msg
            else:
                # Look for the referenced message and send the message id if any
                reference_buffer += msg
                reference_identifier = reference_buffer.content.strip().lower()
                reference_message_id = None
                for history_message in current_channel.messages:
                    if isinstance(history_message, data.HumanMessage_) and history_message.content.strip().lower().startswith(reference_identifier):
                        reference_message_id = history_message.message_id
                        break
                if reference_message_id:
                    yield "#reference#" + str(reference_message_id)
                reference_buffer = None  # Reset the reference buffer after sending
            
        else:
        
            if buffer is None:
                buffer = msg
            else:
                buffer += msg
                
            if reference_buffer is None:
                yield msg.content
            else:
                reference_buffer += msg

    if buffer:
        yield "#send#"
        
        for tool_call in buffer.tool_calls:
            yield "#tool#" + json.dumps(tool_call)

        # Create an internal update for adding the new AIMessage_
        updates = data.InternalUpdates(
            channel_updates={
                current_channel.id: data.InternalChannelUpdates(
                    new_messages=[
                        data.AIMessage_.from_message(buffer)
                    ]
                )
            }
        )
        
    yield updates
