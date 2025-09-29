"""
Génère des message résumés de conversation qui resteront enregistrés dans la state.
Ces messages sont par Channel et sont utilisés pour construire les system prompts 
et garder trace du contexte dans les autres channels.

Créer un summary global des channels qui n'ont pas été exploitées depuis longtemps 
pour le pas avoir à les afficher systématiquement.
"""

from typing import List, Literal, Optional, Tuple, Union
from langchain_core.messages import SystemMessage, AIMessageChunk, AIMessage

import os, datetime, langchain_openai, pydantic
import constants, graphs._data as data, graphs._mcp as mcp, graphs._formatting as formatting, graphs._llm as llm
from locallibs.langrouf.graph import Command, GraphBuilder

async def create_message_summary(content: str) -> str:
    desc: AIMessage = await llm.SUMMARIZE.ainvoke([
        SystemMessage(content=f"Reduce the size of this message to a small paragraph while keeping as much information and meaning as possible: {content}")
    ])
    return desc.content

class SummaryResponse(pydantic.BaseModel):
    summary: str

summary_llm = llm.THINK.with_structured_output(
        SummaryResponse,
    )


async def create_summary(
    current_channel_id: str, 
    history: data.History, 
    least_recent_message: datetime.datetime,
    most_recent_message: datetime.datetime
) -> data.Summary:
    summary_response: SummaryResponse = await summary_llm.ainvoke(
        [
            SystemMessage(content=(
                "Résume toute la conversation à la seconde personne (\"... a parlé de ... et tu as ...\"), "
                "en intégrant les éventuels résumés de messages déjà présents, "
                "mais à l'exception des messages qui commencent par \"from channel ...\", "
                "qui ne servent qu'à apporter du contexte "
                "sur ce qu'il se passe dans les autres channels."
            ))
        ] + formatting.formatted_conversation(
            current_channel_id=current_channel_id,
            history=history,
            max_date=most_recent_message,
            min_date=least_recent_message
        )
    )
    return data.Summary(
        summary=summary_response.summary,
        min_date=least_recent_message,
        max_date=most_recent_message
    )
    

# ========================================================= SUMMARIZE ============================


async def summarize_history(history: data.History, last_summary_check: Optional[datetime.datetime]) -> data.InternalUpdates:
    """
    Check among the chats that weren't checked recently if they are summarizable.
    """
    updates = data.InternalUpdates()

    for channel_id, channel in history.channels.items():
        # NOTE : If never summarized, do summarize. Otherwise, summarize if active since last check.
        if not last_summary_check or channel.last_activity > last_summary_check:
            
            # 1. Check regions to see if something is worth summarizing.

            groups = formatting.group_messages(
                channel.messages,
                time_gap_reference={
                    datetime.datetime.now(): datetime.timedelta(minutes=5),
                    datetime.datetime.now() -  datetime.timedelta(minutes=20): datetime.timedelta(minutes=15),
                    datetime.datetime.now() - datetime.timedelta(hours=1): datetime.timedelta(hours=1),
                    datetime.datetime.now() - datetime.timedelta(hours=6): datetime.timedelta(days=1),
                    datetime.datetime.now() - datetime.timedelta(days=2): datetime.timedelta(days=2),
                },
                max_group_size_reference={
                    datetime.datetime.now(): 4000,
                    datetime.datetime.now() - datetime.timedelta(hours=1): 8000,
                    datetime.datetime.now() - datetime.timedelta(days=1): 20000,
                }
            )
            regions: List[Tuple[datetime.datetime, datetime.datetime]] = []
            for group in groups:
                
                regions.append((
                    group[0].min_date if isinstance(group[0], data.Summary) else group[0].date,
                    group[-1].max_date if isinstance(group[-1], data.Summary) else group[-1].date
                ))
                
            # 2. Ignore regions that are too recent to be summarized, is not large enough, or has a summary
            if regions:
                ignore = [len(regions)-1]  # Most recent region won't be summarized
                
                for i, (group, region) in enumerate(zip(groups, regions)):

                    if i in ignore: continue

                    elif len(group) < 5:  # Check size of the region
                        ignore.append(i)
                        
                    elif (
                        region[1] in channel.summaries 
                        and any(
                            summary.min_date == region[0] 
                            for summary in channel.summaries[region[1]]
                        )
                    ):  # A summary already exists for this region.
                        ignore.append(i)

                for i in sorted(list(ignore), reverse=True):
                    groups.pop(i)
                    regions.pop(i)

            # 3. Create the summaries for the groups that remain
            new_summaries = []
            for group, (min_date, max_date) in zip(groups, regions):
                summary = await create_summary(channel_id, history, min_date, max_date)
                new_summaries.append(summary)
            
            if new_summaries:
                
                # 4. Delete messages past a certain date.
                
                updates.channel_updates[channel_id] = data.InternalChannelUpdates(
                    new_summaries=new_summaries,
                    delete_before=datetime.datetime.now() - datetime.timedelta(days=5)
                )
            # XXX : Static timedelay here sucks, at least guarantee to preserve the oldest summary.

    return updates
