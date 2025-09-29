"""
Plan : 
1. Lancer les tools + Summarization si besoin + Mémorisation si summarization + Activité & Wait
2. Appliquer le routage et utiliser les agents => Node de contrôle
3. Post processing : Déclencher l'exécution des Tool Calls pré autorisés + Template afterthought
"""

import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union, AsyncIterator, Annotated

import os, httpx, re, sqlite3, datetime, langgraph.graph.state, langgraph.checkpoint.sqlite as langgraph_sqlite
import constants, graphs._data as data, graphs._mcp as mcp, graphs._agents as agents_module
from locallibs.langrouf.graph import Command, GraphBuilder

import graphs.processes.agentic_conversation, graphs.processes.select_activity, graphs.processes.summarize, graphs.processes.vision

class State(TypedDict, total=False):

    activity: Annotated[str, lambda _, x: x or data.DEFAULT_ACTIVITY]
    history: Annotated[data.History, data.history_reducer]
    wakeup: Annotated[Optional[data.WakeUp], lambda _, x: data.WakeUp.model_validate(x) if x else None]  # List of wake up requests to process.

    last_summary_check: Annotated[Optional[datetime.datetime], lambda _, x: x]  # Last the summarize node was run, to identify channels that wern't changed.
    
    internal_updates: Annotated[data.InternalUpdates, data.internal_updates_reducer]  # For internal processes. Set to None to reset.    
    internal_activity: Annotated[str, lambda _, x: x or "regular"]  # Dictate which nodes to run in parallel.

conn = sqlite3.connect("./data/checkpoints/meep.sqlite", check_same_thread=False)



# =========================== Chat SubGraph ===========================



CHATG = GraphBuilder("entrypoint", state=State)

@CHATG.node(next=["tools", "activity", "vision", "knowledge"])
async def entrypoint(s: State):
    
    internal_activity = "regular"

    if s["activity"] == agents_module.WAITING:
        internal_activity = "idle"  # Run activity first to know whether to keep waiting or not.
    
    channel = s["history"].get_current_channel()
    if channel.messages:
        # Detect if the earliest message is only a link.
        match = re.search(r'https?://\S+', channel.messages[-1].content)

        if match:
            extracted_link = match.group(0)
            non_link_char_count = len(channel.messages[-1].content) - len(extracted_link)

            if non_link_char_count < 5:  # Mostly a link in the message.
                internal_activity = "vision first"  # Run vision first, then activity and the other nodes together.

    match internal_activity:
        case "regular":
            yield Command(goto=("tools", "activity", "vision", "knowledge"), update={"internal_activity": internal_activity})
        case "vision first":
            yield Command(goto="vision", update={"internal_activity": internal_activity})
        case "idle":
            yield Command(goto="activity", update={"internal_activity": internal_activity})
        case _:
            raise ValueError(f"Unknown internal activity: {internal_activity}")

# ---- Actual processing nodes ----

@CHATG.node(next=["local_merge", "vision", "knowledge", "tools"])
async def activity(s: State):
    
    # This is only necessary because the node is not necessarily terminal (before local_merge).
    match s["internal_activity"]:
        case "regular":
            goto = ("local_merge")
        case "idle":  # TODO : Also make an "idle vision first" state.
            if s["activity"] == agents_module.WAITING:
                goto = ("local_merge")
            else:
                goto = ("vision", "knowledge", "tools")
        case "vision first":
            goto = ("local_merge")
        case _:
            raise ValueError(f"Unknown internal activity: {s['internal_activity']}")
    
    gen = graphs.processes.select_activity.activity_and_waiting_systems(s["history"], s["activity"])
    has_terminated = False
    async for output in gen:

        if isinstance(output, dict):
            yield Command(goto=goto, update=output)
            has_terminated = True
            break

        yield output

    if not has_terminated:
        raise ValueError(f"Activity did not terminate properly, should never happen.")

# ---- Knowledge ----

@CHATG.node(next="local_merge")
async def knowledge(s: State):
    # TODO : Run this node in parallel if in a WAITING state
    # If in a waiting state, run this node AFTER the activity node if it has a positive result.
    # TODO : Check for relevant memory fragments and put them on display.
            
    yield Command(goto="local_merge")

# ---- Vision ----

@CHATG.node(next=["local_merge", "activity", "knowledge", "tools"])
async def vision(s: State):
    
    match s["internal_activity"]:
        case "regular":
            goto = ("local_merge")
        case "idle":
            goto = ("local_merge")
        case "vision first":
            goto = ("activity", "knowledge", "tools")  # XXX : Missing a state for idle + vision first.
        case _:
            raise ValueError(f"Unknown internal activity: {s['internal_activity']}")
    
    updates = await graphs.processes.vision.vision_process_current_channel(s["history"])
            
    if updates.is_empty():
        yield Command(goto=goto)
    else:
        yield Command(goto=goto, update={"internal_updates": updates})

# ---- Router ----

@CHATG.node(next="local_merge")
async def tools(s: State):
    """
    Schedule and check on the tool calls.
    """
    requests: List[mcp.MCPRequest] = []
    reactive_tool_calls, updates = s["history"].find_reactive_tool_calls()
    
    for tool_call, tool_message in reactive_tool_calls:

        if tool_message.internal_status == "confirmed":  # Found confirmed tool call    
            requests.append(mcp.MCPRequest(tool_call=tool_call))
            
    if requests:
        await mcp.MCP_CLIENT.add_requests("meep", requests, local_context={"history": s["history"]})
        
    # Check for responses from the MCP thread.
    # Introduce a small delay to allow requests to be processed if any.
    responses = await mcp.MCP_CLIENT.get_responses("meep", timeout=constants.QUICK_RESPONSE_TIME)
    
    if responses:
        updates = data.internal_updates_reducer(
            updates,
            s["history"].generate_updates_from_mcp_responses(responses)
        )

    if not updates.is_empty():
        yield Command(
            goto="local_merge",
            update={"internal_updates": updates}
        )
    else:
        yield Command(goto="local_merge")  # No tool calls to process, just continue routing.

# ---- Agents ----

@CHATG.node(next=["agents"])
async def local_merge(s: State):
    """Merge nodes and goto the agents."""
    yield Command(goto="agents", update={"history": s["internal_updates"]})

@CHATG.node(next="postprocess")
async def agents(s: State):
    """Invoke custom LLM agent loops with their own tools and prompts depending on the activity state."""

    if s["activity"] == agents_module.WAITING:
        yield Command(goto="postprocess")  # Skip agents if waiting.
        agent = None
    else:
        agent = graphs.processes.agentic_conversation.agent_conversation(history=s["history"], activity=s["activity"])

        has_terminated = False
        async for output in agent:
            
            if isinstance(output, data.InternalUpdates):
                yield Command(goto="postprocess", update={"internal_updates": output})
                has_terminated = True
                break

            yield output

        if not has_terminated:
            raise ValueError("Agent did not terminate properly, should never happen.")

# ---- Post Processing ----

@CHATG.node(next=["__end__"])
async def postprocess(s: State):
    """
    Finalize processing and keep only essential state.
    """
    if not s["history"].is_empty():  # Clear history if not already empty.
        yield Command(goto="__end__")
    else:
        yield Command(
            goto="__end__",
            update={"history": "reset"}
        )

chat_graph = CHATG.compiled()



# =========================== Meep Graph ===========================



MEEPG = GraphBuilder(
    "preprocess", 
    state=State, 
    checkpointer=langgraph_sqlite.SqliteSaver.from_conn_string(conn)
)

@MEEPG.node(next=["chat", "summarize", "wakeup"])
async def preprocess(s: State):
    """
    1. Initialize the state if any value is missing.
    2. Check for the size of the conversation.
    Goto only summarize and memorize if the conversation is too long.
    If it's just slightly too long, go to all three nodes.
    If it's not too long, go to chat.
    """
    if "wakeup" in s and s["wakeup"]:  # Route to wakeup system
        yield Command(goto="wakeup")
        
    else:
        # 1. Schedule updates

        update = {
            "internal_updates": "reset", 
            "internal_activity": "regular", 
        }

        if "activity" not in s or s["activity"] == "":
            update["activity"] = data.DEFAULT_ACTIVITY  # Set default activity if not set
            
        yield Command(goto=("chat", "summarize"), update=update)

@MEEPG.node(next="__end__")
async def wakeup(s: State):
    """
    Wake up the system and prepare for the next tasks.
    """
    if not s.get("wakeup"):
        raise ValueError("Wakeup state is not set. Should never happen.")
    wakeup_event = s["wakeup"]
    if isinstance(wakeup_event, dict):
        wakeup_event = data.WakeUp.model_validate(wakeup_event)  # XXX : Exceptional conversion to pydantic, should be from __data

    recent_message_date = datetime.datetime.now() - datetime.timedelta(days=2)
    selected_channel = None
    skip_channel_date = datetime.datetime.now() - datetime.timedelta(days=2)

    # 1. Check if the event as an user. Find the last active channel for this user.
    if wakeup_event.user_name:
        for channel_id, channel in s["history"].channels.items():
            if channel.last_activity > skip_channel_date:
                continue
            # check if the channel has more recent
            for message in reversed(channel.messages):
                if (
                    isinstance(message, data.HumanMessage_) and 
                    message.author == wakeup_event.user_name and 
                    message.date > recent_message_date
                ):
                    recent_message_date = message.date
                    selected_channel = channel
                    break
            
        # TODO : Use "preferred channel" (DM) if needed (for certain tool calls).
    
    # 2. If no user were found, check if the wakeup event has a channel. Check if the channel has a callback url and an id.
    if selected_channel is None and wakeup_event.channel_id and wakeup_event.channel_id in s["history"].channels:
        selected_channel = s["history"].get_channel(wakeup_event.channel_id)

    # 3. If no channel was found, check if the current channel has a callbackurl and chat using it.
    if selected_channel is None:
        selected_channel = s["history"].get_current_channel()

    if (
        selected_channel 
        and selected_channel.wakeup_url  # Only do something if channel has wakeup url
        and selected_channel.last_activity < wakeup_event.unless_active_since  # Only wake up if channel was not active since the specified date
    ): 

        async with httpx.AsyncClient() as client:
            response = await client.get(selected_channel.wakeup_url)
            if response.status_code != 200: print("Failed to wake up channel", response.text)  # XXX : Ignore exceptions for now

    yield Command(goto="__end__", update={"wakeup": None})

# ---- Parallel tasks ----

@MEEPG.node(next="merge")
async def summarize(s: State):
    """
    Summarize segments of the conversation, and the summaries themselves.
    Delete messages that are too old.
    Identify when to summarize and when not to.
    NOTE : This is the only operation that may reduce the size of the conversation.
    """
    if "last_summary_check" not in s:
        s["last_summary_check"] = None
    
    updates = await graphs.processes.summarize.summarize_history(s["history"], s["last_summary_check"])
    
    for _, channel_update in updates.channel_updates.items():
        for summary in channel_update.new_summaries:
            yield f"**SUMMARY** <{summary.min_date} - {summary.max_date}> {summary.summary}"
            yield "#send#"
    
    yield Command(
        goto="merge", 
        update={
            "internal_updates": updates, 
            "last_summary_check": datetime.datetime.now()
        }
    )

MEEPG.subgraph_node(
    chat_graph, 
    name="chat",
    next="merge"
)

# ---- Finale combination ----

@MEEPG.node(next=("afterthought", "autotools"))
async def merge(s: State):
    """
    Apply all internal updates to the channels using the reducer.
    The history_reducer will handle the complex logic of updating messages and traces.
    """
    if s.get("internal_updates"):
        yield Command(
            goto=("afterthought", "autotools"),
            update={
                "history": s["internal_updates"],  # This will be processed by history_reducer to apply updates
                "internal_updates": "reset"  # Reset updates after applying them
            }
        )
    else:
        yield Command(goto=("afterthought", "autotools"))

@MEEPG.node(next="cleanup")
async def afterthought(s: State):
    """
    Post processing after the main conversation.
    """
    # NOTE : Could be a separate state field that would be processed on a rerun, so no need for a merge.
    yield Command(goto="cleanup")  # TODO : Update internal_updates instead of history

@MEEPG.node(next="cleanup")
async def autotools(s: State):
    """
    Schedule tools that are automatically triggered by the conversation.
    """
    requests: List[mcp.MCPRequest] = []
    reactive_tool_calls, updates = s["history"].find_reactive_tool_calls()

    for tool_call, tool_message in reactive_tool_calls:

        if (
            tool_message.internal_status == "unconfirmed"
            and "skip_confirmation" in tool_call["args"].keys()
            and tool_call["args"]["skip_confirmation"] == True
        ):   
            requests.append(mcp.MCPRequest(tool_call=tool_call))
            
    if requests:
        await mcp.MCP_CLIENT.add_requests("meep", requests)
        
    # Check for responses from the MCP thread.
    # Introduce a small delay to allow requests to be processed if any.
    responses = await mcp.MCP_CLIENT.get_responses("meep", timeout=constants.QUICK_RESPONSE_TIME)
    
    if responses:
        updates = data.internal_updates_reducer(
            updates,
            s["history"].generate_updates_from_mcp_responses(responses)
        )

        for response in responses:
            if response.status != "processing":  # At mwa
                # Command to schedule a rerun of the conversation to process the results.
                yield "#rerun#"
                break

    if not updates.is_empty():
        yield Command(
            goto="cleanup",
            update={"internal_updates": updates}
        )
    else:
        yield Command(goto="cleanup")  # No tool calls to process, just end the graph.

@MEEPG.node(next="__end__")
async def cleanup(s: State):
    """
    Clear InternalUpdates and manage temporary system messages with the lifespan system.
    """
    updates = s["internal_updates"]
    
    for channel_id, channel in s["history"].channels.items():
        if (
            channel.no_temporary_message_before is None or 
            channel.last_activity > channel.no_temporary_message_before
        ):
            channel_updates = data.InternalChannelUpdates()
            # NOTE : Setting to last_activity means ignoring the channel later if no new messages are received
            no_temporary_message_before_this_check = channel.last_activity
            
            # Process messages in this channel for lifespan countdown
            for i, message in enumerate(channel.messages):
                if isinstance(message, data.SystemMessage_) and message.lifespan is not None:
                    if message.lifespan > 1:
                        new_message = message.model_copy(deep=True)
                        new_message.lifespan -= 1  # Decrease lifespan
                        channel_updates.message_updates[i] = new_message
                    else:
                        channel_updates.message_deletes.append(i)  # Delete the message if lifespan is 1
                        
                    # Update the timestamp to avoid rechecking this message
                    no_temporary_message_before_this_check = min(no_temporary_message_before_this_check, message.date - datetime.timedelta(seconds=1))
            
            # Set the no_temporary_message_before timestamp for this channel
            channel_updates.no_temporary_message_before = no_temporary_message_before_this_check
            
            # Apply updates if there are any changes
            if not channel_updates.is_empty():
                local_update = data.InternalUpdates(channel_updates={channel_id: channel_updates})
                updates = data.internal_updates_reducer(updates, local_update)

    yield Command(
        goto="__end__", 
        update={"internal_updates": "reset", "history": updates})

graph = MEEPG.compiled()
