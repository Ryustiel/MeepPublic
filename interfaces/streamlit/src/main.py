"""
Create a simple LLM chat using streamlit.
"""

from typing import Iterator, List, Dict, Literal, Optional, Any, Tuple

import os, streamlit, json, httpx, pydantic

class Message(pydantic.BaseModel):
    type: Literal["human", "ai"]
    content: str
    structured_content: Optional[Dict[str, Any]] = None
    
class ToolCallUpdate(pydantic.BaseModel):
    tool_call_id: str
    state: Literal["confirmed", "rejected", "canceled"]
    content: Optional[str] = None
    
class LocalHistory:
    def __init__(self):
        self.messages: List[Message] = []
        self.confirm_tool_calls: List[ToolCallUpdate] = []
        self.activity: str = "???"
        
        self.new_human_messages: List[str] = []
        self.new_tool_call_updates: List[ToolCallUpdate] = []
        
        self.rerun_immediately: bool = False

@streamlit.cache_resource
def retrieve_local_history() -> LocalHistory:
    return LocalHistory()

LOCAL_HISTORY = retrieve_local_history()
LANGGRAPH_SERVER = os.environ["LANGGRAPH_SERVER_URL"]
MEEP_THREAD = os.environ["MEEP_THREAD_ID"]

# =========================================== STREAMING RESPONSE

def stream_response(history: LocalHistory) -> Iterator[str]:
    
    token_buffer = None
    
    with httpx.Client(timeout=httpx.Timeout(5, read=60)) as client:

        with client.stream(
            "POST",
            LANGGRAPH_SERVER + f"/threads/{MEEP_THREAD}/runs/stream",
            json = {
                "assistant_id": "meep",
                "checkpoint": {"thread_id": MEEP_THREAD},
                "config": {"recursion_limit": 50},
                "stream_mode": ["custom"],
                "stream_subgraphs": True,
                "input": {
                    "history": {
                        "current_channel": "streamlitapp",
                        "tool_updates": [
                            {
                                "tool_call_id": update.tool_call_id, 
                                "internal_status": update.state,
                                "content": update.content if update.content else None
                            }
                            for update in LOCAL_HISTORY.new_tool_call_updates
                        ],
                        "channel_updates": {
                            "streamlitapp": {
                                "name": "Streamlit App",
                                "channel_type": "streamlit",
                                "new_messages": [
                                    {
                                        "type": "human", 
                                        "content": msg,
                                        "author": None
                                    } 
                                    for msg in LOCAL_HISTORY.new_human_messages
                                ],
                            }
                        }
                    }
                },
            },
            headers = {"Content-Type": "application/json"}
        ) as response:

            for line in response.iter_lines():
                
                if line and line.startswith("data:"):
                        
                    content = line[7:-1]

                    if (
                        content.startswith("#typing#") or
                        content.startswith("\"run_id\":\"")
                    ):
                        continue

                    elif content.startswith("#tool#"):
                        try:
                            unescaped_content = content[6:].replace("\\\"", "\"")
                            tool_call = json.loads(unescaped_content)
                            # Only add tool calls which do not have this special flag.
                            if "skip_confirmation" not in tool_call["args"] or not tool_call["args"]["skip_confirmation"]:
                                LOCAL_HISTORY.confirm_tool_calls.append(tool_call)
                            else:
                                args = tool_call['args']
                                del args['skip_confirmation']
                                streamlit.info(f"Running {tool_call['name']} with {args}")

                        except json.JSONDecodeError as e:
                            streamlit.error("Failed to parse tool call " + content[6:] + " - " + str(e))
                            
                        continue
                    
                    elif content.startswith("#activity#"):
                        activity = content[10:]
                        if activity == "waiting":
                            streamlit.warning("WAIT")
                        else:
                            streamlit.info("Switched to " + activity)
                        LOCAL_HISTORY.activity = activity
                        continue

                    elif content.startswith("#send#"):
                        content = "\n"
                        
                    elif content.startswith("#rerun#"):
                        LOCAL_HISTORY.rerun_immediately = True
                        continue

                    if content:
                        yield content

                        if token_buffer is None: 
                            token_buffer = content
                        else: 
                            token_buffer += content

    LOCAL_HISTORY.new_human_messages = []
    LOCAL_HISTORY.new_tool_call_updates = []
    if token_buffer and token_buffer.strip(): 
        history.messages.append(Message(type="ai", content=token_buffer))

# =========================================== PAGE

streamlit.set_page_config(layout="wide", page_title="Meep", page_icon=":book:")

streamlit.title("Meep Chat")

# =========================================== SIDEBAR

with streamlit.sidebar:
    streamlit.header("Thread Management")
    
    if streamlit.button("Create Thread"):
        response = httpx.post(
            LANGGRAPH_SERVER + "/threads", 
            json={"thread_id": MEEP_THREAD, "if_exists": "do_nothing"}
        )
        streamlit.info(response.text)
        
    if streamlit.button("Delete Thread"):
        response = httpx.delete(LANGGRAPH_SERVER + f"/threads/{MEEP_THREAD}")
        streamlit.info(f"Thread deleted: {response.status_code}")

# =========================================== CHAT WIDGET

chat_widget = streamlit.container(border=False)
with chat_widget:
        
    chat_container = streamlit.container(height=590)
    with chat_container:

        for message in LOCAL_HISTORY.messages:

            with streamlit.chat_message(message.type):
                if message.structured_content: streamlit.json(message.structured_content)
                streamlit.markdown(message.content)

    prompt = streamlit.chat_input("Let's chat mowo")

    if prompt or (LOCAL_HISTORY.rerun_immediately and not LOCAL_HISTORY.confirm_tool_calls):
        
        with chat_container:

            # Only cancel tool calls if there's a new prompt (user typed something)
            if prompt and LOCAL_HISTORY.confirm_tool_calls:

                for tool_call in LOCAL_HISTORY.confirm_tool_calls:
                    
                    message = Message(
                        type="ai",
                        content="**CANCELLED**",
                        structured_content=tool_call["args"]
                    )
                    LOCAL_HISTORY.messages.append(message)
                    LOCAL_HISTORY.new_tool_call_updates.append(
                        ToolCallUpdate(
                            tool_call_id=tool_call["id"], 
                            state="canceled",
                            content="User typed a new message, cancelling the tool call."
                        )
                    )

                    with streamlit.chat_message(message.type):  # Immediately display the message
                        if message.structured_content: streamlit.json(message.structured_content)
                        streamlit.markdown(message.content)

                LOCAL_HISTORY.confirm_tool_calls = []

            if LOCAL_HISTORY.rerun_immediately:
                LOCAL_HISTORY.rerun_immediately = False
                
            elif prompt:
                # Add the new human message to the history
                LOCAL_HISTORY.messages.append(Message(type="human", content=prompt))
                LOCAL_HISTORY.new_human_messages.append(prompt)
                
                with streamlit.chat_message("human"):  # Display the user's message
                    streamlit.markdown(prompt)
            
            with streamlit.chat_message("ai"):
                
                confirm_placeholder = streamlit.empty()
                confirm_placeholder.empty()
                
                with confirm_placeholder.container():
                        
                    with streamlit.spinner("Processing..."):    
                        streamlit.write_stream(stream_response(LOCAL_HISTORY))

                    if LOCAL_HISTORY.confirm_tool_calls:

                        streamlit.subheader("Confirm action : **" + LOCAL_HISTORY.confirm_tool_calls[0]["name"] + "**")
                        streamlit.write(LOCAL_HISTORY.confirm_tool_calls[0]["args"])
                        streamlit.button("Confirm", key="tool_confirm_button"+str(LOCAL_HISTORY.confirm_tool_calls[0]["id"]))
                        streamlit.button("Reject", key="tool_reject_button"+str(LOCAL_HISTORY.confirm_tool_calls[0]["id"]))

                    elif LOCAL_HISTORY.rerun_immediately:  # XXX : Will display tool any confirm first
                        streamlit.rerun()

    elif LOCAL_HISTORY.confirm_tool_calls:

        tool_call = LOCAL_HISTORY.confirm_tool_calls[0]

        if streamlit.session_state.get("tool_confirm_button"+str(tool_call["id"])):

            LOCAL_HISTORY.new_tool_call_updates.append(
                ToolCallUpdate(
                    tool_call_id=tool_call["id"], 
                    state="confirmed"
                )
            )
            LOCAL_HISTORY.messages.append(
                Message(
                    type="ai", 
                    content="**CONFIRMED**", 
                    structured_content=tool_call["args"]
                )
            )
            LOCAL_HISTORY.confirm_tool_calls.pop(0)
            
            # Register rerun if all the pending tool calls were completed
            if not LOCAL_HISTORY.confirm_tool_calls:
                LOCAL_HISTORY.rerun_immediately = True

            streamlit.rerun()

        elif streamlit.session_state.get("tool_reject_button"+str(tool_call["id"])):

            LOCAL_HISTORY.new_tool_call_updates.append(
                ToolCallUpdate(
                    tool_call_id=tool_call["id"], 
                    state="rejected",
                    content="User clicked the reject button"
                )
            )
            LOCAL_HISTORY.messages.append(
                Message(
                    type="ai", 
                    content="**REJECTED**", 
                    structured_content=tool_call["args"]
                )
            )
            LOCAL_HISTORY.confirm_tool_calls.pop(0)
            
            # Register rerun if the tool call has a rerun argument
            if "through" in tool_call["args"] and tool_call["args"]["through"]:
                LOCAL_HISTORY.rerun_immediately = True

            streamlit.rerun()

        else:  # Confirm event is active but no button was clicked
            
            with chat_container:
                with streamlit.chat_message("ai"):

                    streamlit.subheader("Confirm action : **" + tool_call["name"] + "**")
                    streamlit.write(tool_call["args"])
                    streamlit.button("Confirm", key="tool_confirm_button"+str(tool_call["id"]))
                    streamlit.button("Reject", key="tool_reject_button"+str(tool_call["id"]))
