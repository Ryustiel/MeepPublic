
import pydantic, datetime

from typing import Dict, Generator, Iterator, Optional, List, Literal, Annotated, Tuple, Any, TypedDict, Union, Self
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, ToolCall, AIMessageChunk
from graphs._mcp import MCPResponse
import constants, graphs._agents as agent_module

# ============================================ MESSAGE HISTORY ============================================

class MessageExtras(pydantic.BaseModel):
    date: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    summary: Optional[str] = None  # Summary of the AIMessage if it was too long

class SystemMessage_(SystemMessage, MessageExtras):
    author: Optional[str] = None
    lifespan: Optional[int] = None  # Number of inputs before deletion, None means permanent

ToolCallStatus = Literal["unconfirmed", "confirmed", "canceled", "rejected", "processing", "completed", "failed"]
EXTERNAL_STATUS_MAPPING = {
    "unconfirmed": "error",
    "confirmed": "error",
    "canceled": "error",
    "rejected": "error",
    "processing": "error",
    "completed": "success",
    "failed": "error"
}

class InternalToolMessage(pydantic.BaseModel):
    """Represents information to build a ToolMessage. This model is a member of AIMessage_"""
    model_config = {"extra": "allow"}
    tool_call_id: str
    internal_status: ToolCallStatus = "unconfirmed"
    status: Literal["error", "success"] = "error"
    content: Optional[str] = None
    
    def set_status(self, internal_status: ToolCallStatus, content: Optional[str] = None):
        """Set both the internal status and the external status variables."""
        self.internal_status = internal_status
        self.status = EXTERNAL_STATUS_MAPPING.get(internal_status, "error")
        if content is not None:
            self.content = content

class ToolMessage_(ToolMessage, MessageExtras):
    pass

class AIMessage_(AIMessage, MessageExtras):
    activity: str = agent_module.DEFAULT_AGENT
    internal_tool_messages: Dict[str, InternalToolMessage] = pydantic.Field(default_factory=dict)
                
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def unpack(self) -> List[Union[Self, ToolMessage_]]:
        """Unpack the AIMessage_ into a list of messages including its tool messages."""
        messages = [self]
        for tool_call_id, tool_message in self.internal_tool_messages.items():
            messages.append(
                ToolMessage_(
                    tool_call_id=tool_call_id,
                    content=tool_message.content if tool_message.content is not None else "",
                    status=tool_message.status,
                    date=self.date
                )
            )
        return messages

    def get_tool_call_info(self, tool_call_id: str) -> Optional[Tuple[ToolCall, ToolCallStatus]]:
        """Get the tool call associated with the AIMessage_."""
        tool_message = self.internal_tool_messages.get(tool_call_id)
        if tool_message:
            for tool_call in self.tool_calls:
                if tool_call["id"] == tool_call_id:
                    return tool_call, tool_message.internal_status
        return None
    
    def all_tool_call_info(self) -> Iterator[Tuple[ToolCall, InternalToolMessage]]:
        for tool_call in self.tool_calls:
            message = self.internal_tool_messages[tool_call["id"]]
            yield tool_call, message

    @classmethod
    def from_message(cls, message: Union[AIMessage, AIMessageChunk]) -> Self:
        """Create an AIMessage_ from an AIMessage or AIMessageChunk."""
        new_instance = cls.model_validate({**message.model_dump(), "type": "ai"})
        for tool_call in new_instance.tool_calls:  # Register any tool call
            new_tool_message = InternalToolMessage(
                tool_call_id=tool_call["id"],
                content="Waiting for user to confirm on the UI before running."
            )
            new_tool_message.set_status("unconfirmed")
            new_instance.internal_tool_messages[tool_call["id"]] = new_tool_message
            
        return new_instance

class HumanMessage_(HumanMessage, MessageExtras):
    author: Optional[str] = None
    message_id: Optional[int] = None  # Will store the Discord message ID if available

StructuredMessage = Annotated[HumanMessage_|AIMessage_|SystemMessage_, pydantic.Field(discriminator="type")]

def model_validate_structured_message(message: Dict[str, Any]) -> StructuredMessage:
    """
    Create a StructuredMessage from a raw dict input.
    """        
    msg_type = message.get("type")
    if msg_type == "human":
        return HumanMessage_(**message)
    elif msg_type == "ai":
        return AIMessage_(**message)
    elif msg_type == "AIMessageChunk":
        return AIMessage_(**{**message, "type": "ai"})  # Convert to AIMessage for consistency
    elif msg_type == "system":
        return SystemMessage_(**message)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")


# ============================================ SUMMARIES ===================================================

class Summary(pydantic.BaseModel):
    """Summary of a section of the conversation."""
    max_date: datetime.datetime
    min_date: datetime.datetime
    summary: str


# ============================================ WAKE UP =====================================================

class WakeUp(pydantic.BaseModel):
    """{"wakeup":{"user_name":"Raphael"}}"""
    channel_id: Optional[str] = None  # URL of the default channel to go to if no user is found
    user_name: Optional[str] = None  # Name of the user to find (last channel they talked in)
    unless_active_since: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)  # Only wake up if Meep did not wake up after that.


# ============================================ INTERNAL UPDATES ============================================

class InternalChannelUpdates(pydantic.BaseModel):
    """
    Store updates for a specific channel.
    """
    name: Optional[str] = None  # Optional name update for the channel
    channel_type: Optional[str] = None  # Optional channel_type update for the channel
    wakeup_url: Optional[str] = None  # Optional wakeup_url update for the channel
    delete_before: Optional[datetime.datetime] = None  # Optional datetime to delete messages before this date
    no_reactive_tool_call_before: Optional[datetime.datetime] = None  # Optional datetime to update the last tool check
    no_temporary_message_before: Optional[datetime.datetime] = None  # Optional datetime to update the last temporary message check
    new_messages: List[StructuredMessage] = pydantic.Field(default_factory=list)  # List of new messages
    message_updates: Dict[int, StructuredMessage] = pydantic.Field(default_factory=dict)  # Message index -> StructuredMessage
    message_deletes: List[int] = pydantic.Field(default_factory=list)  # List of message indices to delete
    message_append_left: List[StructuredMessage] = pydantic.Field(default_factory=list)  # Messages to append to the left of the channel's messages
    new_summaries: List[Summary] = pydantic.Field(default_factory=list)  # Summaries to add to the channel.
    # NOTE : List[Summary] is ordered by min_date ascending (from oldest to newest)

    def is_empty(self) -> bool:
        """
        Check if there are no updates in this channel.
        """
        return (
            self.name is None
            and self.channel_type is None
            and self.wakeup_url is None
            and self.delete_before is None
            and self.no_reactive_tool_call_before is None
            and self.no_temporary_message_before is None
            and not self.new_messages
            and not self.message_updates
            and not self.message_deletes
            and not self.message_append_left
            and not self.new_summaries
        )

    def merge(self, other: Self):
        """
        Merge another InternalChannelUpdates into this one.
        """
        if other.name is not None:
            self.name = other.name
        if other.channel_type is not None:
            self.channel_type = other.channel_type
        if other.wakeup_url is not None:
            self.wakeup_url = other.wakeup_url
        if other.delete_before is not None:
            self.delete_before = other.delete_before
        if other.no_reactive_tool_call_before is not None:
            self.no_reactive_tool_call_before = other.no_reactive_tool_call_before
        if other.no_temporary_message_before is not None:
            self.no_temporary_message_before = other.no_temporary_message_before
        self.new_messages.extend(other.new_messages)
        self.message_updates.update(other.message_updates)
        self.message_deletes.extend(other.message_deletes)
        self.message_append_left.extend(other.message_append_left)
        self.new_summaries.extend(other.new_summaries)


class InternalUpdates(pydantic.BaseModel):
    """
    Store updates from internal processes.
    """
    channel_updates: Dict[str, InternalChannelUpdates] = pydantic.Field(default_factory=dict)  # Channel ID -> InternalChannelUpdates
    current_channel: Optional[str] = None  # Current channel ID, if any
    tool_updates: List[InternalToolMessage] = pydantic.Field(default_factory=list)  # Tool messages to update in the channels.

    def is_empty(self) -> bool:
        return (
            all(cupdates.is_empty() for cupdates in self.channel_updates.values()) and 
            self.current_channel is None and 
            not self.tool_updates
        )

def internal_updates_reducer(
    left: InternalUpdates, 
    right: Union[InternalUpdates, Literal["reset"]]
) -> InternalUpdates:
    """
    Reduce internal updates.
    Accepts either a reset command or an InternalUpdates object to accumulate.
    """
    if right == "reset":
        return InternalUpdates()
    
    elif isinstance(right, InternalUpdates):
        # Initialize left if None
        if left is None:
            left = InternalUpdates()
            
        # Extend tool updates
        left.tool_updates.extend(right.tool_updates)

        # Update to next channel
        if right.current_channel is not None:
            left.current_channel = right.current_channel

        # Accumulate channel updates
        for channel_id, channel_updates in right.channel_updates.items():
            if channel_id not in left.channel_updates:
                left.channel_updates[channel_id] = InternalChannelUpdates()
            
            # Accumulate updates for this channel
            existing_updates = left.channel_updates[channel_id]
            existing_updates.merge(channel_updates)

        return left
    
    else:
        raise ValueError(f"InternalUpdates reducer only accepts InternalUpdates objects or 'reset' command, got {type(right)}.")


# ============================================ CHANNEL STORE ============================================

class Channel(pydantic.BaseModel):
    """
    Metadata and messages. Messages and traces are bound together therefore member of the same data structure.
    NOTE : The methods do mutate the object but can also be used as non mutable to keep track of the internal updates.
    """
    id: str
    name: str
    channel_type: str = "basic"  # Channel type, defaults to "basic"
    wakeup_url: Optional[str] = None # Optional wakeup URL for Discord or other platforms
    messages: List[StructuredMessage] = pydantic.Field(default_factory=list)
    summaries: Dict[datetime.datetime, List[Summary]] = pydantic.Field(default_factory=dict)  # Summaries indexed by their most recent message date for the delete operation and retrieval

    # These ones are updated automatically
    last_activity: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    max_summary_date: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)  # Date of the most recent max_date from all of the active summaries in this channel
    no_reactive_tool_call_before: Optional[datetime.datetime] = None  # Date of the most recent time all tool calls were checked
    no_temporary_message_before: Optional[datetime.datetime] = None  # Date of the most recent time temporary messages were checked


# ============================================ HISTORY ============================================

class History(pydantic.BaseModel):
    """
    Store the channels, their histories and their messages.
    """
    current_channel: Optional[str] = None  # Current channel ID
    channels: Dict[str, Channel] = pydantic.Field(default_factory=dict)

    def is_empty(self) -> bool:
        """Check if the History object is in its default state (empty)."""
        return self.current_channel is None and len(self.channels) == 0

    def get_current_channel(self) -> Channel:
        """
        Get the current channel by ID.
        If no current channel, return an empty channel, marked as "unregistered".
        If current channel exists, extract the channel info from the channel history if available,
        otherwise return the default channel as it is already set.
        """
        if self.current_channel is None:
            return Channel(id="unregistered", name="Unregistered Channel")
        
        # Try to get the channel from the channels dict if it exists
        if self.current_channel in self.channels:
            return self.channels[self.current_channel]
        
        # If the current_channel ID is set but doesn't exist in channels,
        # return a default channel with that ID
        return Channel(id=self.current_channel, name=self.current_channel)
    
    def get_channel(self, channel_id: str) -> Channel:
        """
        Get a channel by its ID.
        If the channel does not exist, raise a ValueError.
        """
        if channel_id not in self.channels:
            raise ValueError(f"Channel {channel_id} not found in history.")
        return self.channels[channel_id]

    def locate_tool_calls(self, tool_call_ids: List[str]) -> Dict[str, Optional[Tuple[str, int]]]:
        """
        Find the channel_id and indexes associated with the requested tool calls.
        Associate None with a tool_call_id if it could not be found.
        Search first on the current channel, then on the other channels ordered by last activity.
        """
        id_queue = tool_call_ids.copy()
        results = {tid: None for tid in tool_call_ids}
        current_channel = self.get_current_channel()
        checked_channels = []
        
        while current_channel:
            checked_channels.append(current_channel.id)  # Will skip this channel next time
            
            for index, message in enumerate(current_channel.messages):
                if isinstance(message, AIMessage_) and message.tool_calls:
                    for tid in message.internal_tool_messages.keys():
                        if tid in id_queue:
                            id_queue.remove(tid)
                            results[tid] = (current_channel.id, index)
                
            current_channel = None    
            if id_queue:
                # Look for a new channel
                for cid, channel in self.channels.items():
                    if cid in checked_channels: continue
                    if current_channel is None or channel.last_activity > current_channel.last_activity:
                        current_channel = channel
        
        return results

    def find_reactive_tool_calls(self) -> Tuple[List[Tuple[ToolCall, InternalToolMessage]], InternalUpdates]:
        """
        Locate reactable tool calls and return their metadata.
        Should be enough to schedule runs.
        Check channels based on their no_reactive_tool_call_before date.
        Update the no_reactive_tool_call_before date for each channel after checking.
        """
        reactable_tool_calls = []
        internal_updates = InternalUpdates()

        for channel in self.channels.values():

            if not channel.messages:
                continue
            if (
                channel.no_reactive_tool_call_before 
                and channel.messages[-1].date <= channel.no_reactive_tool_call_before
            ):
                continue  # Channel does not have recent messages
            
            no_reactive_tool_call_before_this_date: datetime.datetime = channel.messages[-1].date
            for message in reversed(channel.messages):
                
                if channel.no_reactive_tool_call_before and message.date <= channel.no_reactive_tool_call_before:
                    break  # All messages before that were already checked so there is no need.
                
                if isinstance(message, AIMessage_) and message.has_tool_calls():
                    
                    has_reactive_tools = False
                    for tool_call, tool_message in message.all_tool_call_info():
                        if tool_message.internal_status in ("confirmed", "unconfirmed"):
                            reactable_tool_calls.append((tool_call, tool_message))
                            has_reactive_tools = True
                            
                    if has_reactive_tools:
                        no_reactive_tool_call_before_this_date = message.date - datetime.timedelta(seconds=1)  # This message will be checked again later

            internal_updates.channel_updates[channel.id] = InternalChannelUpdates(
                no_reactive_tool_call_before=no_reactive_tool_call_before_this_date
            )

        return reactable_tool_calls, internal_updates

    def generate_updates_from_mcp_responses(self, responses: List[MCPResponse]) -> InternalUpdates:
        """Create proper History updates based on the MCP Responses"""
        
        updates = InternalUpdates()
        
        # Find the messages associated with the tool call ids
        tool_call_ids = [response.tool_message.tool_call_id for response in responses]
        update_locations = self.locate_tool_calls(tool_call_ids)

        cached_messages: Dict[Tuple[str, int], AIMessage_] = {}
        for mcpresp in responses:
            
            location = update_locations[mcpresp.tool_message.tool_call_id]
            
            if location is None:
                raise ValueError("Could not find location for tool call")
            
            # Find the message
            if location in cached_messages:
                message = cached_messages[location]
            else:
                channel_id, index = location
                message = self.channels[channel_id].messages[index]
                
            # Update the content
            tool_call_id = mcpresp.tool_message.tool_call_id
            message.internal_tool_messages[tool_call_id].set_status(
                internal_status=mcpresp.status,
                content=mcpresp.tool_message.content
            )  # XXX : Message is still via reference so this is ok

            # Create a temporary status update message if the AIMessage is not last in the channel
            if index < len(self.channels[channel_id].messages) - 1:
                if channel_id not in updates.channel_updates:
                    updates.channel_updates[channel_id] = InternalChannelUpdates()
                updates.channel_updates[channel_id].new_messages.append(
                    SystemMessage_(
                        content=f"#toolupdated#{tool_call_id}",
                        lifespan=1
                    )
                )
            
        # Register message updates
        for location, message in cached_messages.items():
            channel_id, index = location
            if channel_id not in updates.channel_updates:
                updates.channel_updates[channel_id] = InternalChannelUpdates()
            updates.channel_updates[channel_id].message_updates[index] = message
            
        # Register updates from MCPResponse
        for response in responses:
            if response.updates:
                updates = internal_updates_reducer(updates, response.updates)

        return updates

# ---------- REQUESTS ----------

def history_reducer(
    left: History, 
    right: Union[Dict[str, Any], History, InternalUpdates, Literal["reset"]]
) -> History:
    """
    The "reset" command resets the store.
    Dict updates are applied to the store.
    Passing a History instance replaces the store.
    InternalUpdates are applied to the store, updating messages and traces.
    
    Input from InternalUpdates like :
    {"history": {"current_channel": "streamlit", "channel_updates": {"streamlit": {"name": "Streamlit", "new_messages": [{"type": "human", "author": "Raphael", "content": "Hallo Servus meep lance ton tool 2 stp"}]}}}}
    {"history": {"current_channel": "streamlit", "channel_updates": {"streamlit": {"name": "Streamlit", "new_messages": [{"type": "tool", "tool_call_id": "call_CFfsoM4LNCaXnSS1FYSuiCx7", "internal_status": "confirmed"}]}}}}
    """

    # 0. Initialize the store if it is None.
    if left is None:
        left = History()

    # 1. Reset the store if the command is "reset".
    if right == "reset":
        return History()

    # 2. If the right value is an instance of History, return it.
    elif isinstance(right, History):
        if right is left:
            return left
        return right if left.is_empty() else left

    # 3. DICT updates
    elif isinstance(right, dict):
        
        # 1. Convert all messages and values to the update format, even the tool message updates.
        updates = InternalUpdates.model_validate(right)
        
        # 1.5. Ensure nested channel_updates are properly converted to InternalChannelUpdates objects
        for channel_id, channel_updates in updates.channel_updates.items():
            if isinstance(channel_updates, dict):
                updates.channel_updates[channel_id] = InternalChannelUpdates.model_validate(channel_updates)
        
        # 2. Apply updates to history using the reducer "pseudo recursively".
        left = history_reducer(left, updates)
        return left

    # 4. Apply local updates from InternalUpdates.
    elif isinstance(right, InternalUpdates):

        left = left.model_copy(deep=True)
        
        for channel_id, channel_updates in right.channel_updates.items():
            
            # 1. Create the channel if it does not exist
            if channel_id not in left.channels:
                left.channels[channel_id] = Channel(id=channel_id, name=channel_id)
            
            channel = left.channels[channel_id]
            
            # 1.5 Channel metadata updates
            if channel_updates.name is not None:
                channel.name = channel_updates.name
            if channel_updates.channel_type is not None:
                channel.channel_type = channel_updates.channel_type
            if channel_updates.wakeup_url is not None:
                channel.wakeup_url = channel_updates.wakeup_url
            if channel_updates.no_reactive_tool_call_before is not None:
                channel.no_reactive_tool_call_before = channel_updates.no_reactive_tool_call_before
            if channel_updates.no_temporary_message_before is not None:
                channel.no_temporary_message_before = channel_updates.no_temporary_message_before

            # 2. Update required messages
            for message_id, message_update in channel_updates.message_updates.items():
                if message_id < len(channel.messages):
                    message_update.date = channel.messages[message_id].date  # Preserve original date (safe timeorder)
                    channel.messages[message_id] = message_update
                else:
                    raise ValueError(f"Message ID {message_id} out of bounds for channel {channel_id} messages of length {len(channel.messages)}.")
            
            # NOTE : Updates below change the order of the messages. The above does not.
            
            # 3. Delete required messages.
            for message_index in sorted(channel_updates.message_deletes, reverse=True):
                if message_index < len(channel.messages):
                    del channel.messages[message_index]

            if channel_updates.delete_before is not None:
                channel.messages = [msg for msg in channel.messages if msg.date >= channel_updates.delete_before]
                channel.summaries = {most_recent_message_date: summaries for most_recent_message_date, summaries in channel.summaries.items() if most_recent_message_date >= channel_updates.delete_before}
                # Deleted all summaries that wouldn't encompass a recent enough message.
                
                # Recalculate max_summary_date after summary deletion
                if channel.summaries:
                    # Find the earliest date from all remaining summaries
                    all_summary_dates = []
                    for summaries_list in channel.summaries.values():
                        for summary in summaries_list:
                            all_summary_dates.append(summary.min_date)
                    
                    if all_summary_dates:
                        channel.max_summary_date = min(all_summary_dates)
                else:
                    # No summaries left, reset to now
                    channel.max_summary_date = datetime.datetime.now()

            # 4. Append messages to the left of the channel's messages.
            for new_message in channel_updates.message_append_left:
                # If there's a message at position 0, ensure the new message's date is not later
                if channel.messages and new_message.date > channel.messages[0].date:
                    new_message.date = channel.messages[0].date
                channel.messages.insert(0, new_message)
            
            # 5. Add new messages
            needs_sorting = False
            for new_message in channel_updates.new_messages:
            
                # 5.1. Check if messages are in chronological order and sort if needed
                if len(channel.messages) >= 1 and new_message.date < channel.messages[-1].date:
                    needs_sorting = True

                channel.messages.append(new_message)
                    
            if needs_sorting:
                channel.messages.sort(key=lambda msg: msg.date)
            
            # 6. Update last_activity if new messages were added
            if channel_updates.new_messages:
                channel.last_activity = max(msg.date for msg in channel_updates.new_messages)
                
            # 7. Add summaries and update most recent date
            if channel_updates.new_summaries:
                
                for new_summary in channel_updates.new_summaries:
                    if new_summary.max_date not in channel.summaries:
                        channel.summaries[new_summary.max_date] = [new_summary]
                    else:
                        # Add and sort summaries by min_date
                        channel.summaries[new_summary.max_date].append(new_summary)
                        channel.summaries[new_summary.max_date].sort(key=lambda s: s.min_date)
                
                # XXX : Could be merged with the previous loop
                earliest_new_summary_date = max(summary.max_date for summary in channel_updates.new_summaries)
                
                if earliest_new_summary_date > channel.max_summary_date:
                    channel.max_summary_date = earliest_new_summary_date

            # 8. Update general channel metadata
            if channel_updates.name:
                channel.name = channel_updates.name
            if channel_updates.channel_type:
                channel.channel_type = channel_updates.channel_type
            if channel_updates.wakeup_url:
                channel.wakeup_url = channel_updates.wakeup_url
            if channel_updates.no_reactive_tool_call_before:
                channel.no_reactive_tool_call_before = channel_updates.no_reactive_tool_call_before
            if channel_updates.no_temporary_message_before:
                channel.no_temporary_message_before = channel_updates.no_temporary_message_before

        # 9. General history updates
        if right.current_channel is not None:
            left.current_channel = right.current_channel
            
        if right.tool_updates:
            positions = left.locate_tool_calls([update.tool_call_id for update in right.tool_updates])

            for tool_update in right.tool_updates:

                position = positions[tool_update.tool_call_id]
                if position is not None:

                    channel_id, message_index = position
                    channel = left.channels[channel_id]
                    message = channel.messages[message_index]
                    if not isinstance(message, AIMessage_): raise ValueError("Located message is not an AIMessage_")
                    
                    tool_message = message.internal_tool_messages[tool_update.tool_call_id]
                    tool_message.set_status(
                        internal_status=tool_update.internal_status,
                        content=tool_update.content if tool_update.content else None
                    )

                    # Create a temporary status update message if the AIMessage is not last in the channel
                    if message_index < len(channel.messages) - 1:
                        temp_message = SystemMessage_(
                            content=f"#toolupdated#{tool_update.tool_call_id}",
                            lifespan=1
                        )
                        # Ensure the temporary message date is not earlier than the last message
                        if channel.messages and temp_message.date < channel.messages[-1].date:
                            temp_message.date = channel.messages[-1].date
                        channel.messages.append(temp_message)

        return left

    else:
        raise ValueError(f"history_reducer only accepts History objects, InternalUpdates, dict updates, or 'reset' command, got {type(right)}.")
