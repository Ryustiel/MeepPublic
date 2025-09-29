
from typing import Callable, Dict, Literal, Tuple, List, Optional, Set, Self

import collections, discord, fastapi, uvicorn, contextlib, asyncio, aiofiles, pydantic, datetime, json, os, httpx
import src.embeds as embeds, src.voice as voice

# TODO : Handle autotool and activity updates as notifications
# TODO : While tool calls remain "unconfirmed", wait for all to be confirmed before running the "through" options

# ============================================ CONFIGURATION ===================================================

LOCAL_HISTORY_BACKUP_PATH = "backup.json"
REPLY_SIZE_LIMIT = 70

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
LANGGRAPH_SERVER = os.environ["LANGGRAPH_SERVER_URL"]
MEEP_THREAD = os.environ["MEEP_THREAD_ID"]
CALLBACK_API_URL = os.environ["CALLBACK_API_URL"]

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.reactions = True
DISCORD_CLIENT = discord.Client(intents=intents)

TRACKED_EMOJIS = [
    "✅",  # Confirm
    "❌",  # Reject
]

EVERYONE_CAN_CONFIRM = [
    "test"
]

USER_ID_TO_ICON_URL = {
    361438727492337664: "https://cdn.discordapp.com/avatars/361438727492337664/c92f6ec6a70d28896307064bfa8fbacb.png?size=1024",  # rouf
    1073126449269194802: "https://cdn.discordapp.com/avatars/1073126449269194802/a_33acce0dad2df87645e96e7c640f913d.gif?size=1024", # raphcvr
    # 389039762712821781: "https://cdn.discordapp.com/avatars/361438727492337664/c92f6ec6a70d28896307064bfa8fbacb.png?size=1024", # raykom
    361185767180992513: "https://cdn.discordapp.com/avatars/361185767180992513/f34d3dac911b6a6b4d40b92a623276ac.png?size=1024", # calzonasiro
    524621123904864256: "https://cdn.discordapp.com/avatars/524621123904864256/2504a4b3159a7d0b59f8a74fca86d62e.png?size=1024", # entekallis
    "everyone": "https://images.vexels.com/media/users/3/157512/isolated/preview/d737a872708b488d89d0341ac9b8bc5a-people-contact-icon-people.png?w=360",
}

# ALLOWED CHANNELS
CHANNEL_ID_TO_NAME = {
    868592422110777404: "DM with Raphael",
    # 868551494914420751: "Tast general",
    # 1074737452926910595: "LoL Aha #general",
    # 1177710933137686538: "LoL Aha #bot",
    # 1380845114674380853: "Blah de famille",
}

# ALLOWED USERS
USER_ID_TO_DISCORD_NAME = {
    361438727492337664: "rouf",
    1073126449269194802: "raphcvr",
    389039762712821781: "raykom",
    361185767180992513: "calzonasiro",
    524621123904864256: "entekallis",
}
USER_ID_TO_MEEP_NAME = {
    361438727492337664: "Raphael",
    1073126449269194802: "Raph. Cvr",
    389039762712821781: "Gael",
    361185767180992513: "Laura",
    524621123904864256: "Marius",
}

ROUF = 361438727492337664  # ID of the user "rouf" (Raphael)

# ============================================ DATA TYPES ===================================================

class DebugMessage(pydantic.BaseModel):
    channel_id: int
    message_id: Optional[int] = None  # Message ID of the debug message, if it exists
    activity: Optional[str] = None
    autotools: Optional[str] = None  # Last autotool that was run
    
    def generate_embed(self) -> discord.Embed:
        embed = discord.Embed(
            title=f"Debug Message",
            description=f"Activity: {self.activity if self.activity else 'None'}\n"
                        f"Autotool: {self.autotools if self.autotools else 'None'}",
            color=discord.Color.blue()
        )
        return embed
    
    async def push(self):
        """Send or update."""
        channel = await DISCORD_CLIENT.fetch_channel(self.channel_id)
        if self.message_id is not None:  # Update existing message
            message = await channel.fetch_message(self.message_id)
            await message.edit(embed=self.generate_embed())
        else:  # Send new message
            message = await channel.send(embed=self.generate_embed())
            await message.add_reaction("❌")
            self.message_id = message.id  # Store the message ID for future updates
            
    async def delete(self):
        channel = await DISCORD_CLIENT.fetch_channel(self.channel_id)
        if self.message_id is not None:  # Delete existing message
            message = await channel.fetch_message(self.message_id)
            await message.delete()
        self.message_id = None  # Clear the message ID

global DEBUG_MESSAGE
DEBUG_MESSAGE: Optional[DebugMessage] = None

class DiscordMessage(pydantic.BaseModel):
    type: Literal["human", "system"] = "human"
    message_id: int
    content: str
    author: Optional[str]
    date: datetime.datetime

class ToolCallUpdate(pydantic.BaseModel):
    tool_call_id: str
    state: Literal["confirmed", "rejected", "canceled"]
    content: Optional[str] = None  # Message displayed alongside the status update (e.g. "Tool call canceled by Raphael")

class ConfirmToolCallData(pydantic.BaseModel):
    authorized_user_ids: Set[int] = []
    tool_call: dict

class ToolCallGroup(pydantic.BaseModel):
    state: Literal["go on", "cancel"] = "cancel"
    tool_call_ids: List[str] = pydantic.Field(default_factory=list)  # List of tool call IDs in this group

class LocalHistory(pydantic.BaseModel):
    confirm_tool_calls: Dict[int, ConfirmToolCallData] = pydantic.Field(default_factory=dict)  # message_id -> ConfirmToolCallData
    tool_call_groups: List[ToolCallGroup] = pydantic.Field(default_factory=list)  # Grouped tool calls must be all reacted to to trigger a rerun
    new_messages: Dict[int, List[DiscordMessage]] = pydantic.Field(default_factory=dict)  # channel_id -> list of DiscordMessage
    new_tool_call_updates: Dict[int, List[ToolCallUpdate]] = pydantic.Field(default_factory=dict)  # channel_id -> list of ToolCallUpdate
    # NOTE : Defaultdict behavior for new_messages and new_tool_call_updates is defined in the load method

    def get_group(self, tool_call_id: str) -> Optional[ToolCallGroup]:
        for group in self.tool_call_groups:
            if tool_call_id in group.tool_call_ids:
                return group
        raise ValueError(f"Tool call ID {tool_call_id} not found in any group")

    @classmethod
    def load(cls, path: str) -> Self:
        
        if not os.path.exists(path):
            
            if os.path.dirname(path):  # if any dirs in path create them
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
            with open(path, 'w', encoding="utf-8") as f:  # Create the file
                f.write(cls().model_dump_json(indent=4))

        with open(path, 'r', encoding="utf-8") as f:
            data = f.read()
            if data.strip() == "":
                return cls()
            else:
                instance = cls.model_validate_json(data)
                # Re-initialize as defaultdicts after loading from JSON
                instance.new_messages = collections.defaultdict(list, instance.new_messages)
                instance.new_tool_call_updates = collections.defaultdict(list, instance.new_tool_call_updates)
                return instance
        
    async def save(self, path: str):
        async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
            await f.write(self.model_dump_json(indent=4))

LOCAL_HISTORY = LocalHistory.load(LOCAL_HISTORY_BACKUP_PATH)
LOCAL_HISTORY_LOCK = asyncio.Lock()

SEND_TASK = None

CHANNEL_WAIT_STATUS: Dict[int, bool] = {}  # channel_id -> is waiting status
WAIT_TASKS: Dict[int, asyncio.Task] = {}  # channel_id -> wait task

async def wait_and_check(channel_id: int, duration: int):
    """Wait for duration seconds, then check if buffer needs to be run."""
    await asyncio.sleep(duration)
    
    if CHANNEL_WAIT_STATUS.get(channel_id, True):  # If still True (still waiting)
        channel = await DISCORD_CLIENT.fetch_channel(channel_id)
        if channel is not None:
            print(f"Wait timeout reached for channel {channel_id}, scheduling buffer run.")
            asyncio.create_task(handle_channel_buffer(channel))

def simplify_message_content(message: discord.Message) -> str:
    if len(message.content) < REPLY_SIZE_LIMIT:
        return message.content
    return f"{message.content[:REPLY_SIZE_LIMIT].replace('http', '<link>')}...(+{len(message.content) - REPLY_SIZE_LIMIT} characters)"

# ============================================ ON MESSAGE ===================================================


@DISCORD_CLIENT.event
async def on_message(message: discord.Message):
    
    global SEND_TASK

    if message.author.bot or message.channel.id not in CHANNEL_ID_TO_NAME:
        return

    content = message.content
    if message.reference:
        referenced_message = await message.channel.fetch_message(message.reference.message_id)
        if len(referenced_message.content) > REPLY_SIZE_LIMIT:
            content = (
                f"[referencing an older message from {USER_ID_TO_MEEP_NAME.get(referenced_message.author.id, referenced_message.author.name)}: "
                f"{simplify_message_content(referenced_message)} "
                f"message_id={referenced_message.id}) "
                f"{message.content}]"
            )
        else:
            content = (
                f"[referencing an older message from {USER_ID_TO_MEEP_NAME.get(referenced_message.author.id, referenced_message.author.name)}: "
                f"{referenced_message.content}) "
                f"{message.content}]"
            )
        
    if message.attachments:
        content += "\nAttachments:\n"    
        for attachment in message.attachments:
            content += f"{attachment.filename} {attachment.url}\n"

    async with LOCAL_HISTORY_LOCK:
        
        LOCAL_HISTORY.new_messages[message.channel.id].append(
            DiscordMessage(
                message_id=message.id,
                content=content,
                author=USER_ID_TO_MEEP_NAME.get(message.author.id, None), 
                date=message.created_at
            )
        )
        await LOCAL_HISTORY.save(LOCAL_HISTORY_BACKUP_PATH)

    # If the handle buffer is not already running, start it from this point.
    # It will spin until all buffered messages are processed (even those added to the current channel while replying).
    # If it's already running, it will handle the new message by itself when it finishes processing the current buffer. (look at the end of handle_channel_buffer)    
    if not SEND_TASK or SEND_TASK.done():
        SEND_TASK = asyncio.create_task(handle_channel_buffer(message.channel))
        await SEND_TASK  # XXX Jump to the task immediately to process the message

# ============================================ OTHER AWARE EVENTS ===================================================

@DISCORD_CLIENT.event
async def on_raw_reaction_remove(reaction: discord.RawReactionActionEvent):
    """Create system message events for each removal"""

    if reaction.user_id not in USER_ID_TO_MEEP_NAME or reaction.channel_id not in CHANNEL_ID_TO_NAME:
        return

    emoji = reaction.emoji.name if isinstance(reaction.emoji, discord.PartialEmoji) else reaction.emoji

    async with LOCAL_HISTORY_LOCK:
        LOCAL_HISTORY.new_messages[reaction.channel_id].append(
            DiscordMessage(
                type="system",
                message_id=reaction.message_id,
                content=f"User {USER_ID_TO_MEEP_NAME.get(reaction.user_id, 'UNAVAILABLE')} removed their reaction {emoji}",
                author="Discord Client",
                date=datetime.datetime.now()
            )
        )
        await LOCAL_HISTORY.save(LOCAL_HISTORY_BACKUP_PATH)

@DISCORD_CLIENT.event
async def on_raw_message_edit(event: discord.RawMessageUpdateEvent):
    """Create system message events for each edited message"""

    if event.cached_message is None or event.message is None or event.message.author.id not in USER_ID_TO_MEEP_NAME:
        print(f"Message {event.message_id} edited but not found in cache. Skipping.")
        return
    
    if event.cached_message.content == event.message.content:
        print(f"Message {event.message_id} edited but content is the same. Skipping.")
        return

    if (
        event.channel_id not in CHANNEL_ID_TO_NAME or 
        event.message.author.id not in USER_ID_TO_MEEP_NAME
    ):
        return  # Check if author is known (and not Meep themselves)

    async with LOCAL_HISTORY_LOCK:
        LOCAL_HISTORY.new_messages[event.channel_id].append(
            DiscordMessage(
                type="system",
                message_id=event.message.id,
                content=(
                    f"User {USER_ID_TO_MEEP_NAME.get(event.message.author.id, event.message.author.name)} edited one of their older message: "
                    f"from {simplify_message_content(event.cached_message)} to {simplify_message_content(event.message)}"
                ),
                author="Discord Client",
                date=datetime.datetime.now()
            )
        )
        await LOCAL_HISTORY.save(LOCAL_HISTORY_BACKUP_PATH)

@DISCORD_CLIENT.event
async def on_raw_message_delete(event: discord.RawMessageDeleteEvent):
    """Create system message events for each deleted message"""

    if event.channel_id not in CHANNEL_ID_TO_NAME or event.cached_message.author.id not in USER_ID_TO_MEEP_NAME:
        return
    
    if event.cached_message is None:
        print(f"Message {event.message_id} deleted but not found in cache. Skipping.")
        return

    if event.cached_message.author.id not in USER_ID_TO_DISCORD_NAME:
        return  # Check if author is known (and not Meep themselves)

    content = simplify_message_content(event.cached_message)

    async with LOCAL_HISTORY_LOCK:
        LOCAL_HISTORY.new_messages[event.channel_id].append(
            DiscordMessage(
                type="system",
                message_id=event.cached_message.id,
                content=f"User {USER_ID_TO_MEEP_NAME.get(event.cached_message.author.id, 'UNAVAILABLE')} deleted one of their older messages: {content}",
                author="Discord Client",
                date=datetime.datetime.now()
            )
        )
        await LOCAL_HISTORY.save(LOCAL_HISTORY_BACKUP_PATH)


# ============================================ ON REACTION ADD ===================================================


@DISCORD_CLIENT.event
async def on_raw_reaction_add(reaction: discord.RawReactionActionEvent):
    """Create system message for each added reaction, handle tool call confirmations."""
    
    global SEND_TASK, DEBUG_MESSAGE
    
    if isinstance(reaction.emoji, str):
        emoji = reaction.emoji
    elif isinstance(reaction.emoji, discord.PartialEmoji):
        emoji = reaction.emoji.name
    
    if (
        reaction.user_id == DISCORD_CLIENT.user.id or 
        reaction.channel_id not in CHANNEL_ID_TO_NAME
        # reaction.message_id not in LOCAL_HISTORY.confirm_tool_calls
    ):
        return
    
    if emoji not in TRACKED_EMOJIS:  # Keep track of any reactable emoji

        channel = await DISCORD_CLIENT.fetch_channel(reaction.channel_id)
        message = await channel.fetch_message(reaction.message_id)

        async with LOCAL_HISTORY_LOCK:
            if reaction.channel_id not in LOCAL_HISTORY.new_messages:
                LOCAL_HISTORY.new_messages[reaction.channel_id] = []
            LOCAL_HISTORY.new_messages[reaction.channel_id].append(
                DiscordMessage(
                    type="system",
                    message_id=reaction.message_id,
                    content=(
                        f"{USER_ID_TO_MEEP_NAME.get(reaction.user_id, 'UNAVAILABLE')} "
                        f"reacted with {emoji} to the message {simplify_message_content(message)}"
                    ),
                    author="Discord Client",
                    date=datetime.datetime.now()
                )
            )
            await LOCAL_HISTORY.save(LOCAL_HISTORY_BACKUP_PATH)
            return
    
    # Handle tool call confirmation or rejection
    delete_from_channel = None

    async with LOCAL_HISTORY_LOCK:
        
        tool_call_data = LOCAL_HISTORY.confirm_tool_calls.get(reaction.message_id, None)
        
        if tool_call_data is None:
            
            # XXX : Delete any message with an ❌ reaction that is not a pending tool call
            if emoji == "❌":
                channel = await DISCORD_CLIENT.fetch_channel(reaction.channel_id)
                message = await channel.fetch_message(reaction.message_id)
                await message.delete()

                if DEBUG_MESSAGE and DEBUG_MESSAGE.message_id == reaction.message_id:
                    DEBUG_MESSAGE = None

            return  # No tool call data found for this message
        
        if reaction.user_id in tool_call_data.authorized_user_ids:
            
            if reaction.channel_id not in LOCAL_HISTORY.new_tool_call_updates:
                LOCAL_HISTORY.new_tool_call_updates[reaction.channel_id] = []
            
            # 1. Find the list that contains current tool call
            group = LOCAL_HISTORY.get_group(tool_call_data.tool_call["id"])
            
            match emoji:
                case "✅":
                    LOCAL_HISTORY.new_tool_call_updates[reaction.channel_id].append(
                        ToolCallUpdate(
                            tool_call_id=tool_call_data.tool_call["id"], 
                            state="confirmed"
                        )
                    )
                    group.state = "go on"

                case "❌":
                    LOCAL_HISTORY.new_tool_call_updates[reaction.channel_id].append(
                        ToolCallUpdate(
                            tool_call_id=tool_call_data.tool_call["id"],
                            state="rejected",
                            content=f"{USER_ID_TO_MEEP_NAME.get(reaction.user_id, 'UNAVAILABLE')} clicked the Reject button."
                        )
                    )
                    
            channel = await DISCORD_CLIENT.fetch_channel(reaction.channel_id)
            delete_from_channel = channel

            del LOCAL_HISTORY.confirm_tool_calls[reaction.message_id]  # Remove the tool call data after processing
            group.tool_call_ids.remove(tool_call_data.tool_call["id"])  # Remove the tool call ID from the group
            
            # Only respond immediately if the tool call was confirmed and the group has been emptied by this action
            if emoji == "✅" and not group.tool_call_ids:
                
                if not SEND_TASK or SEND_TASK.done():  # Schedule a rerun
                    SEND_TASK = asyncio.create_task(handle_channel_buffer(channel))
                LOCAL_HISTORY.tool_call_groups.remove(group)  # Remove the group if empty
                    
            await LOCAL_HISTORY.save(LOCAL_HISTORY_BACKUP_PATH)
                    
    if delete_from_channel:
        message = await delete_from_channel.fetch_message(reaction.message_id)
        await message.delete()


# ============================================ BUFFER ===================================================


async def handle_channel_buffer(channel: discord.TextChannel | discord.DMChannel):
    
    global SEND_TASK, DEBUG_MESSAGE
    
    # No longer waiting as this one is reached
    CHANNEL_WAIT_STATUS[channel.id] = False
    
    rerun = True
    while rerun:
        rerun = False
        
        async with LOCAL_HISTORY_LOCK:
            
            messages: List[DiscordMessage] = LOCAL_HISTORY.new_messages.get(channel.id, []).copy()
            tool_call_updates: List[ToolCallUpdate] = LOCAL_HISTORY.new_tool_call_updates.get(channel.id, []).copy()
            
            LOCAL_HISTORY.new_messages[channel.id].clear()
            LOCAL_HISTORY.new_tool_call_updates[channel.id].clear()
            
            await LOCAL_HISTORY.save(LOCAL_HISTORY_BACKUP_PATH)
            
        # TODO : Add more metadata to the input
        
        if isinstance(channel, discord.TextChannel):
            temporary_channel_data = {
                "id": str(channel.id),
                "name": f"{channel.guild.name} {channel.name}"
            }
        else:
            temporary_channel_data = {
                "id": str(channel.id),
                "name": f"DM with {messages[0].author if messages else 'UNAVAILABLE'}"
            }
        sender_names = [msg.author or "UNAVAILABLE" for msg in messages]
        timestamps = [msg.date.isoformat() for msg in messages]
        channel_ids = [str(channel.id)] * len(messages)
        
        # ============================================ SEND TO LANGGRAPH ===================================================
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(5, read=60)) as client:

            token_buffer = ""
            message_reference: Optional[discord.MessageReference] = None
            tool_group_buffer = []  # XXX : group all tool calls from the same stream run

            async with client.stream(
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
                            "current_channel": "discord",
                            "tool_updates": [
                                {
                                    "tool_call_id": update.tool_call_id, 
                                    "internal_status": update.state,
                                    "content": update.content if update.content else None
                                }
                                for update in tool_call_updates
                            ],
                            "channel_updates": {
                                "discord": {  # TODO : Channel id should be variable and depend on something
                                    "name": "Discord " + CHANNEL_ID_TO_NAME.get(channel.id, "UNDEFINED Channel"),
                                    "channel_type": "discord",
                                    "wakeup_url": f"{CALLBACK_API_URL}/wakeup/{channel.id}",  # XXX : Change when using external server
                                    "new_messages": [
                                        {
                                            "type": msg.type, 
                                            "message_id": str(msg.message_id) if msg.message_id else None,  # XXX : Warning for later : Might cause a bug if a system message has "message_id": None
                                            "content": msg.content,
                                            "author": msg.author,
                                        }
                                        for msg in messages
                                    ],
                                }
                            }
                        }
                    },
                },
                headers = {"Content-Type": "application/json"}
            ) as response:

                async for line in response.aiter_lines():

                    if line and line.startswith("data:"):
                            
                        content = line[7:-1]

                        if (
                            content.startswith("\"run_id\":\"")
                        ):
                            continue
                        
                        elif content.startswith("#activity#"):

                            if DEBUG_MESSAGE is None:
                                DEBUG_MESSAGE = DebugMessage(
                                    channel_id = channel.id,
                                    activity = content[10:]
                                )
                            elif DEBUG_MESSAGE.activity == content[10:]:
                                continue  # Don't update if remaining in the same state
                            else:
                                await DEBUG_MESSAGE.delete()
                                DEBUG_MESSAGE = DebugMessage(
                                    channel_id = channel.id,
                                    activity = content[10:]
                                )
                                
                            await DEBUG_MESSAGE.push()
                            continue
                        
                        elif content.startswith("#rerun#"):
                            rerun = True

                        elif content.startswith("#wait#"):
                            duration = int(content[6:])

                            CHANNEL_WAIT_STATUS[channel.id] = True
                            
                            if channel.id in WAIT_TASKS and not WAIT_TASKS[channel.id].done():
                                WAIT_TASKS[channel.id].cancel()

                            WAIT_TASKS[channel.id] = asyncio.create_task(wait_and_check(channel.id, duration))
                            continue

                        elif content.startswith("#typing#"):
                            async with channel.typing():
                                continue
                            
                        elif content.startswith("#reference#"):
                            message_reference = discord.MessageReference(
                                    message_id=int(content[11:]), 
                                    channel_id=channel.id,
                                    guild_id=channel.guild.id if channel.guild else None,
                                    fail_if_not_exists=False
                                )
                            continue

                        # ============================================ PROCESS TOOL CALLS ===================================================

                        elif content.startswith("#tool#"):
                            
                            unescaped_content = content[6:].replace("\\\"", "\"")
                            tool_call = json.loads(unescaped_content)
                            
                            if "skip_confirmation" in tool_call["args"] and tool_call["args"]["skip_confirmation"]:
                                args = tool_call["args"]
                                del args["skip_confirmation"]
                                await channel.send(f"**DEBUG**: Running {tool_call['name']} with args {args}")
                                continue

                            # XXX : Direct management of discord specific tool calls
                            
                            match tool_call["name"]:
                                
                                case "send_structured_message":
                                    
                                    # 1. Send the structured message
                                    await channel.send(
                                        embed = embeds.custom_embed(
                                            tool_call["args"]["title"],
                                            description=tool_call["args"]["body"],
                                            color=tool_call["args"].get("color_hex", None),
                                            image_url=tool_call["args"].get("image_url", None)
                                        )
                                    )
                                    
                                    # 2. Mark the tool call as confirmed (no need for user confirmation)
                                    LOCAL_HISTORY.new_tool_call_updates[channel.id].append(
                                        ToolCallUpdate(
                                            tool_call_id=tool_call["id"], 
                                            state="confirmed"
                                        )
                                    )

                                case _:
                                    
                                    if "skip_confirmation" in tool_call["args"] and tool_call["args"]["skip_confirmation"]:
                                        continue  # Skip creating a confirmation box
                                    
                                    # 1. Decide on display names and icons
                                    allowed_users_ids = set()
                                    
                                    # 1.1 Assigned requestors (by the LLM)
                                    requestor_id = None
                                    if tool_call["args"].get("requestor", None):
                                        for user_id, meep_name in USER_ID_TO_MEEP_NAME.items():
                                            if meep_name.lower() == tool_call["args"]["requestor"].lower():
                                                requestor_id = user_id
                                                allowed_users_ids = {user_id, ROUF}
                                                break
                                            
                                    # 1.2 Natural requestors (depending on the tool call name)
                                    if requestor_id is None:
                                        if tool_call["name"] in EVERYONE_CAN_CONFIRM:
                                            requestor_id = "everyone"
                                            allowed_users_ids = set(USER_ID_TO_DISCORD_NAME.keys())
                                    
                                    # 1.3 Default requestor
                                    if allowed_users_ids == set():
                                        requestor_id = ROUF
                                        allowed_users_ids = {ROUF}
                                    
                                    # 2. Send a confirmation message on discord
                                    confirmation_message = await channel.send(
                                        embed=embeds.cf_tool_call(
                                            tool_call["name"], 
                                            json_input=tool_call["args"],
                                            allowed_users_string=USER_ID_TO_DISCORD_NAME.get(requestor_id, ", ".join(USER_ID_TO_DISCORD_NAME.get(user_id, "UNAVAILABLE") for user_id in allowed_users_ids)),
                                            security_icon_url=USER_ID_TO_ICON_URL.get(requestor_id, ""),
                                        )
                                    )

                                    # 3. Add reactions for interaction
                                    await asyncio.gather(
                                        confirmation_message.add_reaction("✅"), 
                                        confirmation_message.add_reaction("❌")
                                    )
                                    
                                    # 4. Register the tool call in the local history
                                    LOCAL_HISTORY.confirm_tool_calls[confirmation_message.id] = ConfirmToolCallData(
                                        authorized_user_ids=allowed_users_ids, 
                                        tool_call=tool_call
                                    )
                                    
                                    # 5. Store the tool call in the group
                                    tool_group_buffer.append(tool_call["id"])
                                    
                        # ============================================ PROCESS SEND COMMANDS ===================================================
                        
                        elif content.startswith("#send#"):
                            if token_buffer and token_buffer.strip():
                                message: discord.Message = await channel.send(
                                    token_buffer.replace(r"\n", "\n"),
                                    reference=message_reference
                                )
                                message_reference = None
                            token_buffer = ""
                        
                        elif content.startswith("#update#"):
                            if token_buffer and token_buffer.strip():
                                if message is None:
                                    message: discord.Message = await channel.send(
                                        token_buffer.replace(r"\n", "\n"),
                                        reference=message_reference
                                    )
                                    message_reference = None
                                else:
                                    await message.edit(
                                        content=token_buffer,
                                    )
                                    message_reference = None
                                token_buffer = ""
                                
                        else:
                            token_buffer += content
            
        if tool_group_buffer:
            LOCAL_HISTORY.tool_call_groups.append(ToolCallGroup(tool_call_ids=tool_group_buffer))
            
    if token_buffer and token_buffer.strip():
        await channel.send(token_buffer.replace(r"\n", "\n"))

    # ============================================ SCHEDULE NEXT TASKS ===================================================

    # If more human messages were added to the buffer while processing, re-trigger the task
    
    async with LOCAL_HISTORY_LOCK:
        current_messages = LOCAL_HISTORY.new_messages.copy()
        
    for channel_id, messages in current_messages.items():
        # Check if there are any human messages (not system notifications)
        has_human_messages = any(msg.type == "human" for msg in messages)
        if has_human_messages:
            new_channel = await DISCORD_CLIENT.fetch_channel(channel_id)
            if new_channel is not None:
                print(f"Re-triggering handle_channel_buffer for channel {channel_id} with {len(messages)} messages.")
                SEND_TASK = asyncio.create_task(handle_channel_buffer(new_channel))
                break


# ============================================== TINY FAST API ========================================================


@DISCORD_CLIENT.event
async def on_ready():
    print(f"Discord running and logged in as {DISCORD_CLIENT.user}")

@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("FastAPI app running")
    asyncio.create_task(DISCORD_CLIENT.start(DISCORD_TOKEN))
    yield
    print("FastAPI shutting down")
    await DISCORD_CLIENT.close()
    print("Discord client closed.")

app = fastapi.FastAPI(lifespan=lifespan)

@app.get("/wakeup/{channel_id}")
async def wakeup_endpoint(channel_id: int):
    """Meep attempts to talk at the provided channel."""
    # 1. Check if channel is allowed
    if channel_id not in CHANNEL_ID_TO_NAME:
        print("Channel does not exist or is not allowed", str(channel_id))
        return {"status": "error", "message": "Channel does not exist or is not allowed"}
    # 2. Create task handle buffer for said channel
    channel = await DISCORD_CLIENT.fetch_channel(channel_id)
    asyncio.create_task(handle_channel_buffer(channel))
    print(f"Wakeup call sent to channel {channel_id}")
    return {"status": "success", "message": f"Wakeup call sent to channel {channel_id}"}

# if __name__ == "__main__":
#     if True:  # Use custom endpoint
#         uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
#     else:
#         DISCORD_CLIENT.run(DISCORD_TOKEN)
