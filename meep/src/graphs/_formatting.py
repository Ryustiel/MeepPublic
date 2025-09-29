"""
Some formatting functions for the conversation history.
"""

from typing import List, Dict, Literal, Optional, Union, Tuple

import datetime, collections
import constants, graphs._data as data


def _time_ago(dt: datetime.datetime) -> str:
    """Converts a datetime object to a human-readable 'time ago' string."""
    now = datetime.datetime.now()
    
    delta = now - dt
    seconds = delta.total_seconds()

    if seconds < 0:
        return f"{seconds}s"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    else:
        return f"{int(seconds / 86400)}d ago"


def _count_message_size(message: Union[data.StructuredMessage, data.Summary], use_summary: bool) -> int:
    """
    Count the size of a message for the purpose of assembling messages.
    """
    if isinstance(message, data.Summary):
        return len(message.summary)
    elif hasattr(message, 'summary') and message.summary and use_summary:  # Message is a StructuredMessage with a summary
        count = len(message.summary)
    else:
        count = len(message.content)
    if isinstance(message, data.AIMessage_) and message.internal_tool_messages:  # Account for the size of the tool messages
        count += sum(len(tool_msg.content or "") for tool_msg in message.internal_tool_messages.values())
    return count


# ============================================ ASSEMBLY SECTION ======================


def assemble_messages(
    messages: List[data.StructuredMessage], 
    summaries: Dict[datetime.datetime, List[data.Summary]],
    summary_rank_threshold: int = 0,
    use_message_summaries: bool = True,
    max_size: int = 4000,
    min_message: int = 0,
    max_message: Optional[int] = None,
    min_date: Optional[datetime.datetime] = None,
    max_date: Optional[datetime.datetime] = None
) -> List[Union[data.StructuredMessage, data.Summary]]:
    """
    Organise les messages et les summmaries dans un ordre chronologique.
    
    Priority is :
    max_date > min_message > min_date > max_message
    (means min_date can be bypassed if not enough messages are added)
    
    Parameters:
    - messages: A list of structured messages to assemble.
    - summaries: A dict of summaries, indexed by their max date.
    - summary_rank_threshold: Greater numbers use smaller summary spans if available. (so many small summaries instead of a big encompassing summaries whenever possible)
    - use_message_summaries: Whether to count characters from message summaries instead of the message itself if available.
    - max_size: The maximum size of the included messages. If size exceeds this, attempt using a larger summary for a previous message.
    - min_message: The minimum number of non summary messages to include.
    - max_message: The maximum number of messages to include. Priority is to the min_messages parameter in case of conflict.
    - min_date: The minimum date for messages to include.
    - max_date: The maximum date for messages to include.
    """
    character_count = 0
    assembled: List[Union[data.StructuredMessage, data.Summary]] = []

    for i, msg in reversed(list(enumerate(messages))):

        if max_date and msg.date >= max_date:  # Exclude max_date itself
            continue  # Message is too early
        elif min_date and msg.date < min_date:  # Include min_date
            if min_message:
                min_date = msg.date  # Include this message
            else:
                break  # No older messages could be found
        elif max_message and len(assembled) >= max_message:
            break
    
        # NOTE : When summaries are created, their max_date is set to one of the existing messages anyways, its starting point.
        if min_message == 0 and msg.date in summaries:  # Then we should check if a summary exists
            
            # A summary exists
            candidate_summaries = summaries[msg.date]
            idx = min(summary_rank_threshold, len(candidate_summaries) - 1)
            summary = candidate_summaries[idx]
            character_count += _count_message_size(summary, use_message_summaries)
            assembled.append(summary)
        
        else:  # No summary exists or currently below the min number of messages
            if min_message > 0:
                min_message -= 1  # Count down the minimum messages to include
            
            character_count += _count_message_size(msg, use_message_summaries)
            assembled.append(msg)
        
        found_something = True
        while character_count > max_size and found_something:  # Backtrack and try larger summaries

            found_something = False
            for i, item in enumerate(list(reversed(assembled))):  # NOTE : assembled is ordered from the most recent (big date) to the least (small date)
                
                if isinstance(item, data.Summary):
                    item_end = item.max_date
                    item_start = item.min_date
                else:
                    item_end = item.date
                    item_start = item.date
                    
                if item_end not in summaries:  # No summaries available from this message
                    continue
                    
                candidate_summaries = summaries[item_end]
                for candidate_summary in candidate_summaries:

                    if candidate_summary.min_date < item_start:  # Found a larger summary, use it

                        insert_index = None
                        for sub_index, sub_item in reversed(list(enumerate(assembled.copy()))):  # Remove all messages that fall in the new timeframe

                            # NOTE : If removing a summary at this point, any message beyond candidate_summary.min_date were not yet
                            # added because the reversed(list(enumerate(messages))) loop is only at step i=summary.max_date
                            # So simply deleting a summary without any further operation causes no issue.
                            if isinstance(sub_item, data.Summary):
                                if (
                                    sub_item.max_date >= candidate_summary.min_date
                                    and sub_item.max_date <= candidate_summary.max_date
                                ):
                                    insert_index = sub_index
                                    character_count -= _count_message_size(sub_item, use_message_summaries)
                                    assembled.pop(sub_index) 
                            elif (
                                sub_item.date >= candidate_summary.min_date
                                and sub_item.date <= candidate_summary.max_date
                            ):
                                insert_index = sub_index
                                character_count -= _count_message_size(sub_item, use_message_summaries)
                                assembled.pop(sub_index)
                                
                        if insert_index is None: raise ValueError("Could not find index, should never happen because at least the summary itself should be deleted.")
                        
                        assembled.insert(insert_index, candidate_summary)
                        character_count += _count_message_size(candidate_summary, use_message_summaries)

                        found_something = True
                        break  # Check for character_count once more
                    
        if character_count > max_size:  # Then remove the last message (the one that made it all exceed the limit)
            assembled.pop()
            continue  # Ignore this message and keep adding the others according to min_message requirements (if min_message wasn't even reached it suggest this message was too big to begin with)
        
    return assembled[::-1]  # Reverse to have the oldest message first


# ================================= GROUPING SECTION ========================================


def group_messages(
    messages: List[Union[data.StructuredMessage, data.Summary]],
    time_gap_reference: Dict[datetime.datetime, datetime.timedelta],
    max_group_size_reference: Dict[datetime.datetime, Optional[int]] = {}  # If specified, limits the total size of messages in each group depending on the most recent message in the group.
) -> List[List[Union[data.StructuredMessage, data.Summary]]]:
    """
    Group messages and summaries that are close to one another.
    Use both max_date and min_date from summaries (processed like time spans).
    Group at index 0 has the oldest messages.
    
    Parameters:
    - time_gap_reference: {threshold: max_gap} If message is older than threshold, it can be grouped with other older messages up to max_gap apart. Messages before and after thresholds can still be grouped together, if they're close enough.
    - max_group_size_reference: {threshold: max_size} If specified, limits the total size of messages in each group depending on the most recent message in the group.
    """
    time_gap_reference = time_gap_reference.copy()  # Avoid mutating external dict input
    
    groups = []
    current_group = []
    last_message_date: Optional[datetime.datetime] = None
    
    max_threshold = max(thr for thr in time_gap_reference.keys())
    max_time_gap = time_gap_reference[max_threshold]
    del time_gap_reference[max_threshold]
    
    for message in messages:  # Dates get larger over the loop
        
        if time_gap_reference and message.date < max_threshold:
            max_threshold = max(thr for thr in time_gap_reference.keys())
            max_time_gap = time_gap_reference[max_threshold]
            del time_gap_reference[max_threshold]

        if (
            not current_group 
            or (
                isinstance(message, data.Summary)
                and message.min_date - last_message_date <= max_time_gap
            )
            or (
                not isinstance(message, data.Summary)
                and message.date - last_message_date <= max_time_gap
            )
        ):
            current_group.append(message)

        else:
            groups.append(current_group)
            current_group = [message]
        
        if isinstance(message, data.Summary):
            last_message_date = message.max_date
        else:
            last_message_date = message.date

    if current_group:
        groups.append(current_group)
        
    if not max_group_size_reference:
        return groups
        
    # Check for group size
    final_groups = []
    group_queue = collections.deque(groups)
    size_thresholds = sorted(max_group_size_reference.keys())
    
    while group_queue:

        group = group_queue.popleft()
        
        # Compute the max group size based on the message max_date
        max_date = max(msg.max_date if isinstance(msg, data.Summary) else msg.date for msg in group)
        reference = next((key for key in reversed(size_thresholds) if max_date >= key), None)
        if reference is None or max_group_size_reference[reference] is None:
            final_groups.append(group)
            continue
        else:
            max_group_size = max_group_size_reference[reference]

        if len(group) == 1:  # Append the group anyways, truncate to 1.5x the size threshold
            msg = group[0]
            if isinstance(msg, data.Summary):
                new_message = msg.model_copy(deep=True)
                if len(new_message.summary) > 1.5 * max_group_size:
                    new_message.summary = new_message.summary[:int(1.5 * max_group_size)]
            else:
                new_message = msg.model_copy(deep=True)
                if getattr(new_message, 'summary', None):
                    new_message.content = new_message.summary
                if len(new_message.content) > 1.5 * max_group_size:
                    new_message.content = new_message.content[:int(1.5 * max_group_size)]
                    
            final_groups.append([new_message])
            continue

        cumulated_size = 0
        for message in group:
            cumulated_size += _count_message_size(message, use_summary=True)
                
        if cumulated_size > max_group_size:
            
            # TODO : If one message is responsible for over 30% of the size then split into 3 groups
            # : Whatavers before the message, message itself, and whats after
            # NOTE : This could be done as a precheck when initially creating the groups, will be done there
            
            # Split at the largest gap between two messages in the current group
            split_index = 1
            max_gap = None
            previous_message_date = None
            
            for i, message in enumerate(group):
                message_date = message.date if not isinstance(message, data.Summary) else message.max_date
                if previous_message_date is not None:
                    gap = message_date - previous_message_date
                    if max_gap is None or gap > max_gap:
                        split_index = i
                        max_gap = gap
                previous_message_date = message_date

            if split_index <= 0:
                split_index = 1  # Split at least at 1

            # Insert back into the queue for another size check while preserving the order of messages
            left = group[:split_index]
            right = group[split_index:]
            group_queue.appendleft(right)
            group_queue.appendleft(left)
        else:
            final_groups.append(group)

    return final_groups


# ================================= DISPLAY SECTION ========================================


def _format_grouped_messages(  # XXX : Maybe need to reverse message order in group
    group: List[Union[data.HumanMessage_, data.SystemMessage_, data.Summary]], 
    channel_name: str,
    use_summaries: bool = True,  # Use summaries if available
    show_all_dates: bool = False  # Show all dates regardless of grouping
) -> data.HumanMessage_:
    """Formats a list of human messages from the current channel."""
    most_recent_msg = group[-1]

    if not show_all_dates:
        if isinstance(most_recent_msg, data.Summary):
            content_lines = [f"from {_time_ago(most_recent_msg.min_date)} to {_time_ago(most_recent_msg.max_date)}"]
        else:
            content_lines = [_time_ago(most_recent_msg.date)]
    else:
        content_lines = []
    
    for msg in group:
        if isinstance(msg, data.Summary):
            if show_all_dates:
                line = f"*{_time_ago(msg.min_date)} to {_time_ago(msg.max_date)}: {msg.summary}*"
            else:
                line = f"*{msg.summary}*"
        elif msg.type == "system":
            author = msg.author or 'System'
            if show_all_dates:
                line = f"{_time_ago(msg.date)}: [{author}] {msg.content}"
            else:
                line = f"[{author}] {msg.content}"
        elif isinstance(msg, data.AIMessage_):
            raise ValueError("AIMessage_ instances are typically not grouped together.")
        else:
            author = msg.author or 'Unspecified User'
            if use_summaries and msg.summary:
                content = msg.summary
            else:
                content = msg.content
            if show_all_dates:
                line = f"{_time_ago(msg.date)}: {author}: {content}"
            else:
                line = f"{author}: {content}"
        content_lines.append(line)
    content = "\n".join(content_lines)

    return data.HumanMessage_(
        content=content,
        date=most_recent_msg.max_date if isinstance(most_recent_msg, data.Summary) else most_recent_msg.date,
        author="Grouped Messages",
    )


def formatted_conversation(
    history: data.History, 
    current_channel_id: Optional[str] = None,
    use_summaries: bool = True,
    from_time_ago: datetime.timedelta = datetime.timedelta(days=1),
    min_date: Optional[datetime.datetime] = None,
    max_date: Optional[datetime.datetime] = None,
    min_message: int = 3,
    max_message: Optional[int] = None,
) -> List[data.StructuredMessage]:
    """
    Produce a formatted representation of the conversation history.
    """
    if current_channel_id is None:
        current_channel_id = history.current_channel
    if min_date is None:
        min_date = datetime.datetime.now() - from_time_ago

    current_channel = history.get_channel(current_channel_id)
    if not current_channel.messages: return []

    assembled_messages = assemble_messages(
        current_channel.messages,
        current_channel.summaries,
        min_message=min_message,
        max_message=max_message,
        min_date=min_date,
        max_date=max_date,
    )
    
    grouped_messages = group_messages(
        assembled_messages,
        time_gap_reference={
            datetime.datetime.now(): datetime.timedelta(minutes=20), 
            datetime.datetime.now() - datetime.timedelta(hours=2): datetime.timedelta(hours=1),
            datetime.datetime.now() - datetime.timedelta(days=1): datetime.timedelta(days=1)
        }
    )

    display_messages: List[Union[data.StructuredMessage, data.ToolMessage_]] = []
    i = 0
    first_group = True
    for group in grouped_messages:
        
        if not group: continue  # Empty groups should never happen
        
        current_aggregate = []  # Aggregate consecutive Human/System messages
        for message in group:
            
            if isinstance(message, data.AIMessage_):
                
                if current_aggregate: 
                    display_messages.append(
                        _format_grouped_messages(
                            current_aggregate, 
                            current_channel.name, 
                            use_summaries=use_summaries, 
                            show_all_dates=first_group
                        )
                    )
                    current_aggregate = []

                display_messages.extend(message.unpack())  # Also add tool messages if available

            elif isinstance(message, data.Summary):
                current_aggregate.append(message)
                
            else:  # Any other kind of message
                
                if isinstance(message, data.SystemMessage_) and message.lifespan is not None:  # Then check for "dynamic" notifications
                    
                    if "#toolupdated#" in message.content:
                        tool_call_id = message.content[len("#toolupdated#"):]
                        for backtrack_msg in reversed(current_channel.messages):  # Locate tool message
                            if (
                                isinstance(backtrack_msg, data.AIMessage_) 
                                and tool_call_id in backtrack_msg.internal_tool_messages
                            ):
                                tool_message = backtrack_msg.internal_tool_messages[tool_call_id]
                                tool_call, tool_status = backtrack_msg.get_tool_call_info(tool_call_id)
                                message = message.model_copy(deep=True)
                                message.content = (
                                    f"[Tool Updated] {tool_call_id} \"{tool_call['name']}\" "
                                    f"called \"{_time_ago(backtrack_msg.date)}\" with=\"{tool_call['args']}\" "
                                    f"updated to status=\"{tool_status}\" "
                                    f"with update message=\"{tool_message.content}\""
                                )  # Update content to reflect the current state of the tool call

                current_aggregate.append(message)
                
            i += 1
            
        if current_aggregate:
            display_messages.append(
                _format_grouped_messages(
                    current_aggregate, 
                    current_channel.name, 
                    use_summaries=use_summaries, 
                    show_all_dates=first_group
                )
            )

        first_group = False  # After processing the first group, set to False
    
    if not display_messages:
        raise ValueError("No display messages, should never happen because at least one new human message is typically provided when running a new thread.")
    
    # Find channels that were active within the displayed message time span, group the messages accross channels
    # NOTE : Redefining min_date is important because messages older than min_date can be added by assemble_message (because the on min_message constraint is dominant)
    min_date = display_messages[0].date
    active_channels = [c for c in history.channels.values() if c.last_activity > min_date and c.id != current_channel.id]
    
    external_groups: Dict[datetime.datetime, List[Union[data.Summary, data.StructuredMessage]]] = {msg.date: [] for msg in display_messages[:-1]}
    # NOTE : external_groups are indexed by max_date

    for channel in active_channels:
        
        local_external_groups = {msg.date: [] for msg in display_messages[:-1]}  # Store messages that come after the date
        
        assembled_messages = assemble_messages(
            channel.messages,
            channel.summaries,
            min_date=min_date,
            max_date=max_date
        )

        for message in assembled_messages:  # Go through messages by date ascending
            
            candidate_group: Optional[datetime.datetime] = None
            for group_date in local_external_groups.keys():
                # NOTE : At this point message.date is guaranteed to be >= any group_date
                if group_date >= (message.max_date if isinstance(message, data.Summary) else message.date):
                    break  # Encountered a group_date that is greater than the message date => can't fit before
                else:
                    candidate_group = group_date  # <= this group's date (can fit before the key display_message)
            
            if candidate_group:
                local_external_groups[candidate_group].append(message)
        
        for reference_date, group in local_external_groups.items():
            if group:
                message = _format_grouped_messages(group, channel.name, show_all_dates=False)
                message.content = f"From channel {channel.name}\n" + message.content
                external_groups[reference_date].append(message)

    # Insert the groups at the right spot in display_messages
    for i, (reference_date, group) in enumerate(external_groups.items()):
        if group:
            group_message = data.HumanMessage_(
                content=f"Grouped messages from external channels:\n" + "\n".join(msg.content for msg in group),
                date=reference_date,
                author="External Grouped Messages"
            )
            display_messages.insert(i, group_message)
            
    return display_messages
