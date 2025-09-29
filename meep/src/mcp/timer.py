
from typing import List, Literal

import asyncio


async def setup_reminder(seconds: int, message: str, requestor: str, skip_confirmation: bool = True) -> str:
    """
    Setup a notification that will be reminded to you after a specified number of seconds. 
    Skip confirmation to immediatly setup the reminder
    """

    await asyncio.sleep(seconds)
    return f"Waited for {seconds}. Reminder is {message}"

tools: List[callable] = [
    setup_reminder
]
