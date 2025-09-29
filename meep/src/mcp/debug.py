
from typing import List, Literal

import httpx, os, math, asyncio
import mcp.environ

async def perform_action_number_1(parameter: str):
    """Perform action number 1."""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={parameter}&limit=3&appid={os.environ['OPENWEATHER_API_KEY']}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    response.raise_for_status()
    return str(response.json())

async def perform_action_number_2(parameter: int, skip_confirmation: Literal[True] = True):
    """Perform action number 2."""
    return f"Performed the action number 2. The result was {28 * parameter}."

async def perform_action_number_3(param1: float, param2: float):
    """Perform action number 3."""
    url = (
        f"http://api.openweathermap.org/data/2.5/weather?"
        f"lat={param1}&lon={param2}&units=metric&appid={os.environ['OPENWEATHER_API_KEY']}"
    )
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    response.raise_for_status()
    return str(response.json())
 
# (Implement through groupments on the discord thing)
async def perform_action_number_4(parameter: int, requestor: str):
    """Perform action number 4. Requestor is the name of the person making the request."""
    return f"Performed the action number 4 with requestor set as {requestor}. The result was {math.sqrt(2) * parameter}."

import graphs._data as data
async def perform_action_number_5(parameter: int, **kwargs):  # Introspection test
    if 'history' in kwargs:
        history: data.History = kwargs['history']
        channel_id = history.current_channel
        updates = data.InternalUpdates(channel_updates={channel_id: data.InternalChannelUpdates(new_messages=[data.SystemMessage_(author="Introspection", content="Dis mwolo")])})
        return f"The result wuz {parameter * 5} with extra info: {channel_id}", {"updates": updates}
    return f"The result wuz {parameter * 5} with no extra info."

async def perform_action_number_6(parameter: int):
    await asyncio.sleep(8)
    return f"Performed the action number 6. Result is {parameter * 6}."

tools: List[callable] = [
    perform_action_number_1,
    perform_action_number_2,
    perform_action_number_3,
    perform_action_number_4,
    perform_action_number_5,
    perform_action_number_6
]
