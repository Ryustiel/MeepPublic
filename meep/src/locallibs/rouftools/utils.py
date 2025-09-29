
from typing import Any

import asyncio, threading

def start_event_loop(loop: asyncio.BaseEventLoop):
    """Run an asyncio event loop in a separate thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create a new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def run_in_parallel_event_loop(future: asyncio.Future) -> Any:
    """
    Run the coroutine in a separate thread and return the result.

    NOTE : This is designed to handle calling async functions from within a synchronous function nested in an asynchronous call.
    (So Async => Sync => Async)
    """
    # Create a new event loop and thread
    loop = asyncio.new_event_loop()  # Keep a reference to the loop for later
    thread = threading.Thread(target=start_event_loop, args=(loop,), daemon=True)
    thread.start()

    try:
        # Submit coroutine to the event loop running in another thread
        future = asyncio.run_coroutine_threadsafe(coro=future, loop=loop)
        result = future.result()  # This blocks until the coroutine completes
    finally:
        # Cleanup: Stop the event loop and join the thread
        loop.call_soon_threadsafe(loop.stop)
        thread.join()  # Waiting for the loop to stop
        
    return result
