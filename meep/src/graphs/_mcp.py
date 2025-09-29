"""
A simple local "mock" MCP server.
Tools just asynchronously executed right here locally instead of a remote MCP server.
"""

from typing import Any, Dict, List, Literal, Optional, Self

import os, pydantic, httpx, asyncio, datetime, time, locallibs.rouftools as rouftools
import constants, mcp.environ

__all__ = ['MCPRequest', 'MCPResponse', 'MCPThread', 'MCPClient', 'MCP_CLIENT', 'GLOBAL_TOOKIT', 'get_toolkit']



# ======================================= WAKEUP


async def wakeup(
    channel_id: Optional[str] = None,
    user_name: Optional[str] = None, 
    unless_active_since: datetime.datetime = datetime.datetime.now()
):
    """Send a wakeup call to the assistant."""

    async with httpx.AsyncClient() as client:
        # Schedule run and move on, don't wait for output
        await client.post(
            f"{mcp.environ.LANGGRAPH_SERVER}/threads/{mcp.environ.MEEP_THREAD}/runs",
            json={
                "assistant_id": "meep",
                "checkpoint": {"thread_id": mcp.environ.MEEP_THREAD},
                "config": {"recursion_limit": 50},
                "input": {
                    "wakeup": {
                        "channel_id": channel_id,
                        "user_name": user_name,
                        "unless_active_since": unless_active_since.isoformat()
                    }
                }
            },
            headers = {"Content-Type": "application/json"}
        )
        
async def wakeup_after(
    delay: int,
    channel_id: Optional[str] = None,
    user_name: Optional[str] = None, 
    unless_active_since: datetime.datetime = datetime.datetime.now()
):
    """Send a wakeup call to the assistant unless activity happened since the specified date."""

    await asyncio.sleep(delay)

    await wakeup(
        channel_id=channel_id,
        user_name=user_name,
        unless_active_since=unless_active_since
    )


# ======================================= MCP PROCESS

class MCPRequest(pydantic.BaseModel):
    tool_call: rouftools.ToolCall
    ignore_webhook_on_quick_completion: bool = True
    webhook: Optional[str] = None
    created_at: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    def __hash__(self): return hash(self.tool_call["id"])
    def __eq__(self, other): return self.tool_call["id"] == other.tool_call["id"]

class MCPResponse(pydantic.BaseModel):
    tool_message: rouftools.ToolMessage
    response_time: float
    updates: Optional[pydantic.BaseModel] = None  # Allow introspective tools to modify the state
    status: Literal["processing", "completed", "failed"] = "processing"
    def __hash__(self): return hash(self.tool_message.tool_call_id)
    def __eq__(self, other: Self): return self.tool_message.tool_call_id == other.tool_message.tool_call_id

class MCPThread:
    
    def __init__(self, toolkit: Optional[rouftools.ToolKit] = None):
        self.pending_requests: Dict[MCPRequest, asyncio.Task] = {}
        self.terminal_responses: List[MCPResponse] = []
        self.toolkit = toolkit or rouftools.ToolKit()
        self._lock = asyncio.Lock()

    async def add_request(self, request: MCPRequest, local_context: Optional[dict] = None):
        """
        Add a request to the thread and start processing it.
        """
        async with self._lock:
            task = asyncio.create_task(self._process_request(request, local_context))
            self.pending_requests[request] = task
        
    async def wait_thread_completed(self):
        """
        Wait for all requests in the thread to complete.
        """
        # Lock or not, this wait method should never be called before add_request are called.
        # This is only prevention.
        async with self._lock:
            running_tasks = list(self.pending_requests.values())

        if running_tasks: await asyncio.wait(running_tasks, return_when=asyncio.ALL_COMPLETED)

    async def _process_request(self, request: MCPRequest, local_context: Optional[dict] = None) -> MCPResponse:
        """
        Run the tool call and produce a response.
        """
        await asyncio.sleep(0.1)  # Simulate some processing delay
        start_time = time.monotonic()
        try:
            result_list = await self.toolkit.arun([request.tool_call], extra_kwargs=local_context)
            result = result_list[0]
            response_time = time.monotonic() - start_time
            response = MCPResponse(
                tool_message=result,
                response_time=response_time,
                status="completed" if result.status == "success" else "failed",
                updates=result.artifact["updates"] if result.artifact and isinstance(result.artifact, dict) and "updates" in result.artifact else None
            )
        except Exception as e:
            response_time = time.monotonic() - start_time
            response = MCPResponse(
                tool_message=rouftools.ToolMessage(
                    tool_call_id=request.tool_call["id"],
                    status="error",
                    content="MCP Failed to execute the tool: " + str(e)
                ),
                response_time=response_time,
                status="failed"
            )
        finally:
            async with self._lock:
                self.terminal_responses.append(response)
                del self.pending_requests[request]

            if "requestor" in request.tool_call["args"]:
                await wakeup(
                    user_name=request.tool_call["args"]["requestor"],
                    unless_active_since=datetime.datetime.now()
                )
            else:
                await wakeup(
                    unless_active_since=datetime.datetime.now()
                )  # XXX Attempt waking up at current channel

    async def current_responses(self) -> List[MCPResponse]:
        """
        Get the current responses from the thread.
        This include both completed and processing responses.
        Completed or failed responses will be removed from the queue after retrieval.
        """
        async with self._lock:
            responses = self.terminal_responses.copy()
            pending = list(self.pending_requests.keys())
            self.terminal_responses.clear()

        # Add processing status for pending requests
        for request in pending:
            processing_response = MCPResponse(
                tool_message=rouftools.ToolMessage(
                    tool_call_id=request.tool_call["id"],
                    status="success",
                    content="Tool is being executed on the MCP server, this message will be updated once done."
                ),
                response_time=time.monotonic() - request.created_at.timestamp(),
                status="processing"
            )
            responses.append(processing_response)
        
        return responses


class MCPClient:
    """
    Completed or Failed responses, once retrieved, will be removed from the queue.
    Processing responses are computed dynamically.
    MCPClient runs the tools in a separate task, send responses on a short wait if possible,
    and use a "ask later or send webhook" approach for long-running tasks.
    """
    
    def __init__(self, default_toolkit: Optional[rouftools.ToolKit] = None):
        self.threads: Dict[str, MCPThread] = {}
        self.default_toolkit = default_toolkit

    # NOTE : Thread "ToolKit" will be used by both the MCP executions and the LLM. 
    # A true MCP server would not rely on ToolKits and implement a JSON based interface.

    def get_thread(self, thread_id: str) -> MCPThread:
        if thread_id not in self.threads:
            self.threads[thread_id] = MCPThread(toolkit=self.default_toolkit)
        return self.threads[thread_id]

    async def add_requests(self, thread_id: str, requests: List[MCPRequest], local_context: Optional[Dict[str, Any]] = None):
        """
        Add a list of requests to the thread and start processing them.
        (they're immediately added to the asyncio queue)
        """
        thread = self.get_thread(thread_id)
        for request in requests:
            await thread.add_request(request, local_context=local_context)

    async def get_responses(self, thread_id: str, timeout: float = 0) -> List[MCPResponse]:
        """
        Wait for at most timeout (seconds) or until all requests are processed, 
        then gather completed responses and pending ones.
        Responses should be instant if the no new tool call were added to the thread.
        Completed or failed responses will be removed from the queue after retrieval.
        """
        thread = self.get_thread(thread_id)

        # Wait until either the tasks are done or the timeout occurs
        await asyncio.wait(
            [asyncio.create_task(thread.wait_thread_completed()), asyncio.create_task(asyncio.sleep(timeout))],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Fetch the response messages from the thread
        responses = await thread.current_responses()
        return responses
    
    
# ---------------------- Module Level stuff until we make an external MCP server ----------------------

if True:
    # XXX Create a global MCP client instance with access to all tools.
    # XXX With an actual MCP server the tool schemas would be loaded from the server.
    
    GLOBAL_TOOKIT = rouftools.ToolKit()

    tool_functions: List[callable] = []
    for filename in os.listdir("mcp"):
        if filename.endswith(".py") and not "environ" in filename:
            module_name = f"mcp.{filename[:-3]}"
            module = __import__(module_name, fromlist=[''])
            # Every module should define a `tools` list.
            if hasattr(module, "tools"):
                tool_functions.extend(module.tools)
    
    for tool_function in tool_functions:
        GLOBAL_TOOKIT.tool(tool_function)
        
    MCP_CLIENT = MCPClient(default_toolkit=GLOBAL_TOOKIT)
    
    def get_toolkit(tools: List[str]) -> rouftools.ToolKit:
        """
        Get a ToolKit with only the specified tools.
        XXX : Should be retrieved from the MCP server in a real implementation.
        """
        for tool_name in tools:
            if tool_name not in GLOBAL_TOOKIT.tools:
                raise ValueError(f"Tool {tool_name} not found in the global toolkit.")
            
        return rouftools.ToolKit(
            tools={
                tool_name: GLOBAL_TOOKIT.tools[tool_name] 
                for tool_name in tools 
            }
        )
