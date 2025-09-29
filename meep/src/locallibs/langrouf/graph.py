
from typing import Any, Dict, List, TypeVar, Generic, Optional, Union, Callable, Literal, Type, TypedDict

import asyncio, inspect, pydantic

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph
from langgraph.types import Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer

from .edges import EdgeDescriptor

python_next = next  # An alias to the next function

State = TypeVar("State", bound=Union[Dict, pydantic.BaseModel])



class GraphBuilder(Generic[State]):
    """
    An interface for creating LangGraph systems easily.
    """

    def __init__(self, start: str, state: Type[State], checkpointer: Optional[BaseCheckpointSaver] = None):
        """
        Parameters:
            start (str): The name of the start node.
            state (Type[State]): The state type (Pydantic model or TypedDict) for initializing the graph.
        """
        self.__graph = StateGraph(state_schema=state)
        self.__edges: EdgeDescriptor = EdgeDescriptor(start=start)
        self.__checkpointer = checkpointer

    def compiled(self) -> CompiledStateGraph:
        self.__edges.attach_edges(self.__graph)
        compiled_graph = self.__graph.compile(checkpointer=self.__checkpointer)
        return compiled_graph
    
    def subgraph_node(self, graph: CompiledStateGraph, name: str, next: str = "__end__") -> None:
        """
        Add a pre-compiled graph to the current graph.

        Parameters:
            graph (CompiledStateGraph): The pre-compiled graph to add.
            name (str): The name of the node in the current graph.
        """
        self.__edges.add(name, next)
        self.__graph.add_node(node=name, action=graph)
    
    def node(self, name: Optional[str] = None, next: str | List[str] = []) -> Callable[[State], Dict[str, Any]]:
        """
        Add a node to the graph based on the decorated function, which now produces a node update output. 
        Edges will be created based on "next".

        Parameters:
            name (Optional[str]): The name of the node. If not specified, use the name of the function.
            next (List[str]): The nodes this one connects to.
        """
        if isinstance(next, str): next = [next]
        
        if isinstance(name, Callable):  # Decorator mode
            func = name
            name = None
        else:
            func = None


        # ------------------------------------------- Decorator / Inspection


        def decorator(f: Callable):

            if name is None:
                local_name = f.__name__
            else:
                local_name = name

            # Inspect the signature to make sure the State is passed as an argument (or nothing)
            sig = inspect.signature(f)
            if len(sig.parameters) > 1:
                raise ValueError(
                    f"StateGraph nodes can only have one parameter. "
                    f"Got {len(sig.parameters)} params at node \"{local_name}\". Expected 1."
                )
            
            param = python_next(iter(sig.parameters.values()), None)

            if len(sig.parameters) == 0:
                raise ValueError(
                    f"Node \"{local_name}\" must have a state parameter, "
                    f"even if you don't want to modify the state in that node."
                )
            
            is_async = asyncio.iscoroutinefunction(f)
            is_gen = inspect.isgeneratorfunction(f)
            is_asyncgen = inspect.isasyncgenfunction(f)
            
            def build_command(output: None | str | Command | dict) -> Optional[Command]:
                if isinstance(output, dict):
                    return Command(update=output)
                elif type(output).__name__ == "Command":
                    return output
                else:
                    return None
                
            # ---------------------------------------------------- Node Function

            async def node_function(s: State):

                stream_writer = get_stream_writer()
                command = {}

                if is_asyncgen:
                    async for event in f(s):
                        if type(event).__name__ == "Command":
                            command = build_command(event)
                            break
                        else:
                            stream_writer(event)

                elif is_gen:
                    for event in f(s):
                        if type(event).__name__ == "Command":
                            command = build_command(event)
                            break
                        else:
                            stream_writer(event)

                elif is_async:
                    result = await f(s)
                    command = build_command(result)

                else:
                    command = build_command(f(s))

                return command
            

            # ------------------------------------------- Graph Update


            # Add the edge to the graph
            if len(next) == 1:
                self.__edges.add(local_name, next[0])

            # Add appropriate type hints if the node has many edges stemming from it.
            elif len(next) > 1:
                # Graph redirections are handled via Command outputs inside the node, 
                # but the graph schema is built using type hints which have to be added here.
                node_function.__annotations__["return"] = Command[Literal.__getitem__(tuple(next))]

            # Add the node to the graph
            self.__graph.add_node(node=local_name, action=node_function)
            
            return node_function
        

        # ---------------------------------------------------------------- Factory Logic

        
        if func:  # Decorator mode
            return decorator(func)
        else:  # Factory mode
            return decorator
