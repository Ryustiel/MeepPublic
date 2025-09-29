
from typing import (
    Any, Union, Dict, 
    List, Tuple, Literal, 
    Iterator, AsyncIterator, 
    Type, Optional, Callable,
)
from abc import ABC
import pydantic, asyncio, traceback

from langchain_core.tools import BaseTool as LangchainBaseTool
from langchain_core.messages import ToolMessage, ToolMessageChunk, ToolCall

from .utils import run_in_parallel_event_loop, get_or_create_event_loop


# TODO : BaseTool be called with a ToolCall object ;
# TODO : ToolKit would extract ToolCalls from an AIMessage and gather all responses from inner tools


# BaseTool (core component)

class BaseTool(LangchainBaseTool, ABC):
    """
    An extension of the BaseTool class that also interfaces streaming.

    ### How to build a tool (Subclass this):
        1. Specify a str "**name**" and a str "**description**" attribute.
        2. Specify a type "**args_schema**" pydantic model
        that determines the tool's input schema.
        3. Implement some of the following methods 
        -> **_run**, **_arun**, **_stream**, or **_astream**

    ### Notes:
        - Input will be a pydantic model adhering to the provided args_schema.
        - If the stream method is not further implemented in the subclass, 
        streaming will yield a single "response" event. 
        - The asynchronous run (arun) is derived from the synchronous run method 
        by default and vice versa. At least one of those methods 
        should be implemented if using the built-in **raphlib.LLMWithTools**.

    ### Exemple:

        class TemplateTool(BaseTool):
            name: str = "get_information"
            description: str = "Must be run once before replying to the user."
            args_schema: type = pydantic_model_from_options(
                random_fruit_name=str
            )
            
            async def _arun(self, inp: BaseModel) -> str:
                ...
    """
    name: str
    description: str
    args_schema: Optional[Type[pydantic.BaseModel]] = None  # Arg schema if provided should be the only argument of _run methods

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        methods_to_check = ['_stream', '_astream', '_run', '_arun']
        available = []
        for name in methods_to_check:
            base_method = getattr(BaseTool, name)
            subclass_method = getattr(cls, name, None)
            # If the method was overridden (i.e. is a different function)
            if subclass_method is not None and subclass_method is not base_method:
                available.append(name)
        if not available:
            raise NotImplementedError(
                f"Subclass {cls.__name__} must override at least one of {methods_to_check}"
            )
        cls.available_methods = available

    def _extract_tool_call_id(self, mixed_parameters: Optional[Union[str, dict, ToolCall]]) -> str:
        if isinstance(mixed_parameters, dict) and "id" in mixed_parameters.keys():
            return mixed_parameters["id"]
        else:
            return "None"

    def _extract_parameters(self, mixed_parameters: Optional[Union[str, dict, ToolCall]]) -> Optional[pydantic.BaseModel]:
        """
        Turn the parameters into a homogeneous pydantic model to be used by the actual tool methods.
        This function can contain parsing errors.
        """
        if isinstance(mixed_parameters, self.args_schema):  # Mixed parameters is a prebuilt schema
            return mixed_parameters
        elif isinstance(mixed_parameters, str):
            return self.args_schema.model_validate_json(mixed_parameters)
        elif isinstance(mixed_parameters, dict):
            if 'args' in mixed_parameters.keys():
                return self.args_schema.model_validate(mixed_parameters['args'])
            else: # Is a normal dict input
                return self.args_schema.model_validate(mixed_parameters)
        elif isinstance(mixed_parameters, ToolCall):
            return self.args_schema.model_validate(mixed_parameters.to_dict())
        else:
            raise ValueError(f"Unsupported parameters format. Expected str, dict, or ToolCall. Instead got {type(mixed_parameters).__name__}.")

    # ================================================================= DEFAULT BEHAVIOR METHODS

    async def _arun(self, inp: Optional[pydantic.BaseModel] = None, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the tool and return the output or errors
        """
        if "_run" in self.available_methods:
            return self._run(inp=inp, extra_kwargs=extra_kwargs)
        
        else:  # _astream or _stream
            event = None
            async_gen_instance = self._astream(inp=inp, extra_kwargs=extra_kwargs)
            async for event in async_gen_instance:
                pass
            return event

    def _run(self, inp: Optional[pydantic.BaseModel] = None, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:

        if "_arun" in self.available_methods:
            coro = self._arun(inp=inp, extra_kwargs=extra_kwargs)
            if get_or_create_event_loop().is_running():
                return run_in_parallel_event_loop(future=coro)
            else:
                return asyncio.run(main=coro)

        else:  # _stream or _astream
            event = None
            gen_instance = self._stream(inp=inp) if inp else self._stream()
            for event in gen_instance:
                pass
            return event

    async def _astream(self, inp: Optional[pydantic.BaseModel] = None, extra_kwargs: Optional[Dict[str, Any]] = None) -> AsyncIterator[Any]:
        """
        Streams the tool output as stream events, asynchronously (errors, partial responses, full response).
        """
        if "_stream" in self.available_methods:

            gen_instance = self._stream(inp=inp, extra_kwargs=extra_kwargs)
            for event in gen_instance:
                yield event

        else:  # _arun or _run
            coro = self._arun(inp=inp, extra_kwargs=extra_kwargs)
            yield await coro

    def _stream(self, inp: Optional[pydantic.BaseModel] = None, extra_kwargs: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
        """
        Streams the tool output as stream events (errors, partial responses, full response).
        """
        if "_astream" in self.available_methods:

            async_gen_instance = self._astream(inp=inp, extra_kwargs=extra_kwargs)
            try:
                loop = get_or_create_event_loop()
                if loop.is_running():
                    while True: 
                        yield run_in_parallel_event_loop(future=async_gen_instance.__anext__())
                else:
                    while True: 
                        yield loop.run_until_complete(future=async_gen_instance.__anext__())
        
            except StopAsyncIteration:
                pass

        else:  # _run or _arun
            result = self._run(inp=inp) if inp else self._run()
            yield result

    # ================================================================= GATEWAY METHODS

    def _process_output(self, output: Union[str, Tuple[str, ...]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        if isinstance(output, tuple):
            return str(output[0]), output[1]
        return str(output), None

    async def arun(
        self, 
        mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None, 
        extra_kwargs: Optional[Dict[str, Any]] = None
    )  -> ToolMessage:
        """
        Execute the tool and return the output or errors
        """
        
        try:
            inp = self._extract_parameters(mixed_parameters=mixed_parameters) if self.args_schema else None

            result = await self._arun(inp=inp, extra_kwargs=extra_kwargs)
            content, artifact = self._process_output(result)

            return ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "success",
                content = content,
                artifact = artifact
            )
        
        except Exception as e:
            return ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "error",
                content = " ".join(traceback.format_exception(type(e), e, e.__traceback__))
            )
            
    def run(
        self, 
        mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None
    ) -> ToolMessage:
        """
        Execute the tool and return the output or errors.
        """
        try:
            inp = self._extract_parameters(mixed_parameters=mixed_parameters) if self.args_schema else None
            
            result = self._run(inp=inp, extra_kwargs=extra_kwargs)
            content, artifact = self._process_output(result)

            return ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "success",
                content = content,
                artifact = artifact
            )
        
        except Exception as e:
            return ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "error",
                content = " ".join(traceback.format_exception(type(e), e, e.__traceback__))
            )
            
    async def astream(
        self, 
        mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[ToolMessageChunk | ToolMessage]:
        """
        Streams the tool output as stream events, asynchronously (errors, partial responses, full response).
        """
        buffer = "Generator Tool Trace\n<start>\n"
        try:
            inp = self._extract_parameters(mixed_parameters=mixed_parameters) if self.args_schema else None
            stream = self._astream(inp=inp, extra_kwargs=extra_kwargs)

            tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters)

            async for event in stream:

                content, artifact = self._process_output(event)

                yield ToolMessageChunk(
                    tool_call_id = tool_call_id,
                    status = "success",
                    content = content,
                    artifact = artifact
                )
        
        except Exception as e:
            yield ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "error",
                content = " ".join(traceback.format_exception(type(e), e, e.__traceback__))
            )

    def stream(
        self, 
        mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None, 
        extra_kwargs: Optional[Dict[str, Any]] = None
    ) -> Iterator[ToolMessageChunk | ToolMessage]:
        """
        Streams the tool output as stream events (errors, partial responses, full response).
        """
        buffer = "Generator Tool Trace\n<start>\n"
        try:
            inp = self._extract_parameters(mixed_parameters=mixed_parameters) if self.args_schema else None
            stream = self._stream(inp=inp, extra_kwargs=extra_kwargs)
            
            tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters)

            for event in stream:

                content, artifact = self._process_output(event)

                yield ToolMessageChunk(
                    tool_call_id = tool_call_id,
                    status = "success",
                    content = content,
                    artifact = artifact
                )
        
        except Exception as e:
            yield ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "error",
                content = " ".join(traceback.format_exception(type(e), e, e.__traceback__))
            )
            
    def __str__(self):
        if self.args_schema is None:
            return f"\n=== Tool\nName: \n{self.name}\n\nModel: \nNo Input\n\nDescription: \n{self.description}\n=== End Tool"
        else:
            return f"\n=== Tool\nName: \n{self.name}\n\nModel: \n{self.args_schema.__name__} > {self.args_schema.model_fields}\n\nDescription: \n{self.description}\n=== End Tool"
