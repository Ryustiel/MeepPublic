
import asyncio
import graphs._llm as llm, graphs.processes.vision as vision
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

import locallibs.rouftools as rouftools

TK = rouftools.ToolKit()

@TK.tool
async def experiment(number: int, **kwargs):
    return "output_message", {"special": 3 + number + kwargs["external_input"]}

LLM = llm.CUSTOM.bind_tools(list(TK))

# 1. Tool decorator automatically add kwargs if there are None.
# 2. New Run function that outputs a new data structure and parse output.
# 3. Run function now supports kwargs that can be passed on to the functions.

response1 = LLM.invoke("Run the experiment with input 1")

results = TK.run(response1.tool_calls, extra_kwargs={"external_input": 2})
print(results)
