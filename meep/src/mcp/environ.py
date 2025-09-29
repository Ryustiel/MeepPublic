"""
Environment variables for local MCP server.
"""

import os

LANGGRAPH_SERVER = os.environ["LANGGRAPH_SERVER_URL"]
MEEP_THREAD = os.environ["MEEP_THREAD_ID"]  # TODO : Move inside the History variable
# or find a way to access thread id from within the graph
