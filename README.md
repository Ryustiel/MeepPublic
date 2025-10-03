# Meep — Multi-Interface Conversational Platform

Meep is a modular, multi-interface conversational AI built on LangGraph. It can interact with users through different "channels" like Discord and Streamlit, maintaining a unified, persistent state across time and channels. The agent is aware of the conversations it's having in multiple channels. The architecture also decouples the core agent logic from the message input "interfaces", allowing for a scalable and maintainable system.

---

## TLDR

-   **Multi-Interface Interaction:** Chat with "Meep" via Discord or a Streamlit web application simultaneously.
-   **Service-Oriented Architecture:** The system is composed of containerized services (core agent, Discord bot, Streamlit UI) orchestrated with Docker Compose. The interfaces (discord, streamlit) can be replicated.
-   **Advanced State Machine:** Built with LangGraph, Meep uses a complex stateful graph with parallel execution and clear reduce operations that manages conversation history, switches between different "agent activities," and runs background processes like dynamic conversation summarization and URL / media analysis.
-   **Asynchronous Tool Execution:** A MCP (Model Control Protocol) based service handles long-running tool calls asynchronously, preventing the main conversation loop from blocking and using callback mechanisms to return results. This preserves full functionnality during tasks like image generation.
-   **Human-in-the-Loop Confirmation:** Some Tool calls require user approval trigger confirmation flows on both Discord (using message reactions ✅/❌) and Streamlit (using buttons).
-   **Rich Contextual Awareness:** The agent can process URLs to "see" images, read web pages. It maintains separate histories for each channel, enabling context-aware dialogue across different platforms.
-   **Prompt Optimization:** The channel histories and other contextual elements such as explored URLs are dynamically culled (contextually hidden, summarized or grouped) to prevent hallucinations and reduce costs.

---

## Tech Stack and Architectural Patterns

-   **LLM Orchestration:**
    -   **LangGraph:** For defining the core control flow as a state machine.
    -   `langrouf`: A custom, declarative graph builder to simplify LangGraph development.
    -   `rouftools`: A lightweight, decorator-based framework for creating and binding tools to the LLM.
    
    Note : langrouf and rouftools are custom dependencies I'm reusing in other projects, you might want to check them out. I didn't python-package them because they're still in active development.

-   **Backend & Core Logic (`meep` service):**
    -   Python 3.13 with `uv` for package management.
    -   LangChain for LLM integrations (OpenAI, Google Gemini).
    -   FastAPI and `uvicorn` for the Discord callback server.

-   **Interfaces (`discord` & `streamlit` services):**
    -   **Discord.py:** For the full-featured Discord bot.
    -   **Streamlit:** For the interactive web-based chat UI.

-   **Containerization & Deployment:**
    -   **Docker & Docker Compose:** To define, build, and run the multi-service application.

-   **Databases & State Persistence:**
    -   **SQLite (`langgraph-checkpoint-sqlite`):** For persistent, thread-safe checkpointing of the LangGraph state.
    -   **ChromaDB:** As a vector store for semantic search capabilities.
    -   Custom async JSON file-based storage for caching and memory.

-   **Architectural Patterns:**
    -   **Service-Oriented Architecture:** Message processing and generation logic is separated from interfaces which are tailored to a specific messaging service.
    -   **Stateful Graph Execution:** The entire application logic is a state machine which supports langgraph's sophisticated logging, enabling robust and inspectable flows.
    -   **Agent-Based Routing:** An "Activity" system dynamically routes tasks to specialized sub-agents based on conversation context.
    -   **Asynchronous Task Management (MCP Service):** Decouples tool execution from the main agent loop for non-blocking I/O. This was designed to convert easily to its own independent service (MCP Server).
    -   **Human-in-the-Loop (HITL):** Ensures user control over potentially sensitive or costly actions. Discord users are authentified using their true IDs, enabling safe use in public channels.

---

## Highlights

-   **Multi-Interface, Single Brain**
    The `meep` service acts as the central brain, while services like the Discord bot and Streamlit app are treated as distinct "channels." Meep maintains a persistent `History` object that tracks messages, summaries, and metadata for each channel, allowing it to move seamlessly between different contexts, like a "social media user". 

-   **Scalable Service**
    Meep itself (the "brain") can be replicated by using separate and independent graph states (containing `History` objects). The langgraph based server handles this natively with the "thread" mechanism, designed to receive messages and events from a subset of interfaces and update a particular bound graph state.

-   **Sophisticated LangGraph Control Flow**
    The core logic is not a simple chain but a complex graph with parallel execution paths, conditional routing, and sub-graphs. It dynamically decides which processes to run (e.g., `vision` vs. `activity` first) based on user input, and uses an "Activity" state (`conversing`, `debug`, `waiting`) to select the appropriate agent, tools, and prompts.

-   **Asynchronous Mock Control Plane (MCP)**
    To avoid blocking the agent on long-running tasks, tool calls are offloaded to the MCP. This component simulates a remote execution server, running tools in background `asyncio` tasks. For quick tasks, it returns results almost instantly; for longer ones, it relies on a `wakeup` callback mechanism to notify the agent when the results are ready, triggering a new run of the graph.

-   **Persistent, Multi-Channel State**
    Using `langgraph-checkpoint-sqlite`, the entire state of the conversation, including all channels and their histories, is automatically persisted. This ensures that the agent can be stopped and restarted without losing context, and can manage an arbitrary number of independent conversations.

-   **Automated Conversation Management**
    Meep includes background processes for maintaining conversation quality. The `summarize` process automatically condenses parts of the conversation based on age and message density to keep the context window manageable, while the `vision` process enriches the agent's understanding by analyzing content from URLs found in messages.

-   **Declarative Tool and Graph Creation**
    With the custom `locallibs/rouftools` and `locallibs/langrouf` libraries, adding new functionality is streamlined. New tools can be created by simply decorating a Python function, and the graph structure is defined declaratively, making the complex control flow easier to read and maintain.

---

## Installation

### Prerequisites

-   Docker and Docker Compose
-   Python 3.13 (for local development, if needed)
-   API keys for services like OpenAI, LangSmith, Google AI, etc.

### Setup & Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Ryustiel/MeepPublic
    cd https://github.com/Ryustiel/MeepPublic
    ```

2.  **Configure Environment Variables**
    This project uses a centralized `environ/` directory for configuration. You may create and populate some of the following files with your credentials. (only the core agent service is required)

    - `environ/meep.env`: For the core agent service (OpenAI, LangSmith connections).
    ```dotenv
    OPENAI_API_KEY="sk-..."
    
    # Optional logging
    LANGSMITH_TRACING=true
    LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
    LANGSMITH_API_KEY="lsv2_..."
    LANGSMITH_PROJECT="your-project-name"
    
    # These are set to match the docker-compose.yml services by default
    LANGGRAPH_SERVER_URL="http://meep:8080"
    MEEP_THREAD_ID="123e4567-e89b-12d3-a456-426614174000"  # Working example.
    ```

    NOTE : The MEEP_THREAD_ID in the core service is specifically for the internal MCP Service, which is not yet caching the source thread id. Implementing this will be a matter of request flow.
    
    - `environ/discord_1.env`: For the demo instance of the Discord bot (Discord token, callback URL).
    ```dotenv
    DISCORD_TOKEN="..."
    
    # Should point to the exposed port and host of the discord_interface_1 service
    CALLBACK_API_URL="http://<YOUR_HOST_IP_OR_DOMAIN>:8000" 
    
    # Should point to the meep service within the Docker network
    LANGGRAPH_SERVER_URL="http://meep:8080"
    MEEP_THREAD_ID="123e4567-e89b-12d3-a456-426614174000"  # Working example
    ```
    
    - `environ/streamlit_1.env`: For the example minimal Streamlit interface.
    ```dotenv
    # This version does not support callbacks, so no callback urls.
    # Should point to the meep service within the Docker network
    LANGGRAPH_SERVER_URL="http://meep:8080"
    MEEP_THREAD_ID="123e4567-e89b-12d3-a456-426614174000"  # Working example
    ```

    - `environ/credentials/service_account.json`: For Google Cloud services (like Vertex AI).

3.  **Build and Run with Docker Compose**
    This command will build the images for each service and start the entire application stack.
    ```bash
    docker-compose up --build
    ```
    The services will be available at their specified ports:
    -   `meep` (LangGraph Server): `8080`
    -   `discord_interface_1` (FastAPI callbacks): `8000`
    -   `streamlit_interface_1`: `8501`

---

## Configuration

-   **Service Endpoints & URLs:** URLs like `LANGGRAPH_SERVER_URL` and `CALLBACK_API_URL` are defined in the `.env` files and `docker-compose.yml`. Ensure they correctly reference the service names (e.g., `http://meep:8080`).
-   **LLM Models:** The LLM models used for different tasks (decision-making, summarization, conversation) can be configured in `meep/src/graphs/_llm.py`.
-   **Discord Permissions:** Allowed Discord channels and users are hardcoded in `interfaces/discord/src/main.py` in the `CHANNEL_ID_TO_NAME` and `USER_ID_TO_DISCORD_NAME` dictionaries.
-   **Adding New Tools:** To add a new tool, create a Python file in `meep/src/mcp/`, define your function(s), and export them in a `tools` list. The MCP will automatically discover and bind them.

---

## Project Structure

```text
.
├── docker-compose.yml          # Defines and orchestrates the multi-service application.
├── environ/                    # Centralized directory for all environment variables.
│   ├── credentials/
│   │   └── service_account.json
│   ├── discord_1.env
│   ├── meep.env
│   └── streamlit_1.env
├── interfaces/                 # User-facing services (the "channels").
│   ├── discord/                # Discord bot implementation.
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   └── streamlit/              # Streamlit web UI implementation.
│       ├── src/
│       ├── Dockerfile
│       └── pyproject.toml
└── meep/                       # The core agent logic and LangGraph application.
    └── src/
        ├── graphs/             # LangGraph definitions and process logic.
        │   ├── agents/
        │   │   └── meep.py     # Main graph definition, including subgraphs.
        │   ├── processes/      # Nodes implementing specific logic (activity, summarization, etc.).
        │   ├── _data.py        # Pydantic models and reducers for graph state (History, Channel, Messages).
        │   ├── _agents.py      # Metadata and prompt parts for different agent "activities".
        │   └── _mcp.py         # Mock Control Plane for async tool execution.
        ├── locallibs/          # Custom, reusable libraries.
        │   ├── langrouf/       # Declarative GraphBuilder for LangGraph.
        │   └── rouftools/      # Decorator-based toolkit for creating LLM tools.
        ├── mcp/                # Implementations of all available tools to select agents.
        │   ├── debug.py
        │   ├── seiso.py        # Tools for interacting with ChromaDB.
        │   └── ...
        ├── constants.py        # Global constants and thresholds.
        ├── langgraph.json      # LangGraph CLI configuration.
        └── pyproject.toml      # Dependencies for the core meep service.
```
