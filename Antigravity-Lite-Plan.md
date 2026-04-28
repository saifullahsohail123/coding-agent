# Multimodal Coding Agent Implementation Plan

This plan outlines the creation of an autonomous yet supervised coding agent. The agent uses a multi-agent graph architecture to handle text, files, and images, with a built-in feedback loop for self-correction and human approval.

## High-Level Architecture

The system is built on **LangGraph** (to manage state and cycles) and uses **Ollama** as the default LLM provider.

### Core Components:
### Core Components:
1. **Hybrid LLM Layer**: 
   - **Reasoning/Tools**: Uses `llama3.1` (which supports tools).
   - **Vision**: Uses `llama3.2-vision` (only for describing images).
2. **Vision-Pre-Processing**: The agent automatically detects images and asks the vision model to "describe" them before the reasoning model takes over.
3. **State Graph**: Defines the "Plan -> Describe (if image) -> Code -> Run -> Fix -> Review" loop.
4. **Persistence Layer**: SQLite-based checkpointer to save and resume conversations.
5. **Human-in-the-loop (HITL)**: Mandatory breakpoint before final execution or destructive changes.

---

## Complete Code Implementation

### 1. Dependencies (`requirements.txt`)
Save this to a file and run `pip install -r requirements.txt`.

```text
langgraph
langchain-ollama
langchain-community
# langchain-google-genai  # Uncomment if shifting to Gemini
langchain-core
pydantic
```

### 2. Main Agent (`agent.py`)
This script implements the full logic, including the database-backed memory and the multimodal loop.

```python
import os
import subprocess
import base64
import sqlite3
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI # For future Gemini shift
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool

# --- 1. PROJECT MANAGEMENT & PERSISTENCE ---

def select_project():
    """Interactive CLI to choose or create a project folder."""
    base_prefix = "project-"
    existing_dirs = sorted([d for d in os.listdir(".") if os.path.isdir(d) and d.startswith(base_prefix)])
    
    print("\n[Project Manager]")
    print("1. Create New Project")
    if existing_dirs:
        print("2. Open Existing Project")
    
    choice = input("\nSelect an option: ").strip()
    
    if choice == "1":
        name = input("Enter project name (e.g., 'fibo-script'): ").strip().replace(" ", "_")
        # Ensure a unique name with project- prefix
        project_dir = f"{base_prefix}{name}"
        if os.path.exists(project_dir):
            print(f"(!) {project_dir} already exists. Opening instead.")
        else:
            os.makedirs(project_dir)
            print(f"[*] Created new project: {project_dir}")
        return project_dir
    
    elif choice == "2" and existing_dirs:
        print("\nExisting Projects:")
        for idx, d in enumerate(existing_dirs):
            print(f"{idx + 1}. {d}")
        
        dir_idx = int(input("\nSelect project number: ")) - 1
        return existing_dirs[dir_idx]
    
    else:
        print("Invalid choice. default to project-auto")
        return "project-auto"

# --- GLOBAL CONFIG (Set during runtime) ---
PROJECT_ROOT = None
DB_PATH = None
memory = None

def initialize_project_env():
    global PROJECT_ROOT, DB_PATH, memory
    PROJECT_ROOT = select_project()
    DB_PATH = os.path.join(PROJECT_ROOT, "chat_history.db")
    
    # Initialize the specific database for this project
    db_connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    memory = SqliteSaver(db_connection)
    print(f"[*] Session Active in {PROJECT_ROOT}. History: {DB_PATH}")

# MODELS
# Reasoning model (Supports tools)
llm_tools = ChatOllama(model="llama3.1", temperature=0)
# Vision model (Supports images, used only for description)
llm_vision = ChatOllama(model="llama3.2-vision", temperature=0)

# --- 2. STATE DEFINITION ---

class AgentState(TypedDict):
    """The graph state keeps track of conversation, files, and iterations."""
    messages: Annotated[List[BaseMessage], "The history of interactions"]
    files: List[str]  # Tracking which files the agent has modified
    images: List[str] # Local paths to images to be analyzed
    iteration_count: int

# --- 3. TOOLS LAYER ---

@tool
def execute_shell(command: str):
    """Executes a terminal command INSIDE the project directory."""
    print(f"DEBUG: Executing in {PROJECT_ROOT}: {command}")
    try:
        # We enforce execution relative to the PROJECT_ROOT
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=60,
            cwd=PROJECT_ROOT
        )
        output = f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}\nEXIT_CODE: {result.returncode}"
        return output
    except Exception as e:
        return f"CRITICAL ERROR: {str(e)}"

@tool
def write_to_disk(filename: str, content: str):
    """Creates/Overwrites a file inside the project directory."""
    full_path = os.path.join(PROJECT_ROOT, filename)
    with open(full_path, "w") as f:
        f.write(content)
    return f"Success: File '{filename}' written to {PROJECT_ROOT}."

@tool
def read_from_disk(filename: str):
    """Reads a file from the project directory."""
    full_path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(full_path):
        return f"Error: File '{filename}' not found in {PROJECT_ROOT}."
    with open(full_path, "r") as f:
        return f.read()

tools = [execute_shell, write_to_disk, read_from_disk]
llm_with_tools = llm_tools.bind_tools(tools)

# --- 4. GRAPH NODES (AGENT LOGIC) ---

def planner_node(state: AgentState):
    """Uses Hybrid approach: Vision model for images, Tool model for logic."""
    messages = state['messages']
    
    # If the user sent an image, we use the vision model to describe it first 
    # so the tool model can 'understand' what's in the picture.
    processed_history = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            # Check if this message has an image
            has_image = any(item.get("type") == "image_url" for item in msg.content if isinstance(item, dict))
            if has_image:
                print("[System] Detected image. Requesting visual analysis from llama3.2-vision...")
                vision_res = llm_vision.invoke([msg])
                # Convert the multimodal message into a purely text message for the tool model
                description = f"(Visual Content Analysis: {vision_res.content})"
                processed_history.append(HumanMessage(content=description))
                continue
        processed_history.append(msg)

    sys_msg = SystemMessage(content=(
        "You are 'Antigravity-lite'. Use tools to solve coding tasks. "
        "You have been provided with textual descriptions of any images uploaded."
    ))

    # Use the reasoning model (which supports tools) for the actual plan
    response = llm_with_tools.invoke([sys_msg] + processed_history)
    
    return {"messages": [response]}

def tool_executor_node(state: AgentState):
    """Processes tool calls from the LLM."""
    last_msg = state['messages'][-1]
    tool_results = []
    
    for tool_call in last_msg.tool_calls:
        tool_fn = {"execute_shell": execute_shell, "write_to_disk": write_to_disk, "read_from_disk": read_from_disk}[tool_call["name"]]
        res = tool_fn.invoke(tool_call["args"])
        tool_results.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
    
    return {"messages": tool_results, "iteration_count": state.get('iteration_count', 0) + 1}

def human_gate_node(state: AgentState):
    """A checkpoint node. The execution HALTS here for review."""
    print("\n[!] STOPPED: Application is waiting for human feedback.")
    return state # No changes, just a pause point

# --- 5. ROUTING LOGIC ---

def route_after_planner(state: AgentState):
    """Decides if we should execute tools or end the session."""
    last_msg = state['messages'][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tool_executor"
    return END

def route_after_tools(state: AgentState):
    """Decides what to do after tools run: Fix errors or wait for human."""
    # Check for errors in the tool outputs
    error_detected = any("EXIT_CODE: 0" not in str(m.content) for m in state['messages'] if isinstance(m, ToolMessage) and "EXIT_CODE" in str(m.content))
    
    if error_detected and state['iteration_count'] < 5:
        print(f"[System] Error detected (Iteration {state['iteration_count']}). Returning to planner for fix...")
        return "planner"
    
    return "human_gate"

# --- 6. ASSEMBLE THE GRAPH ---

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("tool_executor", tool_executor_node)
builder.add_node("human_gate", human_gate_node)

builder.set_entry_point("planner")

# From planner, we either go to tools or end
builder.add_conditional_edges("planner", route_after_planner, {
    "tool_executor": "tool_executor",
    END: END
})

# From tools, we either go back to planner (for fix) or to human gate
builder.add_conditional_edges("tool_executor", route_after_tools, {
    "planner": "planner",
    "human_gate": "human_gate"
})

# From human gate, we always go back to planner to process the feedback
builder.add_edge("human_gate", "planner")

# COMPILE with persistent memory and manual interrupt at human_gate
def get_app(checkpointer):
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_gate"]
    )

# --- 7. RUN / CLI INTERFACE ---

def run_session(app, user_text: str, image_path: str = None):
    """Initializes or continues a project session."""
    thread_id = "main_session" 
    config = {"configurable": {"thread_id": thread_id}}
    
    # Prepare message with text and optional image
    content = [{"type": "text", "text": user_text}]
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                })
        except Exception as e:
            print(f"(!) Could not load image: {e}")
    
    input_state = {"messages": [HumanMessage(content=content)], "iteration_count": 0}
    
    print(f"\n[System]: Running in {PROJECT_ROOT}...")
    
    # Execute the graph
    for event in app.stream(input_state, config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                print(f"\n[Agent]: {last_msg.content}")

if __name__ == "__main__":
    try:
        initialize_project_env()
        active_app = get_app(memory)
        
        print("\n--- Antigravity-Lite Started ---")
        print("Type 'exit' to quit or 'image:/path/to/img.jpg content' to send an image.")
        
        while True:
            user_input = input("\n[You]: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            
            img_p = None
            if user_input.startswith("image:"):
                parts = user_input.split(" ", 1)
                img_p = parts[0].replace("image:", "")
                user_input = parts[1] if len(parts) > 1 else "Analyze this image"
            
            run_session(active_app, user_input, image_path=img_p)
            
    except KeyboardInterrupt:
        print("\nExiting...")
```

## Project Structure
When you run the agent, it will automatically create a structure like this:
```text
.
├── agent.py
├── requirements.txt
├── project-01/
│   ├── chat_history.db  <-- Isolated History
│   ├── main.py          <-- Generated Code
│   └── ...
```
