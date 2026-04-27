# Multimodal Coding Agent Implementation Plan

This plan outlines the creation of an autonomous yet supervised coding agent. The agent uses a multi-agent graph architecture to handle text, files, and images, with a built-in feedback loop for self-correction and human approval.

## High-Level Architecture

The system is built on **LangGraph** (to manage state and cycles) and uses **Ollama** as the default LLM provider.

### Core Components:
1. **Multimodal LLM Layer**: Uses `llama3.2-vision` (or similar) via Ollama for text/image processing.
2. **State Graph**: Defines the "Code -> Run -> Fix -> Review" loop.
3. **Persistence Layer**: SQLite-based checkpointer to save and resume conversations.
4. **Toolbox**: File system read/write and subprocess execution.
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

# LLM: Ollama (Llama 3.2 Vision)
llm = ChatOllama(model="llama3.2-vision", temperature=0)

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
llm_with_tools = llm.bind_tools(tools)

# --- 4. GRAPH NODES (AGENT LOGIC) ---

def planner_node(state: AgentState):
    """Analyzes requirements (Text + Images) and generates a tool call."""
    messages = state['messages']
    
    # SYSTEM PROMPT: Defines the agent's behavior
    sys_msg = SystemMessage(content=(
        "You are 'Antigravity-lite', an expert self-healing coding assistant. "
        "Use tools to write code, execute it, and fix errors. "
        "If images tags are present in history, analyze them for UI/Logic requirements."
    ))

    # Incorporate images if it's the first time processing them
    response = llm_with_tools.invoke([sys_msg] + messages)
    
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

def router(state: AgentState):
    """Logic to decide: Continue, Loop back for fix, or Stop for Human."""
    last_msg = state['messages'][-1]
    
    if not last_msg.tool_calls:
        return END
    
    # Check if there was an execution error in the last response
    error_detected = any("EXIT_CODE: 0" not in str(m.content) for m in state['messages'] if isinstance(m, ToolMessage) and "EXIT_CODE" in str(m.content))
    
    # If error detected and we haven't looped too much, try fixing
    if error_detected and state['iteration_count'] < 5:
        return "planner"
    
    return "human_gate"

# --- 6. ASSEMBLE THE GRAPH ---

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("tool_executor", tool_executor_node)
builder.add_node("human_gate", human_gate_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "tool_executor")
builder.add_conditional_edges("tool_executor", router, {
    "planner": "planner",
    "human_gate": "human_gate",
    END: END
})
builder.add_edge("human_gate", "planner") # After human resume, it goes back to plan

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
